import os
import pickle

import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100

from data.utils import single_one_hot_encode, single_one_hot_encode_rev
from losses.dsm import dsm, cdsm
from metrics.mcc import mean_corr_coef, mean_corr_coef_out_of_sample
from models.ebm import ModularUnnormalizedConditionalEBM, ModularUnnormalizedEBM
from models.nets import ConvMLP, FullMLP, SimpleLinear
from models.refinenet_dilated import RefineNetDilated


def feature_net(config):
    if config.model.architecture.lower() == 'convmlp':
        return ConvMLP(config)
    elif config.model.architecture.lower() == 'mlp':
        return FullMLP(config)
    elif config.model.architecture.lower() == 'unet':
        return RefineNetDilated(config)


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def get_dataset(args, config, test=False, rev=False, one_hot=True, subset=False, shuffle=True):
    total_labels = 10 if config.data.dataset.lower().split('_')[0] != 'cifar100' else 100
    reduce_labels = total_labels != config.n_labels
    if config.data.dataset.lower() in ['mnist_transferbaseline', 'cifar10_transferbaseline',
                                       'fashionmnist_transferbaseline', 'cifar100_transferbaseline']:
        print('loading baseline transfer dataset')
        rev = True
        test = False
        subset = True
        reduce_labels = True

    if config.data.random_flip is False:
        transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        if not test:
            transform = transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()
            ])

    if config.data.dataset.lower().split('_')[0] == 'mnist':
        dataset = MNIST(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform)
    elif config.data.dataset.lower().split('_')[0] in ['fashionmnist', 'fmnist']:
        dataset = FashionMNIST(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform)
    elif config.data.dataset.lower().split('_')[0] == 'cifar10':
        dataset = CIFAR10(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform)
    elif config.data.dataset.lower().split('_')[0] == 'cifar100':
        dataset = CIFAR100(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform)
    else:
        raise ValueError('Unknown config dataset {}'.format(config.data.dataset))

    if type(dataset.targets) is list:
        # CIFAR10 and CIFAR100 store targets as list, unlike (F)MNIST which uses torch.Tensor
        dataset.targets = np.array(dataset.targets)

    if not rev:
        labels_to_consider = np.arange(config.n_labels)
        target_transform = lambda label: single_one_hot_encode(label, n_labels=config.n_labels)
        cond_size = config.n_labels

    else:
        labels_to_consider = np.arange(config.n_labels, total_labels)
        target_transform = lambda label: single_one_hot_encode_rev(label, start_label=config.n_labels,
                                                                   n_labels=total_labels)
        cond_size = total_labels - config.n_labels
    if reduce_labels:
        idx = np.any([np.array(dataset.targets) == i for i in labels_to_consider], axis=0).nonzero()
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    if one_hot:
        dataset.target_transform = target_transform
    if subset and args.subset_size != 0:
        dataset = torch.utils.data.Subset(dataset, np.arange(args.subset_size))
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=0)

    return dataloader, dataset, cond_size


def train(args, config, conditional=True):
    save_weights = 'baseline' not in config.data.dataset.lower()  # we don't need the
    if args.subset_size == 0:
        conditional = False
    # load dataset
    dataloader, dataset, cond_size = get_dataset(args, config, one_hot=True)
    # define the energy model
    if conditional:
        f = feature_net(config).to(config.device)
        g = SimpleLinear(cond_size, f.output_size, bias=False).to(config.device)
        energy_net = ModularUnnormalizedConditionalEBM(f, g,
                                                       augment=config.model.augment,
                                                       positive=config.model.positive)
    else:
        f = feature_net(config).to(config.device)
        energy_net = ModularUnnormalizedEBM(f)
    # get optimizer
    optimizer = get_optimizer(config, energy_net.parameters())
    # train
    step = 0
    loss_track_epochs = []
    for epoch in range(config.training.n_epochs):
        loss_track = []
        for i, (X, y) in enumerate(dataloader):
            step += 1
            energy_net.train()
            X = X.to(config.device)
            X = X / 256. * 255. + torch.rand_like(X) / 256.
            if config.data.logit_transform:
                X = logit_transform(X)
            # compute loss
            if conditional:
                loss = cdsm(energy_net, X, y, sigma=0.01)
            else:
                loss = dsm(energy_net, X, sigma=0.01)
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_track.append(loss.item())
            loss_track_epochs.append(loss.item())

            if step >= config.training.n_iters and save_weights:
                enet, energy_net_finalLayer = energy_net.f, energy_net.g
                # save final checkpoints for distribution!
                states = [
                    enet.state_dict(),
                    optimizer.state_dict(),
                ]
                print('saving weights under: {}'.format(args.checkpoints))
                # torch.save(states, os.path.join(args.checkpoints, 'checkpoint_{}.pth'.format(step)))
                torch.save(states, os.path.join(args.checkpoints, 'checkpoint.pth'))
                torch.save([energy_net_finalLayer], os.path.join(args.checkpoints, 'finalLayerweights_.pth'))
                pickle.dump(energy_net_finalLayer,
                            open(os.path.join(args.checkpoints, 'finalLayerweights.p'), 'wb'))
                return 0

            if step % config.training.snapshot_freq == 0:
                enet, energy_net_finalLayer = energy_net.f, energy_net.g
                print('checkpoint at step: {}'.format(step))
                # save checkpoint for transfer learning! !
                # torch.save([energy_net_finalLayer], os.path.join(args.log, 'finalLayerweights_.pth'))
                # pickle.dump(energy_net_finalLayer,
                #             open(os.path.join(args.log, 'finalLayerweights.p'), 'wb'))
                # states = [
                #     enet.state_dict(),
                #     optimizer.state_dict(),
                # ]
                # torch.save(states, os.path.join(args.log, 'checkpoint_{}.pth'.format(step)))
                # torch.save(states, os.path.join(args.log, 'checkpoint.pth'))

        if config.data.dataset.lower() in ['mnist_transferbaseline', 'cifar10_transferbaseline',
                                           'fashionmnist_transferbaseline', 'cifar100_transferbaseline']:
            # save loss track during epoch for transfer baseline
            pickle.dump(loss_track,
                        open(os.path.join(args.output,
                                          'size{}_seed{}.p'.format(args.subset_size, args.seed)), 'wb'))

    if config.data.dataset.lower() in ['mnist_transferbaseline', 'cifar10_transferbaseline',
                                       'fashionmnist_transferbaseline', 'cifar100_transferbaseline']:
        # save loss track during epoch for transfer baseline
        print('saving loss track under: {}'.format(args.output))
        pickle.dump(loss_track_epochs,
                    open(os.path.join(args.output,
                                      'all_epochs_SIZE{}_SEED{}.p'.format(args.subset_size, args.seed)), 'wb'))

    # save final checkpoints for distrubution!
    if save_weights:
        enet, energy_net_finalLayer = energy_net.f, energy_net.g
        states = [
            enet.state_dict(),
            optimizer.state_dict(),
        ]
        print('saving weights under: {}'.format(args.checkpoints))
        # torch.save(states, os.path.join(args.checkpoints, 'checkpoint_{}.pth'.format(step)))
        torch.save(states, os.path.join(args.checkpoints, 'checkpoint.pth'))
        torch.save([energy_net_finalLayer], os.path.join(args.checkpoints, 'finalLayerweights_.pth'))
        pickle.dump(energy_net_finalLayer,
                    open(os.path.join(args.checkpoints, 'finalLayerweights.p'), 'wb'))


def transfer(args, config):
    """
    once an icebeem is pretrained on some labels (0-7), we train only secondary network (g in our manuscript)
    on unseen labels 8-9 (these are new datasets)
    """
    conditional = args.subset_size != 0
    # load data
    dataloader, dataset, cond_size = get_dataset(args, config, test=False, rev=True, one_hot=True, subset=True)
    # load the feature network f
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)
    f = feature_net(config).to(config.device)
    f.load_state_dict(states[0])
    if conditional:
        # define the feature network g
        g = SimpleLinear(cond_size, f.output_size, bias=False).to(config.device)
        energy_net = ModularUnnormalizedConditionalEBM(f, g, augment=config.model.augment,
                                                       positive=config.model.positive)
        # define the optimizer
        parameters = energy_net.g.parameters()
        optimizer = get_optimizer(config, parameters)
    else:
        # no learning is involved: just evaluate f on the new labels, with g = 1
        energy_net = ModularUnnormalizedEBM(f)
        optimizer = None
    # start optimizing!
    eCount = 10
    loss_track_epochs = []
    for epoch in range(eCount):
        print('epoch: ' + str(epoch))
        loss_track = []
        for i, (X, y) in enumerate(dataloader):
            X = X.to(config.device)
            X = X / 256. * 255. + torch.rand_like(X) / 256.
            if conditional:
                loss = cdsm(energy_net, X, y, sigma=0.01)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # just evaluate the DSM loss using the pretarined f --- no learning
                loss = dsm(energy_net, X, sigma=0.01)
                loss.backward()  # strangely, without this line, the script requires twice as much GPU memory
            loss_track.append(loss.item())
            loss_track_epochs.append(loss.item())

        pickle.dump(loss_track,
                    open(os.path.join(args.output,
                                      'size{}_seed{}.p'.format(args.subset_size, args.seed)), 'wb'))
    print('saving loss track under: {}'.format(args.output))
    pickle.dump(loss_track_epochs,
                open(os.path.join(args.output,
                                  'all_epochs_SIZE{}_SEED{}.p'.format(args.subset_size, args.seed)), 'wb'))


def semisupervised(args, config):
    """
    after pretraining an icebeem (or unconditional EBM) on labels 0-7, we use the learnt features to classify
    labels in classes 8-9
    """
    from warnings import filterwarnings
    filterwarnings('ignore')
    class_model = LinearSVC  # LogisticRegression
    test_size = config.data.split_size
    # load data
    dataloader, dataset, cond_size = get_dataset(args, config, test=True, rev=True, one_hot=False, subset=False)
    # load feature network f
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)
    f = feature_net(config).to(config.device)
    f.load_state_dict(states[0])

    accs = []
    for i in range(5):
        representations = np.zeros((10000, f.output_size))
        labels = np.zeros((10000,))
        counter = 0
        for i, (X, y) in enumerate(dataloader):
            X = X.to(config.device)
            rep_i = f(X).view(-1, f.output_size).data.cpu().numpy()
            representations[counter:(counter + rep_i.shape[0])] = rep_i
            labels[counter:(counter + rep_i.shape[0])] = y.data.cpu().numpy()
            counter += rep_i.shape[0]
        representations = representations[:counter]
        labels = labels[:counter]

        labels -= config.n_labels
        rep_train, rep_test, lab_train, lab_test = train_test_split(scale(representations), labels, test_size=test_size,
                                                                    random_state=config.data.random_state)
        clf = class_model(random_state=0, max_iter=2000).fit(rep_train, lab_train)
        acc = accuracy_score(lab_test, clf.predict(rep_test)) * 100
        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print('#' * 10)
    msg = 'Accuracy of ' + args.baseline * 'unconditional' + (
            1 - args.baseline) * 'transfer' + ' representation: acc= {} \pm {}'.format(np.round(mean_acc, 2),
                                                                                       np.round(std_acc, 2))
    print(msg)
    print('#' * 10)


def compute_representations(args, config, conditional=True):
    """
    we train an icebeem model or an unconditional EBM across multiple random seeds and
    compare the reproducibility of representations via CCA

    first we train the entire network, then we save the activations !
    """
    # train the energy model on full train dataset and save feature maps
    train(args, config, conditional=conditional)
    # load test data
    dataloader, dataset, cond_size = get_dataset(args, config, test=True, one_hot=False, subset=False, shuffle=False)
    # load feature mapts
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)
    f = feature_net(config).to(config.device)
    f.load_state_dict(states[0])

    # compute and save test features
    representations = np.zeros((10000, f.output_size))
    labels = np.zeros((10000,))
    counter = 0
    for i, (X, y) in enumerate(dataloader):
        X = X.to(config.device)
        rep_i = f(X).view(-1, f.output_size).data.cpu().numpy()
        representations[counter:(counter + rep_i.shape[0])] = rep_i
        labels[counter:(counter + rep_i.shape[0])] = y.data.cpu().numpy()
        counter += rep_i.shape[0]
    representations = representations[:counter]
    labels = labels[:counter]

    print('saving test representations under: {}'.format(args.checkpoints))
    pickle.dump({'rep': representations, 'lab': labels},
                open(os.path.join(args.checkpoints, 'test_representations.p'), 'wb'))


def compute_mcc(args, config):
    rep1 = pickle.load(
        open(os.path.join(args.checkpoints, 'seed{}'.format(args.seed), 'test_representations.p'), 'rb'))['rep']
    rep2 = pickle.load(
        open(os.path.join(args.checkpoints, 'seed{}'.format(args.second_seed), 'test_representations.p'), 'rb'))[
        'rep']

    # cutoff = 50 if args.dataset == 'CIFAR100' else 5
    # ii = np.where(res_cond[0]['lab'] < cutoff)[0]  # in sample points to learn from
    # iinot = np.where(res_cond[0]['lab'] >= cutoff)[0]  # out of sample points
    cutoff = 5000  # half the test dataset
    ii = np.arange(cutoff)
    iinot = np.arange(cutoff, 2 * cutoff)

    mcc_strong_out = mean_corr_coef_out_of_sample(x=rep1[ii], y=rep2[ii], x_test=rep1[iinot], y_test=rep2[iinot])
    mcc_strong_in = (mean_corr_coef(x=rep1[ii], y=rep2[ii]))

    pickle.dump({'in': mcc_strong_in, 'out': mcc_strong_out},
                open(os.path.join(args.output, 'mcc_strong_{}_{}.p'.format(args.seed, args.second_seed)), 'wb'))

    cca_dim = 20
    cca = CCA(n_components=cca_dim)
    cca.fit(rep1[ii], rep2[ii])
    res_out = cca.transform(rep1[iinot], rep2[iinot])
    mcc_weak_out = mean_corr_coef(res_out[0], res_out[1])
    res_in = cca.transform(rep1[ii], rep2[ii])
    mcc_weak_in = mean_corr_coef(res_in[0], res_in[1])

    pickle.dump({'in': mcc_weak_in, 'out': mcc_weak_out},
                open(os.path.join(args.output, 'mcc_weak_{}_{}.p'.format(args.seed, args.second_seed)), 'wb'))


def plot_representation(args, config):
    max_seed = max_seed_baseline = args.n_sims

    mcc_strong_cond_in = []
    mcc_strong_cond_out = []
    mcc_weak_cond_in = []
    mcc_weak_cond_out = []
    for i in range(args.seed, max_seed):
        for j in range(i + 1, max_seed):
            temp = pickle.load(open(os.path.join(args.output, 'mcc_strong_{}_{}.p'.format(i, j)), 'rb'))
            mcc_strong_cond_in.append(temp['in'])
            mcc_strong_cond_out.append(temp['out'])
            temp = pickle.load(open(os.path.join(args.output, 'mcc_weak_{}_{}.p'.format(i, j)), 'rb'))
            mcc_weak_cond_in.append(temp['in'])
            mcc_weak_cond_out.append(temp['out'])
    mcc_strong_uncond_in = []
    mcc_strong_uncond_out = []
    mcc_weak_uncond_in = []
    mcc_weak_uncond_out = []
    for i in range(args.seed, max_seed_baseline):
        for j in range(i + 1, max_seed_baseline):
            temp = pickle.load(open(os.path.join(args.output_baseline, 'mcc_strong_{}_{}.p'.format(i, j)), 'rb'))
            mcc_strong_uncond_in.append(temp['in'])
            mcc_strong_uncond_out.append(temp['out'])
            temp = pickle.load(open(os.path.join(args.output_baseline, 'mcc_weak_{}_{}.p'.format(i, j)), 'rb'))
            mcc_weak_uncond_in.append(temp['in'])
            mcc_weak_uncond_out.append(temp['out'])

    def _print_stats(res_cond, res_uncond, title=''):
        print('Statistics for {}:\tC\tU'.format(title))
        print('Mean:\t\t{}\t{}'.format(np.mean(res_cond), np.mean(res_uncond)))
        print('Median:\t\t{}\t{}'.format(np.median(res_cond), np.median(res_uncond)))
        print('Std:\t\t{}\t{}'.format(np.std(res_cond), np.std(res_uncond)))
        cond_sorted = np.sort(res_cond)[::-1]
        uncond_sorted = np.sort(res_uncond)[::-1]
        print('Top 2:\t\t{}\t{}\n\t\t{}\t{}\nLast 2:\t\t{}\t{}\n\t\t{}\t{}'.format(
            cond_sorted[0], uncond_sorted[0], cond_sorted[1], uncond_sorted[1], cond_sorted[-2], uncond_sorted[-2],
            cond_sorted[-1], uncond_sorted[-1]))

    def _boxplot(res_strong_cond, res_strong_uncond, res_weak_cond, res_weak_uncond, ylabel='in sample', ext='in'):
        sns.set_style("whitegrid")
        sns.set_palette('deep')
        capsprops = whiskerprops = boxprops = dict(linewidth=2)
        medianprops = dict(linewidth=2, color='firebrick')
        data = [res_weak_cond, res_weak_uncond, res_strong_cond, res_strong_uncond]
        labels = ['ICE-BeeM\nWeak', 'Baseline\nWeak', 'ICE-BeeM\nStrong', 'Baseline\nStrong']
        colours = [sns.color_palette()[2], sns.color_palette()[4], sns.color_palette()[2], sns.color_palette()[4]]
        fig, ax = plt.subplots(figsize=(4, 4))
        bp = ax.boxplot(data, whis=1.5, showfliers=False, boxprops=boxprops, capprops=capsprops,
                        whiskerprops=whiskerprops, medianprops=medianprops)
        for i in range(len(colours)):
            plt.setp(bp['boxes'][i], color=colours[i])
            plt.setp(bp['whiskers'][2 * i], color=colours[i])
            plt.setp(bp['whiskers'][2 * i + 1], color=colours[i])
            plt.setp(bp['caps'][2 * i], color=colours[i])
            plt.setp(bp['caps'][2 * i + 1], color=colours[i])
            # plt.setp(bp['fliers'][i], color=colours[i], marker='D')
        ax.set_xlim(0.5, len(data) + 0.5)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('MCC {}'.format(ylabel))
        ax.set_title('Identifiability of representations')
        fig.tight_layout()
        file_name = 'representation_'
        if config.model.positive:
            file_name += 'p_'
        if config.model.augment:
            file_name += 'a_'
        if config.model.final_layer:
            file_name += str(config.model.feature_size) + '_'
        plt.savefig(os.path.join(args.run, '{}{}_{}.pdf'.format(file_name, args.dataset.lower(), ext)))

    # print some statistics
    _print_stats(mcc_strong_cond_in, mcc_strong_uncond_in, title='strong iden. in sample')
    _print_stats(mcc_strong_cond_out, mcc_strong_uncond_out, title='strong iden. out of sample')
    _print_stats(mcc_weak_cond_in, mcc_weak_uncond_in, title='weak iden. in sample')
    _print_stats(mcc_weak_cond_out, mcc_weak_uncond_out, title='weak iden. out of sample')
    # boxplot
    _boxplot(mcc_strong_cond_in, mcc_strong_uncond_in, mcc_weak_cond_in, mcc_weak_uncond_in,
             ylabel='in sample',
             ext='in_{}_{}_{}'.format(max_seed, max_seed_baseline, config.model.architecture.lower()))
    _boxplot(mcc_strong_cond_out, mcc_strong_uncond_out, mcc_weak_cond_out, mcc_weak_uncond_out,
             ylabel='out of sample',
             ext='out_{}_{}_{}'.format(max_seed, max_seed_baseline, config.model.architecture.lower()))


def plot_transfer(args, config):
    sns.set_style("whitegrid")
    sns.set_palette('deep')

    # collect results for transfer learning
    samplesSizes = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]

    resTransfer = {x: [] for x in samplesSizes}
    resBaseline = {x: [] for x in samplesSizes}

    # load transfer results
    for x in samplesSizes:
        files = [f for f in os.listdir(args.output) if 'SIZE{}'.format(x) in f]
        f_temp = [f for f in os.listdir(args.output) if 'size{}'.format(x) in f][0]
        x_len = len(pickle.load(open(os.path.join(args.output, f_temp), 'rb')))
        for f in files:
            resTransfer[x].append(np.median(pickle.load(open(os.path.join(args.output, f), 'rb'))[-x_len:]))

        files = [f for f in os.listdir(args.output_baseline) if 'SIZE{}'.format(x) in f]
        f_temp = [f for f in os.listdir(args.output_baseline) if 'size{}'.format(x) in f][0]
        x_len = len(pickle.load(open(os.path.join(args.output_baseline, f_temp), 'rb')))
        for f in files:
            resBaseline[x].append(np.median(pickle.load(open(os.path.join(args.output_baseline, f), 'rb'))[-x_len:]))

        print(
            'Transfer: ' + str(np.median(resTransfer[x]) * 1e4) + '\tBaseline: ' + str(np.median(resBaseline[x]) * 1e4))

    config_type = config.model.architecture + '-' + str(
        config.model.feature_size) * config.model.final_layer + 'p' * config.model.positive + 'a' * config.model.augment
    tex_msg = '{} & \emph{{{}}} & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\'
    print(tex_msg.format(
        config.data.dataset, config_type,
        1e4 * np.mean(resTransfer[6000]), 1e4 * np.std(resTransfer[6000]),
        1e4 * np.mean(resTransfer[0]), 1e4 * np.std(resTransfer[0]),
        1e4 * np.mean(resBaseline[6000]), 1e4 * np.std(resBaseline[6000]),
        1e4 * np.mean(resBaseline[0]), 1e4 * np.std(resBaseline[0])))

    samplesSizes.remove(0)
    resTsd = np.array([np.std(resTransfer[x]) * 1e4 for x in samplesSizes])
    resT = np.array([np.median(resTransfer[x]) * 1e4 for x in samplesSizes])
    resBas = np.array([np.median(resBaseline[x]) * 1e4 for x in samplesSizes])

    f, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
    ax1.plot(samplesSizes, resT, marker='v', label='Transfer', linewidth=2, color=sns.color_palette()[2])
    ax1.fill_between(samplesSizes, resT + 2 * resTsd, resT - 2 * resTsd, alpha=.25, color=sns.color_palette()[2])
    ax1.plot(samplesSizes, resBas, marker='o', label='Baseline', linewidth=2, color=sns.color_palette()[4])
    ax1.legend()
    ax1.set_xlabel('Train dataset size')
    ax1.set_ylabel('CDSM Objective (scaled)')
    ax1.set_title('Transfer learning')
    f.tight_layout()
    file_name = 'transfer_'
    if config.model.positive:
        file_name += 'p_'
    if config.model.augment:
        file_name += 'a_'
    if config.model.final_layer:
        file_name += str(config.model.feature_size) + '_'
    plt.savefig(os.path.join(args.run,
                             '{}{}_{}.pdf'.format(file_name, args.dataset.lower(), config.model.architecture.lower())))
