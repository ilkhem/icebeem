import os
import pickle
import warnings

import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100

from data.utils import single_one_hot_encode
from losses.dsm import dsm, cdsm
from metrics.mcc import mean_corr_coef, mean_corr_coef_out_of_sample
from models.ebm import ModularUnnormalizedConditionalEBM, ModularUnnormalizedEBM
from models.nets import SimpleLinear
from models.refinenet_dilated import RefineNetDilated


def my_collate(batch, nSeg=8, one_hot=False, total_labels=10):
    modified_batch = []
    for item in batch:
        image, label = item
        if one_hot:
            idx = np.nonzero(label)[0]
            if idx in range(nSeg):
                modified_batch.append((image, label[:nSeg]))
        else:
            if label in range(nSeg):
                modified_batch.append(item)
    return default_collate(modified_batch)


def my_collate_rev(batch, nSeg=8, one_hot=False, total_labels=10):
    modified_batch = []
    for item in batch:
        image, label = item
        if one_hot:
            idx = np.nonzero(label)[0]
            if idx in range(nSeg, total_labels):
                modified_batch.append((image, label[nSeg:]))
        else:
            if label in range(nSeg, total_labels):
                modified_batch.append(item)
    return default_collate(modified_batch)


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
    no_collate = total_labels == config.n_labels
    if config.data.dataset.lower() in ['mnist_transferbaseline', 'cifar10_transferbaseline',
                                       'fashionmnist_transferbaseline', 'cifar100_transferbaseline']:
        print('loading baseline transfer dataset')
        rev = True
        test = True
        subset = True
        no_collate = False
    print('DEBUG: rev {} test {} subset {} no_collate {} one_hot {}'.format(rev, test, subset, no_collate, one_hot))

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

    if one_hot:
        target_transform = lambda label: single_one_hot_encode(label, n_labels=total_labels)
    else:
        target_transform = lambda label: label

    if config.data.dataset.lower().split('_')[0] == 'mnist':
        dataset = MNIST(os.path.join(args.run, 'datasets'), train=not test, download=True,
                        transform=transform, target_transform=target_transform)
    elif config.data.dataset.lower().split('_')[0] in ['fashionmnist', 'fmnist']:
        dataset = FashionMNIST(os.path.join(args.run, 'datasets'), train=not test, download=True,
                               transform=transform, target_transform=target_transform)
    elif config.data.dataset.lower().split('_')[0] == 'cifar10':
        dataset = CIFAR10(os.path.join(args.run, 'datasets'), train=not test, download=True,
                          transform=transform, target_transform=target_transform)
    elif config.data.dataset.lower().split('_')[0] == 'cifar100':
        dataset = CIFAR100(os.path.join(args.run, 'datasets'), train=not test, download=True,
                           transform=transform, target_transform=target_transform)
    else:
        raise ValueError('Unknown config dataset {}'.format(config.data.dataset))

    if not rev:
        collate_helper = lambda batch: my_collate(batch, nSeg=config.n_labels, one_hot=one_hot)
        cond_size = config.n_labels
        drop_last = False
    else:
        collate_helper = lambda batch: my_collate_rev(batch, nSeg=config.n_labels, one_hot=one_hot,
                                                      total_labels=total_labels)
        drop_last = True
        cond_size = total_labels - config.n_labels
    if subset:
        id_range = list(range(args.subsetSize))
        dataset = torch.utils.data.Subset(dataset, id_range)
    if not no_collate:
        dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=0,
                                collate_fn=collate_helper, drop_last=drop_last)
    else:
        dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=0,
                                drop_last=drop_last)

    # print('DEBUG: len(dset) {} type(dse t) {}'.format(len(dataset), type(dataset)))

    return dataloader, dataset, cond_size


def train(args, config, conditional=True):
    # load dataset
    dataloader, dataset, cond_size = get_dataset(args, config, one_hot=True)
    # define the energy model
    if conditional:
        f = RefineNetDilated(config).to(config.device)
        g = SimpleLinear(cond_size, f.output_size, bias=False).to(config.device)
        energy_net = ModularUnnormalizedConditionalEBM(f, g,
                                                       augment=config.model.augment,
                                                       positive=config.model.positive)
    else:
        f = RefineNetDilated(config).to(config.device)
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

            if step >= config.training.n_iters:
                enet, energy_net_finalLayer = energy_net.f, energy_net.g
                # save final checkpoints for distribution!
                states = [
                    enet.state_dict(),
                    optimizer.state_dict(),
                ]
                print('saving weights under: {}'.format(args.checkpoints))
                torch.save(states, os.path.join(args.checkpoints, 'checkpoint_{}.pth'.format(step)))
                torch.save(states, os.path.join(args.checkpoints, 'checkpoint.pth'))
                torch.save([energy_net_finalLayer], os.path.join(args.checkpoints, 'finalLayerweights_.pth'))
                pickle.dump(energy_net_finalLayer,
                            open(os.path.join(args.checkpoints, 'finalLayerweights.p'), 'wb'))
                return 0

            if step % config.training.snapshot_freq == 0:
                enet, energy_net_finalLayer = energy_net.f, energy_net.g
                print('checkpoint at step: {}'.format(step))
                # save checkpoint for transfer learning! !
                torch.save([energy_net_finalLayer], os.path.join(args.log, 'finalLayerweights_.pth'))
                pickle.dump(energy_net_finalLayer,
                            open(os.path.join(args.log, 'finalLayerweights.p'), 'wb'))
                states = [
                    enet.state_dict(),
                    optimizer.state_dict(),
                ]
                torch.save(states, os.path.join(args.log, 'checkpoint_{}.pth'.format(step)))
                torch.save(states, os.path.join(args.log, 'checkpoint.pth'))

        if config.data.dataset.lower() in ['mnist_transferbaseline', 'cifar10_transferbaseline',
                                           'fashionmnist_transferbaseline', 'cifar100_transferbaseline']:
            # save loss track during epoch for transfer baseline
            pickle.dump(loss_track,
                        open(os.path.join(args.output,
                                          'size{}_seed{}.p'.format(args.subsetSize, args.seed)), 'wb'))

    if config.data.dataset.lower() in ['mnist_transferbaseline', 'cifar10_transferbaseline',
                                       'fashionmnist_transferbaseline', 'cifar100_transferbaseline']:
        # save loss track during epoch for transfer baseline
        print('saving loss track under: {}'.format(args.output))
        pickle.dump(loss_track_epochs,
                    open(os.path.join(args.output,
                                      'all_epochs_SIZE{}_SEED{}.p'.format(args.subsetSize, args.seed)), 'wb'))

    # save final checkpoints for distrubution!
    enet, energy_net_finalLayer = energy_net.f, energy_net.g
    states = [
        enet.state_dict(),
        optimizer.state_dict(),
    ]
    print('saving weights under: {}'.format(args.checkpoints))
    torch.save(states, os.path.join(args.checkpoints, 'checkpoint_{}.pth'.format(step)))
    torch.save(states, os.path.join(args.checkpoints, 'checkpoint.pth'))
    torch.save([energy_net_finalLayer], os.path.join(args.checkpoints, 'finalLayerweights_.pth'))
    pickle.dump(energy_net_finalLayer,
                open(os.path.join(args.checkpoints, 'finalLayerweights.p'), 'wb'))


def transfer(args, config):
    """
    once an icebeem is pretrained on some labels (0-7), we train only secondary network (g in our manuscript)
    on unseen labels 8-9 (these are new datasets)
    """
    # load data
    test_loader, dataset, cond_size = get_dataset(args, config, test=True, rev=True, one_hot=True, subset=True)
    # load the feature network f
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)
    f = RefineNetDilated(config).to(config.device)
    f.load_state_dict(states[0])
    # define the feature network g
    g = SimpleLinear(cond_size, f.output_size, bias=False).to(config.device)
    energy_net = ModularUnnormalizedConditionalEBM(f, g, augment=config.model.augment, positive=config.model.positive)
    # define the optimizer
    parameters = energy_net.g.parameters()
    optimizer = get_optimizer(config, parameters)
    # start optimizing!
    step = 0
    eCount = 10
    loss_track_epochs = []
    for epoch in range(eCount):
        print('epoch: ' + str(epoch))
        loss_track = []
        for i, (X, y) in enumerate(test_loader):
            step += 1

            X = X.to(config.device)
            X = X / 256. * 255. + torch.rand_like(X) / 256.

            loss = cdsm(energy_net, X, y, sigma=0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_track.append(loss.item())
            loss_track_epochs.append(loss.item())

        pickle.dump(loss_track,
                    open(os.path.join(args.output,
                                      'size{}_seed{}.p'.format(args.subsetSize, args.seed)), 'wb'))
    print('saving loss track under: {}'.format(args.output))
    pickle.dump(loss_track_epochs,
                open(os.path.join(args.output,
                                  'all_epochs_SIZE{}_SEED{}.p'.format(args.subsetSize, args.seed)), 'wb'))


def semisupervised(args, config):
    """
    after pretraining an icebeem (or unconditional EBM) on labels 0-7, we use the learnt features to classify
    labels in classes 8-9
    """
    class_model = LinearSVC  # LogisticRegression
    test_size = config.data.split_size
    # load data
    test_loader, dataset, cond_size = get_dataset(args, config, test=True, rev=True, one_hot=False, subset=False)
    # load feature network f
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)
    f = RefineNetDilated(config).to(config.device)
    f.load_state_dict(states[0])

    # allow for multiple channels and distinct image sizes
    representations = np.zeros((10000,
                                config.data.image_size * config.data.image_size * config.data.channels))
    labels = np.zeros((10000,))
    counter = 0
    for i, (X, y) in enumerate(test_loader):
        X = X.to(config.device)
        rep_i = f(X).view(-1, config.data.image_size * config.data.image_size * config.data.channels).data.cpu().numpy()
        representations[counter:(counter + rep_i.shape[0]), :] = rep_i
        labels[counter:(counter + rep_i.shape[0])] = y.data.cpu().numpy()
        counter += rep_i.shape[0]
    representations = representations[:counter, :]
    labels = labels[:counter]

    labels -= 8
    rep_train, rep_test, lab_train, lab_test = train_test_split(scale(representations), labels, test_size=test_size,
                                                                random_state=config.data.random_state)
    clf = class_model(random_state=0, max_iter=2000).fit(rep_train, lab_train)
    acc = accuracy_score(lab_test, clf.predict(rep_test)) * 100
    print('#' * 10)
    msg = 'Accuracy of ' + args.baseline * 'unconditional' + (
            1 - args.baseline) * 'transfer' + ' representation: acc={}'.format(np.round(acc, 2))
    print(msg)
    print('#' * 10)


def cca_representations(args, config, conditional=True):
    """
    we train an icebeem model or an unconditional EBM across multiple random seeds and
    compare the reproducibility of representations via CCA

    first we train the entire network, then we save the activations !
    """
    # train the energy model on full train dataset and save feature maps
    train(args, config, conditional=conditional)
    # load test data
    test_loader, dataset, cond_size = get_dataset(args, config, test=True, one_hot=False, subset=False, shuffle=False)
    # load feature mapts
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)
    f = RefineNetDilated(config).to(config.device)
    f.load_state_dict(states[0])

    # compute and save test features
    representations = np.zeros((10000, config.data.image_size * config.data.image_size * config.data.channels))
    labels = np.zeros((10000,))
    counter = 0
    for i, (X, y) in enumerate(test_loader):
        X = X.to(config.device)
        rep_i = f(X).view(-1, config.data.image_size * config.data.image_size * config.data.channels).data.cpu().numpy()
        representations[counter:(counter + rep_i.shape[0]), :] = rep_i
        labels[counter:(counter + rep_i.shape[0])] = y.data.cpu().numpy()
        counter += rep_i.shape[0]
    representations = representations[:counter, :]
    labels = labels[:counter]

    print('saving test representations under: {}'.format(args.checkpoints))
    pickle.dump({'rep': representations, 'lab': labels},
                open(os.path.join(args.checkpoints, 'test_representations.p'), 'wb'))


def plot_representation(args):
    if len(os.listdir(args.output)) < 2:
        # MCC values haven't been computed yet
        # load in trained representations
        res_cond = []
        res_uncond = []
        for f in os.listdir(args.checkpoints):
            print('loading conditional test representations from: {}'.format(f))
            res_cond.append(pickle.load(open(os.path.join(f, 'test_representations.p'), 'rb')))

        for f_baseline in os.listdir(args.checkpoints_baseline):
            print('loading unconditional test representations from: {}'.format(f_baseline))
            res_uncond.append(pickle.load(open(os.path.join(f_baseline, 'test_representations.p'), 'rb')))

        # check things are in correct order
        assert np.max(np.abs(res_cond[0]['lab'] - res_cond[1]['lab'])) == 0
        assert np.max(np.abs(res_uncond[0]['lab'] - res_uncond[1]['lab'])) == 0
        assert np.max(np.abs(res_cond[0]['lab'] - res_uncond[0]['lab'])) == 0

        # now we compare representation identifiability (strong case)
        mcc_strong_cond = []
        mcc_strong_uncond = []
        ii = np.where(res_cond[0]['lab'] < 5)[0]
        iinot = np.where(res_cond[0]['lab'] >= 5)[0]

        for i in range(args.seed, args.nSims):
            for j in range(i + 1, args.nSims):
                mcc_strong_cond.append(
                    mean_corr_coef_out_of_sample(x=res_cond[i]['rep'][ii, :], y=res_cond[j]['rep'][ii, :],
                                                 x_test=res_cond[i]['rep'][iinot, :],
                                                 y_test=res_cond[j]['rep'][iinot, :]))
                mcc_strong_uncond.append(
                    mean_corr_coef_out_of_sample(x=res_uncond[i]['rep'][ii, :], y=res_uncond[j]['rep'][ii, :],
                                                 x_test=res_uncond[i]['rep'][iinot, :],
                                                 y_test=res_uncond[j]['rep'][iinot, :]))
                # mcc_strong_cond.append( mean_corr_coef( res_cond[i]['rep'], res_cond[j]['rep']) )
                # mcc_strong_uncond.append( mean_corr_coef( res_uncond[i]['rep'], res_uncond[j]['rep']) )

        # save results:
        pickle.dump({'mcc_strong_cond': mcc_strong_cond, 'mcc_strong_uncond': mcc_strong_uncond},
                    open(os.path.join(args.output, 'strongMCC.p', 'wb')))

        # no we compare representation identifiability for weaker case

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cutoff = 50 if args.dataset == 'CIFAR100' else 5
        print('Cutoff: {}'.format(cutoff))
        ii = np.where(res_cond[0]['lab'] < cutoff)[0]
        iinot = np.where(res_cond[0]['lab'] >= cutoff)[0]

        mcc_weak_cond = []
        mcc_weak_uncond = []

        cca_dim = 20
        for i in range(args.seed, args.nSims):
            for j in range(i + 1, args.nSims):
                cca = CCA(n_components=cca_dim)
                cca.fit(res_cond[i]['rep'][ii, :], res_cond[j]['rep'][ii, :])

                res = cca.transform(res_cond[i]['rep'][iinot, :], res_cond[j]['rep'][iinot, :])
                mcc_weak_cond.append(mean_corr_coef(res[0], res[1]))

                # now repeat on the baseline!
                ccabase = CCA(n_components=cca_dim)
                ccabase.fit(res_uncond[i]['rep'][ii, :], res_uncond[j]['rep'][ii, :])

                resbase = cca.transform(res_uncond[i]['rep'][iinot, :], res_uncond[j]['rep'][iinot, :])
                mcc_weak_uncond.append(mean_corr_coef(resbase[0], resbase[1]))

        # save results:
        pickle.dump({'mcc_weak_cond': mcc_weak_cond, 'mcc_weak_uncond': mcc_weak_uncond},
                    open(os.path.join(args.output, 'weakMCC.p', 'wb')))
    else:
        mcc_strong = pickle.load(open(os.path.join(args.output, 'strongMCC.p'), 'rb'))
        mcc_strong_cond, mcc_strong_uncond = mcc_strong['mcc_strong_cond'], mcc_strong['mcc_strong_uncond']
        mcc_weak = pickle.load(open(os.path.join(args.output, 'weakMCC.p'), 'rb'))
        mcc_weak_cond, mcc_weak_uncond = mcc_weak['mcc_weak_cond'], mcc_weak['mcc_weak_uncond']

    print('Statistics for strong iden.:\tC\tU')
    print('Mean:\t\t{}\t{}'.format(np.mean(mcc_strong_cond), np.mean(mcc_strong_uncond)))
    print('Median:\t\t{}\t{}'.format(np.median(mcc_strong_cond), np.median(mcc_strong_uncond)))
    print('Std:\t\t{}\t{}'.format(np.std(mcc_strong_cond), np.std(mcc_strong_uncond)))
    cond_sorted = np.sort(mcc_strong_cond)[::-1]
    uncond_sorted = np.sort(mcc_strong_uncond)[::-1]
    print('Top 2:\t\t{}\t{}\n\t\t{}\t{}\nLast 2:\t\t{}\t{}\n\t\t{}\t{}'.format(
        cond_sorted[0], uncond_sorted[0], cond_sorted[1], uncond_sorted[1], cond_sorted[-2], uncond_sorted[-2],
        cond_sorted[-1], uncond_sorted[-1]))

    print('Statistics for weak iden.:\tC\tU')
    print('Mean:\t\t{}\t{}'.format(np.mean(mcc_weak_cond), np.mean(mcc_weak_uncond)))
    print('Median:\t\t{}\t{}'.format(np.median(mcc_weak_cond), np.median(mcc_weak_uncond)))
    print('Std:\t\t{}\t{}'.format(np.std(mcc_weak_cond), np.std(mcc_weak_uncond)))
    cond_sorted = np.sort(mcc_weak_cond)[::-1]
    uncond_sorted = np.sort(mcc_weak_uncond)[::-1]
    print('Top 2:\t\t{}\t{}\n\t\t{}\t{}\nLast 2:\t\t{}\t{}\n\t\t{}\t{}'.format(
        cond_sorted[0], uncond_sorted[0], cond_sorted[1], uncond_sorted[1], cond_sorted[-2], uncond_sorted[-2],
        cond_sorted[-1], uncond_sorted[-1]))

    # plot boxplot
    sns.set_style("whitegrid")
    sns.set_palette('deep')
    data = [mcc_weak_cond, mcc_weak_uncond, mcc_strong_cond, mcc_strong_uncond]
    labels = ['weak cond', 'weak uncond', 'strong cond', 'strong uncond']
    colours = [sns.color_palette()[2], sns.color_palette()[4], sns.color_palette()[2], sns.color_palette()[4]]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, whis=1.5)
    for i in range(len(colours)):
        plt.setp(bp['boxes'][i], color=colours[i])
        plt.setp(bp['whiskers'][2 * i], color=colours[i])
        plt.setp(bp['whiskers'][2 * i + 1], color=colours[i])
        plt.setp(bp['caps'][2 * i], color=colours[i])
        plt.setp(bp['caps'][2 * i + 1], color=colours[i])
        plt.setp(bp['fliers'][i], color=colours[i], marker='D')
    ax.set_xlim(0.5, len(data) + 0.5)
    ax.set_xticklabels(labels, rotation=45, fontsize=9)
    ax.set_ylabel('MCC out of sample')
    ax.set_title('Quality of representations on {}'.format(args.dataset))
    fig.tight_layout()
    plt.savefig(os.path.join(args.run, 'representation_{}.pdf'.format(args.dataset.lower())))


def plot_transfer(args):
    sns.set_style("whitegrid")
    sns.set_palette('deep')

    # collect results for transfer learning
    samplesSizes = [500, 1000, 2000, 3000, 5000, 6000]

    resTransfer = {x: [] for x in samplesSizes}
    resBaseline = {x: [] for x in samplesSizes}

    # load transfer results
    for x in samplesSizes:
        files = [f for f in os.listdir(args.output) if 'size{}'.format(x) in f]
        for f in files:
            resTransfer[x].append(np.median(pickle.load(open(os.path.join(args.output, f), 'rb'))))

        files = [f for f in os.listdir(args.output_baseline) if 'size{}'.format(x) in f]
        for f in files:
            resBaseline[x].append(np.median(pickle.load(open(os.path.join(args.output_baseline, f), 'rb'))))

        print(
            'Transfer: ' + str(np.median(resTransfer[x]) * 1e4) + '\tBaseline: ' + str(np.median(resBaseline[x]) * 1e4))

    resTsd = np.array([np.std(resTransfer[x]) * 1e4 for x in samplesSizes])

    resT = np.array([np.median(resTransfer[x]) * 1e4 for x in samplesSizes])
    resBas = np.array([np.median(resBaseline[x]) * 1e4 for x in samplesSizes])

    f, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
    ax1.plot(samplesSizes, resT, label='Transfer', linewidth=2, color=sns.color_palette()[2])
    ax1.fill_between(samplesSizes, resT + 2 * resTsd, resT - 2 * resTsd, alpha=.25, color=sns.color_palette()[2])
    ax1.plot(samplesSizes, resBas, label='Baseline', linewidth=2, color=sns.color_palette()[4])
    ax1.legend()
    ax1.set_xlabel('Train dataset size')
    ax1.set_ylabel('DSM Objective (scaled)')
    ax1.set_title('Conditional DSM Objective')
    f.tight_layout()
    plt.savefig(os.path.join(args.run, 'transfer_{}.pdf'.format(args.dataset.lower())))
