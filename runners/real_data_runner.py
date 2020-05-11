import logging
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
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
        rev = True
        test = True
        subset = True
        no_collate = False

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
                                 drop_last=True)

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
        loss_vals = []
        for i, (X, y) in enumerate(dataloader):
            step += 1

            # enet.train()
            energy_net.train()
            X = X.to(config.device)
            X = X / 256. * 255. + torch.rand_like(X) / 256.
            if config.data.logit_transform:
                X = logit_transform(X)

            if conditional:
                loss = cdsm(energy_net, X, y, sigma=0.01)
            else:
                loss = dsm(energy_net, X, sigma=0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info("step: {}, loss: {}, maxLabel: {}".format(step, loss.item(), y.max()))
            loss_vals.append(loss.item())
            loss_track_epochs.append(loss.item())

            if step >= config.training.n_iters:
                enet, energy_net_finalLayer = energy_net.f, energy_net.g
                # save final checkpoints for distrubution!
                states = [
                    enet.state_dict(),
                    optimizer.state_dict(),
                ]
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
            pickle.dump(loss_vals,
                        open(os.path.join(args.run, args.dataset + '_Baseline_Size' + str(
                            args.subsetSize) + "_Seed" + str(args.seed) + '.p'), 'wb'))

    if config.data.dataset.lower() in ['mnist_transferbaseline', 'cifar10_transferbaseline',
                                       'fashionmnist_transferbaseline', 'cifar100_transferbaseline']:
        # save loss track during epoch for transfer baseline
        pickle.dump(loss_track_epochs,
                    open(os.path.join(args.run, args.dataset + '_Baseline_epochs_Size' + str(
                        args.subsetSize) + "_Seed" + str(args.seed) + '.p'), 'wb'))

    # save final checkpoints for distrubution!
    enet, energy_net_finalLayer = energy_net.f, energy_net.g
    states = [
        enet.state_dict(),
        optimizer.state_dict(),
    ]
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
    SUBSET_SIZE = args.SubsetSize
    SEED = args.seed
    DATASET = args.dataset.upper()
    print('DATASET: ' + DATASET + ' SUBSET SIZE: ' + str(SUBSET_SIZE) + '\tSEED: ' + str(SEED))

    # load data
    test_loader, dataset, cond_size = get_dataset(args, config, test=True, rev=True, one_hot=True, subset=True)
    # load the feature network f
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    states = torch.load(ckpt_path, map_location=config.device)
    f = RefineNetDilated(config).to(config.device)
    f.load_state_dict(states[0])
    print('loaded energy network')
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

        pickle.dump(loss_track, open(os.path.join(args.run, DATASET.lower() + 'TransferCDSM_Size' + str(
            SUBSET_SIZE) + "_Seed" + str(SEED) + '.p'), 'wb'))

    pickle.dump(loss_track_epochs, open(os.path.join(args.run, DATASET.lower() + 'TransferCDSM_epochs_Size' + str(
        SUBSET_SIZE) + "_Seed" + str(SEED) + '.p'), 'wb'))


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
    states = torch.load(ckpt_path, map_location=config.device)
    f = RefineNetDilated(config).to(config.device)
    f.load_state_dict(states[0])
    print('loaded energy network')

    representations = np.zeros((10000,
                                config.data.image_size * config.data.image_size * config.data.channels))  # allow for multiple channels and distinct image sizes
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
    print('loaded representations')

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

    DATASET = args.dataset.upper()
    retrain = args.retrainNets

    print('RUNNING REPRESENTATION EXPs ON DATASET: ' + DATASET)
    if args.baseline: print('RUNNING BASELINES')

    # overwrite n_labels to full dataset
    config.n_labels = 10 if config.data.dataset.lower().split('_')[0] != 'cifar100' else 100

    # change random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # this will be used to reload the network later
    ckpt_path = os.path.join(args.run, 'logs', args.doc, 'checkpoint.pth')
    print(ckpt_path)

    print(args)
    if retrain:
        print('training networks ..')
        train(args, config, conditional=conditional)

    # finally, save learnt representations
    states = torch.load(ckpt_path, map_location=config.device)
    f = RefineNetDilated(config).to(config.device)
    # f = torch.nn.DataParallel(f)
    f.load_state_dict(states[0])

    # load data
    test_loader, dataset, cond_size = get_dataset(args, config, test=True, one_hot=False, subset=False, shuffle=False)
    print('loaded test data')

    # allow for multiple channels and distinct image sizes
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

    import pickle
    pickle.dump({'rep': representations, 'lab': labels},
                open(os.path.join(args.run, 'logs', args.doc, 'test_representations.p'), 'wb'))
    print('\ncomputed and saved representations over test data\n')
