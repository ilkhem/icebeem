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
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST

from losses.dsm import conditional_dsm, dsm
from models.refinenet_dilated_baseline import RefineNetDilated


def my_collate(batch, nSeg=8):
    modified_batch = []
    for item in batch:
        image, label = item
        if label in range(nSeg):
            modified_batch.append(item)
    return default_collate(modified_batch)


def my_collate_rev(batch, nSeg=8):
    modified_batch = []
    for item in batch:
        image, label = item
        if label in range(nSeg, 10):
            modified_batch.append(item)
    return default_collate(modified_batch)


class PreTrainer:
    """
    This class trains an ICEBEEM or an unconditional EBM on a subset of MNIST, CIFAR, or FMNIST for some specific
    labels.
    For instance, for our transfer learning experiments, we want pretrain an icebeem on labels 0-7.
    This is done using this class.
    As a comparison, we want to train an icebeem on 8-9, for varying subset size, and this class allows that
    """

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.nSeg = config.n_labels
        self.seed = args.seed
        self.subsetSize = args.SubsetSize  # subset size, only for baseline transfer learning, otherwise ignored!
        print('Number of segments: ' + str(self.nSeg))

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self, conditional=True):
        if conditional:
            print('USING CONDITIONAL DSM')

        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets'), train=True, download=True,
                              transform=tran_transform)

        elif self.config.data.dataset == 'MNIST':
            print('RUNNING REDUCED MNIST')
            dataset = MNIST(os.path.join(self.args.run, 'datasets'), train=True, download=True,
                            transform=tran_transform)

        elif self.config.data.dataset == 'FashionMNIST':
            dataset = FashionMNIST(os.path.join(self.args.run, 'datasets'), train=True, download=True,
                                   transform=tran_transform)

        elif self.config.data.dataset == 'MNIST_transferBaseline':
            # use same dataset as transfer_nets.py
            # we can also use the train dataset since the digits are unseen anyway
            dataset = MNIST(os.path.join(self.args.run, 'datasets'), train=False, download=True,
                            transform=test_transform)
            print('TRANSFER BASELINES !! Subset size: ' + str(self.subsetSize))

        elif self.config.data.dataset == 'CIFAR10_transferBaseline':
            # use same dataset as transfer_nets.py
            # we can also use the train dataset since the digits are unseen anyway
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets'), train=False, download=True,
                              transform=test_transform)
            print('TRANSFER BASELINES !! Subset size: ' + str(self.subsetSize))

        elif self.config.data.dataset == 'FashionMNIST_transferBaseline':
            # use same dataset as transfer_nets.py
            # we can also use the train dataset since the digits are unseen anyway
            dataset = FashionMNIST(os.path.join(self.args.run, 'datasets'), train=False, download=True,
                                   transform=test_transform)
            print('TRANSFER BASELINES !! Subset size: ' + str(self.subsetSize))

        else:
            raise ValueError('Unknown config dataset {}'.format(self.config.data.dataset))

        # apply collation
        if self.config.data.dataset in ['MNIST', 'CIFAR10', 'FashionMNIST']:
            collate_helper = lambda batch: my_collate(batch, nSeg=self.nSeg)
            print('Subset size: ' + str(self.subsetSize))
            id_range = list(range(self.subsetSize))
            dataset = torch.utils.data.Subset(dataset, id_range)
            dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_helper)

        elif self.config.data.dataset in ['MNIST_transferBaseline', 'CIFAR10_transferBaseline',
                                          'FashionMNIST_transferBaseline']:
            # trains a model on only digits 8,9 from scratch
            print('Subset size: ' + str(self.subsetSize))
            id_range = list(range(self.subsetSize))
            dataset = torch.utils.data.Subset(dataset, id_range)
            dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=0,
                                    drop_last=True, collate_fn=my_collate_rev)
            print('loaded reduced subset')
        else:
            raise ValueError('Unknown config dataset {}'.format(self.config.data.dataset))

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        # define the g network
        energy_net_finalLayer = torch.ones((self.config.data.image_size * self.config.data.image_size, self.nSeg)).to(
            self.config.device)
        energy_net_finalLayer.requires_grad_()

        # define the f network
        enet = RefineNetDilated(self.config).to(self.config.device)
        enet = torch.nn.DataParallel(enet)

        # training
        optimizer = self.get_optimizer(list(enet.parameters()) + [energy_net_finalLayer])
        step = 0
        loss_track_epochs = []
        for epoch in range(self.config.training.n_epochs):
            loss_vals = []
            for i, (X, y) in enumerate(dataloader):
                step += 1

                enet.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                y -= y.min()  # need to ensure its zero centered !
                if conditional:
                    loss = conditional_dsm(enet, X, y, energy_net_finalLayer, sigma=0.01)
                else:
                    loss = dsm(enet, X, sigma=0.01)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logging.info("step: {}, loss: {}, maxLabel: {}".format(step, loss.item(), y.max()))
                loss_vals.append(loss.item())
                loss_track_epochs.append(loss.item())

                if step >= self.config.training.n_iters:
                    # save final checkpoints for distrubution!
                    states = [
                        enet.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.checkpoints, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.checkpoints, 'checkpoint.pth'))
                    torch.save([energy_net_finalLayer], os.path.join(self.args.checkpoints, 'finalLayerweights_.pth'))
                    pickle.dump(energy_net_finalLayer,
                                open(os.path.join(self.args.checkpoints, 'finalLayerweights.p'), 'wb'))
                    return 0

                if step % self.config.training.snapshot_freq == 0:
                    print('checkpoint at step: {}'.format(step))
                    # save checkpoint for transfer learning! !
                    torch.save([energy_net_finalLayer], os.path.join(self.args.log, 'finalLayerweights_.pth'))
                    pickle.dump(energy_net_finalLayer,
                                open(os.path.join(self.args.log, 'finalLayerweights.p'), 'wb'))
                    states = [
                        enet.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

            if self.config.data.dataset in ['MNIST_transferBaseline', 'CIFAR10_transferBaseline']:
                # save loss track during epoch for transfer baseline
                pickle.dump(loss_vals,
                            open(os.path.join(self.args.run, self.args.dataset + '_Baseline_Size' + str(
                                self.subsetSize) + "_Seed" + str(self.seed) + '.p'), 'wb'))

        if self.config.data.dataset in ['MNIST_transferBaseline', 'CIFAR10_transferBaseline']:
            # save loss track during epoch for transfer baseline
            pickle.dump(loss_track_epochs,
                        open(os.path.join(self.args.run, self.args.dataset + '_Baseline_epochs_Size' + str(
                            self.subsetSize) + "_Seed" + str(self.seed) + '.p'), 'wb'))

        # save final checkpoints for distrubution!
        states = [
            enet.state_dict(),
            optimizer.state_dict(),
        ]
        torch.save(states, os.path.join(self.args.checkpoints, 'checkpoint_{}.pth'.format(step)))
        torch.save(states, os.path.join(self.args.checkpoints, 'checkpoint.pth'))
        torch.save([energy_net_finalLayer], os.path.join(self.args.checkpoints, 'finalLayerweights_.pth'))
        pickle.dump(energy_net_finalLayer,
                    open(os.path.join(self.args.checkpoints, 'finalLayerweights.p'), 'wb'))


def transfer(args, config):
    """
    once an icebeem is pretrained on some labels (0-7), we train only secondary network (g in our manuscript)
    on unseen labels 8-9 (these are new datasets)
    """
    SUBSET_SIZE = args.SubsetSize
    SEED = args.seed
    DATASET = args.dataset.upper()

    print('DATASET: ' + DATASET + ' SUBSET SIZE: ' + str(SUBSET_SIZE) + '\tSEED: ' + str(SEED))

    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    # ckpt_path = os.path.join(args.logs, 'checkpoint_3000.pth')
    states = torch.load(ckpt_path, map_location='cuda:0')
    score = RefineNetDilated(config).to('cuda:0')
    score = torch.nn.DataParallel(score)
    score.load_state_dict(states[0])
    print('loaded energy network')

    # now load in the data
    test_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor()
    ])

    if DATASET == 'MNIST':
        test_dataset = MNIST(os.path.join(args.run, 'datasets'), train=False, download=True, transform=test_transform)
    elif DATASET == 'CIFAR10':
        test_dataset = CIFAR10(os.path.join(args.run, 'datasets'), train=False, download=True, transform=test_transform)
    else:
        raise ValueError('Unknown dataset {}'.format(DATASET))
    id_range = list(range(SUBSET_SIZE))
    dataset = torch.utils.data.Subset(test_dataset, id_range)

    collate_helper = lambda batch: my_collate_rev(batch, nSeg=config.n_labels)
    test_loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=1,
                             drop_last=True, collate_fn=collate_helper)
    print('loaded test data')

    energy_net_finalLayer = torch.ones((config.data.image_size * config.data.image_size, 2)).to(config.device)
    energy_net_finalLayer.requires_grad_()

    # define the optimizer
    parameters = [energy_net_finalLayer]
    optimizer = optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                           betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad)

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

            y = y - y.min()  # make zero indexed for conditional_dsm function
            loss = conditional_dsm(score, X, y, energy_net_finalLayer, sigma=0.01)

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

    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    # ckpt_path = os.path.join(args.logs, 'checkpoint_5000.pth')
    states = torch.load(ckpt_path, map_location='cuda:0')
    score = RefineNetDilated(config).to('cuda:0')
    score = torch.nn.DataParallel(score)
    score.load_state_dict(states[0])
    print('loaded energy network')

    # now load in the data
    test_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor()
    ])

    if args.dataset.lower() == 'mnist':
        test_dataset = MNIST(os.path.join(args.run, 'datasets'), train=False, download=True, transform=test_transform)
    elif args.dataset.lower() == 'cifar10':
        test_dataset = CIFAR10(os.path.join(args.run, 'datasets'), train=False, download=True, transform=test_transform)
    elif args.dataset.lower() in ['fmnist', 'fashionmnist']:
        test_dataset = FashionMNIST(os.path.join(args.run, 'datasets'), train=False, download=True,
                                    transform=test_transform)
    else:
        raise ValueError('Unknown dataset {}'.format(args.dataset))

    collate_helper = lambda batch: my_collate_rev(batch, nSeg=config.n_labels)

    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=1,
                             drop_last=True, collate_fn=collate_helper)
    print('loaded test data')

    representations = np.zeros((10000, config.data.image_size * config.data.image_size * config.data.channels )) # allow for multiple channels and distinct image sizes
    labels = np.zeros((10000,))
    counter = 0
    for i, (X, y) in enumerate(test_loader):
        rep_i = score(X).view(-1, config.data.image_size * config.data.image_size * config.data.channels ).data.cpu().numpy()
        representations[counter:(counter + rep_i.shape[0]), :] = rep_i
        labels[counter:(counter + rep_i.shape[0])] = y.data.numpy()
        counter += rep_i.shape[0]
    representations = representations[:counter, :]
    labels = labels[:counter]
    print('loaded representations')

    labels -= 8
    rep_train, rep_test, lab_train, lab_test = train_test_split(scale(representations), labels, test_size=test_size,
                                                                random_state=config.data.random_state)
    clf = class_model(random_state=0, max_iter=2000).fit(rep_train, lab_train)
    acc = accuracy_score(lab_test, clf.predict(rep_test)) * 100
    print('#' * 10 )
    msg = 'Accuracy of ' + args.baseline * 'unconditional' + (
            1 - args.baseline) * 'transfer' + ' representation: acc={}'.format(np.round(acc, 2))
    print(msg)
    print('#' * 10 )
