### transfer learning on estimated networks !
#
#

import os
import pickle
from torch import optim
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import MNIST, CIFAR10

from losses.dsm import conditional_dsm
from models.refinenet_dilated_baseline import RefineNetDilated


def transfer(args, config):
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

    def my_collate(batch):
        modified_batch = []
        for item in batch:
            image, label = item
            if label in range(8, 10):
                modified_batch.append(item)
        return default_collate(modified_batch)

    test_loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=1,
                             drop_last=True, collate_fn=my_collate)
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
