### run conditional denoising score matching experiments on MNIST
#
#
# much of this could it adapted from: https://github.com/ermongroup/ncsn/
#

import numpy as np
import tqdm
from losses.dsm import conditional_dsm, dsm
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models.refinenet_dilated_baseline import RefineNetDilated
from torch.utils.data.dataloader import default_collate
import pickle 

__all__ = ['mnist_runner']

def my_collate(batch, nSeg=7):
    modified_batch = []
    for item in batch:
        image, label = item
        if label in range(nSeg):
            modified_batch.append(item)
    return default_collate(modified_batch)

def my_collate_rev(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label in range(8,10):
            modified_batch.append(item)
    return default_collate(modified_batch)

class mnist_runner():
    def __init__(self, args, config, nSeg=7, subsetSize=None, seed=0):
        self.args = args
        self.config = config
        self.nSeg = nSeg # number of segments provided 
        self.subsetSize = subsetSize # subset size, only for baseline transfer learning, otherwise ignored!
        self.seed = seed # subset size, only for baseline transfer learning, otherwise ignored!
        print('USING CONDITIONING DSM')
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

    def train(self):
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
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True, transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True, transform=test_transform)

        elif self.config.data.dataset == 'MNIST':
            print('RUNNING REDUCED MNIST')
            dataset = MNIST('datasets/', train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST('datasets_test/', train=False, download=True,
                                 transform=test_transform)

        elif self.config.data.dataset == 'MNIST_transferBaseline':
            # use same dataset as transfer_nets.py
            test_dataset = MNIST('datasets/mnist_test', train=False, download=True, transform=test_transform)
            print('TRANSFER BASELINES !! Subset size: ' + str(self.subsetSize))
            id_range = list(range(self.subsetSize))
            testset_1 = torch.utils.data.Subset(test_dataset, id_range)

            # dataset = MNIST('/nfs/ghome/live/ricardom/IMCA/ncsn-master/datasets', train=True, download=True,
            #                 transform=tran_transform)
            # test_dataset = MNIST('/nfs/ghome/live/ricardom/IMCA/ncsn-master/datasets_test', train=False, download=True,
            #                      transform=test_transform)

        elif self.config.data.dataset == 'CIFAR10_transferBaseline':
            test_dataset = CIFAR10('datasets/cifar10_test', train=False, download=True, transform=test_transform)
            print('TRANSFER BASELINES !! Subset size: ' + str(self.subsetSize))
            id_range = list(range(self.subsetSize))
            testset_1 = torch.utils.data.Subset(test_dataset, id_range)


        # apply collation for all datasets ! (we only consider MNIST and CIFAR10 anyway!)
        if self.config.data.dataset in ['MNIST', 'CIFAR10']:
            collate_helper = lambda batch: my_collate( batch, nSeg = self.nSeg) 
            dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=0, collate_fn = collate_helper)
            test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=1, drop_last=True,  collate_fn = collate_helper)
        elif self.config.data.dataset in ['MNIST_transferBaseline', 'CIFAR10_transferBaseline']:
            # trains a model on only digits 8,9 from scratch
            dataloader = DataLoader(testset_1, batch_size=self.config.training.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn = my_collate_rev )
            print('loaded mnist reduced subset')
            # SUBSET_SIZE = 500
            # id_range = list(range(SUBSET_SIZE))
            # testset_1 = torch.utils.data.Subset(test_dataset, id_range)
            # dataloader = DataLoader(testset_1, batch_size=self.config.training.batch_size, shuffle=True, num_workers=1, collate_fn = collate_helper)
            # test_loader = DataLoader(testset_1, batch_size=self.config.training.batch_size, shuffle=True,
                                 # num_workers=1, drop_last=True,  collate_fn = my_collate_rev)

        else:
            dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=1)
            test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=1, drop_last=True)

        if False:
            test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        # define the final linear layer weights
        energy_net_finalLayer = torch.ones((  self.config.data.image_size * self.config.data.image_size, self.nSeg )).to(self.config.device)
        energy_net_finalLayer.requires_grad_()

        #tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        enet = RefineNetDilated(self.config).to(self.config.device)

        enet = torch.nn.DataParallel(enet)

        optimizer = self.get_optimizer( list(enet.parameters()) + [energy_net_finalLayer])

        if False: #self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            enet.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        for epoch in range(self.config.training.n_epochs):
            loss_vals = []
            for i, (X, y) in enumerate(dataloader):
                #print(y.max())
                step += 1

                enet.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                # replace this with either dsm or dsm_conditional_score_estimation function !!
                y -= y.min() # need to ensure its zero centered !
                loss = conditional_dsm(enet, X, y, energy_net_finalLayer,  sigma=0.01)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}, maxLabel: {}".format(step, loss.item(), y.max()))
                loss_vals.append( loss.item() )
                if step >= self.config.training.n_iters:
                    return 0

                if False: #step % 100 == 0:
                    enet.eval()
                    try: 
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    with torch.no_grad():
                        test_dsm_loss = conditional_dsm(enet, test_X, test_y, energy_net_finalLayer,  sigma=0.01)

                    #tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    if self.config.data.dataset == 'MNIST_transferBaseline':
                        # just save the losses, thats all we care about
                        if self.config.data.store_loss:
                            #print('only storing losses')
                            import pickle 
                            pickle.dump( loss_vals, open('transfer_exp/transferRes/Baseline_Size' + str(self.subsetSize) + "_Seed" + str(self.seed) + '.p', 'wb'))
                        else:
                            pass
                        if True:
                            # save this one time for transfer learning!
                            states = [
                                enet.state_dict(),
                                optimizer.state_dict(),
                            ]
                            torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                            torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
                            # and the final layer weights !
                            #import pickle
                            #torch.save( [energy_net_finalLayer], 'finalLayerweights_.pth')
                            #pickle.dump( energy_net_finalLayer, open('finalLayerweights.p', 'wb') )
                    else:
                        states = [
                            enet.state_dict(),
                            optimizer.state_dict(),
                        ]
                        torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
                        import pickle
                        torch.save( [energy_net_finalLayer],  os.path.join(self.args.log,'finalLayerweights_.pth') )
                        pickle.dump( energy_net_finalLayer, open(  os.path.join(self.args.log,'finalLayerweights.p'), 'wb') )

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=1000, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod)
                x_mod = x_mod + step_lr * grad + noise
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = RefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        score.eval()

        if self.config.data.dataset == 'MNIST' or self.config.data.dataset == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'MNIST':
                dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                                transform=transform)
            else:
                dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fmnist'), train=True, download=True,
                                       transform=transform)

            dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            samples, _ = next(data_iter)
            samples = samples.cuda()

            samples = torch.rand_like(samples)
            all_samples = self.Langevin_dynamics(samples, score, 1000, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))

        elif self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
            samples, _ = next(iter(dataloader))

            samples = torch.rand(100, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            all_samples = self.Langevin_dynamics(samples, score, 1000, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)

            dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            samples, _ = next(data_iter)
            samples = samples.cuda()
            samples = torch.rand_like(samples)

            all_samples = self.Langevin_dynamics(samples, score, 1000, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
