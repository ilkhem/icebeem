### Pytorch implementation of training EBMs via FCE

import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.imca import ContrastiveConditionalDataset, ConditionalDataset
from data.utils import to_one_hot
from .ebm import VATLoss


class FCETrainer:
    "write a first version for an unconditional flow"

    def __init__(self, EBM, flow, device='cpu', results_file='results.txt'):
        """
        This is written for an EBM and a flow from nflib
        """
        self.ebm = EBM.to(device)
        self.flow = flow.to(device)
        self.device = device
        self.input_size = self.ebm.input_size
        self.cond_size = self.ebm.cond_size
        use_cuda = device == 'cuda' and torch.cuda.is_available()
        self._loader_params = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.results_file = results_file


    def flow_log_pdf(self, x, y=None):
        _, prior_log_pdf, log_det = self.flow(x)
        return prior_log_pdf + log_det

    def ebm_log_pdf(self, x, y, augment=False, positive=False):
        return self.ebm.log_pdf(x, y, augment, positive)

    def sample_noise(self, n):
        return self.flow.sample(n)[-1].detach().cpu().numpy()

    def _pretrain_train_step(self, dataloader, optimizer, epoch, epochs, log_interval=100):
        for i, (x, y) in enumerate(dataloader):
            # this is written specifically for a labeled dataset
            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)  # in case x is in image

            # the label might not be used if we want the flow to be unconditional
            loss = - self.flow_log_pdf(x).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log_output = 'epoch {:3d} / {};\tloss {:.4f}'.format(epoch, epochs, loss.item())
        # print(log_output)
        print(log_output, file=open(self.results_file, 'a'))

    def pretrain_flow(self, data, epochs=20, batch_size=100, lr=1e-4, wd=1e-5):
        # pretrain the flow model using MLE to have a good approximation of the data density
        # before the contrastive phase
        print('Pretraining flow model')
        x, y = data
        pretrain_dataset = ConditionalDataset(x, y)
        train_dataloader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, **self._loader_params)
        optimizer = optim.Adam(self.flow.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        self.flow.train()
        for i in range(start_epoch, start_epoch + epochs):
            self._pretrain_train_step(train_dataloader, optimizer, i, start_epoch + epochs)

    def _train_step(self, fce_loader, optimizer, epoch, epochs, augment, positive, cutoff, useVAT, train_ebm):
        objConstant = 2. * train_ebm - 1
        loss_criterion = nn.BCELoss()
        num_correct = 0
        for i, (x, y, label) in enumerate(fce_loader):
            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)
            label = label.to(self.device)

            flow_logpdf = self.flow_log_pdf(x).view(-1, 1)
            ebm_logpdf = self.ebm_log_pdf(x, y, augment=augment, positive=positive).view(-1, 1)
            # define logits
            logits = torch.cat((ebm_logpdf - flow_logpdf, flow_logpdf - ebm_logpdf), 1).to(self.device)
            logits *= objConstant
            # compute accuracy:
            num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()
            # define loss
            loss = loss_criterion(torch.sigmoid(logits), label)
            # consider adding VAT loss when training EBM
            if useVAT and train_ebm:
                vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                lds = vat_loss(self.ebm, x)
                loss += 1 * lds

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print some statistics
        n = len(fce_loader.dataset)
        accuracy = num_correct / n
        network = train_ebm * 'ebm' + (1 - train_ebm) * 'flow'
        log_output = '{}: epoch {}/{};\tloss: {:.4f};\taccuracy: {:.3f}'.format(network, epoch, epochs,
                                                                                loss.item(), accuracy)
        # print(log_output)
        print(log_output, file=open(self.results_file, 'a'))

        bk = False
        if accuracy > cutoff:
            # stop training
            log_output = 'accuracy {:.3f}/{} cutoff value satisfied .. stopping training\n----------\n'.format(accuracy,
                                                                                                               cutoff)
            # print(log_output)
            print(log_output, file=open(self.results_file, 'a'))
            bk = True
        return bk

    def train(self, data, epochs=50, batch_size=100, lr=1e-4, wd=1e-5, network='ebm',
              augment=True, positive=False, finalLayerOnly=False, useVAT=False, cutoff=None):
        # setup
        x, y = data
        network = network.lower()
        if network == 'ebm':
            train_ebm = True
            model = self.ebm
        elif network == 'flow':
            train_ebm = False
            model = self.flow
        else:
            raise ValueError('wrong network {}'.format(network))
        if cutoff is None:
            cutoff = 1. * train_ebm
        print('FCE step for the {}'.format(network))

        # generate noise data from flow, and setup contrastive dataset
        # print('Generating samples from flow ...')
        st = time.time()
        n = x.shape[0]
        noise_x = self.sample_noise(n)
        contrastive_y = (np.ones(y.shape) / y.shape[1]).astype(np.float32)  # unconditional so far
        labels = np.array([0] * n + [1] * n)
        dataset = ContrastiveConditionalDataset(np.vstack((x, noise_x)).astype(np.float32),
                                                np.vstack((y, contrastive_y)),
                                                to_one_hot(labels)[0].astype(np.float32), device=self.device)
        fce_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **self._loader_params)
        # print('... done in {}!'.format(time.time() - st))
        # define optimizer
        if finalLayerOnly:
            if not train_ebm:
                raise ValueError("The EBM's g network can't be optimized when training the Flow")
            optimizer = optim.Adam(list(self.ebm.g.parameters()) + [self.ebm.log_norm], lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        model.train()
        start_epoch = 0
        for i in range(start_epoch, start_epoch + epochs):
            bk = self._train_step(fce_loader, optimizer, i, start_epoch + epochs,
                                  augment, positive, cutoff, useVAT, train_ebm)
            if bk:
                break

    def unmix_samples(self, x, network):
        """
        perform unmixing of samples
        """
        if network.lower() == 'ebm':
            # unmix using EBM:
            recov = self.ebm.f(torch.tensor(x.astype(np.float32))).detach().cpu().numpy()
        elif network.lower() == 'flow':
            # unmix using flow model
            recov = self.flow(torch.tensor(x.astype(np.float32)))[0].detach().cpu().numpy()
        else:
            raise ValueError('wrong network {}'.format(network))

        return recov
