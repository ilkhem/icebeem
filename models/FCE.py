### Pytorch implementation of training EBMs via FCE

import math
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from .ebm import VATLoss
from .data import SingleDataset, FCEDataset, ConditionalFCEDataset, to_one_hot, ContrastiveMNIST, ContrastiveAlphabet
from .utils import mnist_one_hot_transform, contrastive_one_hot_transform, single_one_hot_encode, one_hot_encode


class FCE(object):
    """
    train an energy based model using noise contrastive estimation
    """

    def __init__(self, data, energy_MLP, flow_model, device='cpu'):
        self.data = data
        self.device = device
        self.energy_MLP = energy_MLP.to(self.device)
        self.ebm_norm = -5.
        self.flow_model = flow_model.to(self.device)  # flow model, must have sample and log density capabilities
        self.noise_samples = None

    def sample_noise(self, n):
        return self.flow_model.sample(n)[-1].detach().cpu().numpy()

    def noise_logpdf(self, dat):
        """
        compute log density under flow model
        """
        zs, prior_logprob, log_det = self.flow_model(dat)
        flow_logdensity = (prior_logprob + log_det)
        return flow_logdensity

    def pretrain_flow(self, epochs=50, lr=1e-4):
        optimizer = optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=1e-5)  # todo tune WD
        print("number of params: ", sum(p.numel() for p in self.flow_model.parameters()))

        dset = SingleDataset(self.data.astype(np.float32))
        train_loader = DataLoader(dset, shuffle=True, batch_size=128)

        # run optimization
        loss_vals = []

        self.flow_model.train()
        for e in range(epochs):
            loss_val = 0
            for _, dat in enumerate(train_loader):
                dat = dat.to(self.device, non_blocking=True)

                zs, prior_logprob, log_det = self.flow_model(dat)
                logprob = prior_logprob + log_det
                loss = torch.sum(-logprob)  # NLL

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

            print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
            loss_vals.append(loss_val)
        return loss_vals

    def train_ebm_fce(self, epochs=500, lr=.0001):
        """
        FCE training of EBM model
        """
        print('Training energy based model using FCE')

        # sample noise data
        n = self.data.shape[0]
        self.noise_samples = self.sample_noise(n)  # self.noise_dist.sample( n )
        # define classification labels
        y = np.array([0] * n + [1] * n)

        # define
        dat_fce = FCEDataset(np.vstack((self.data, self.noise_samples)).astype(np.float32),
                             to_one_hot(y)[0].astype(np.float32))
        fce_loader = DataLoader(dat_fce, shuffle=True, batch_size=128)
        # define log normalization constant
        ebm_norm = self.ebm_norm  # -5.
        logNorm = torch.from_numpy(np.array(ebm_norm).astype(np.float32)).requires_grad_()

        # define optimizer
        optimizer = optim.Adam(list(self.energy_MLP.parameters()) + [logNorm], lr=lr)
        # begin optimization
        loss_criterion = nn.BCELoss()

        self.energy_MLP.train()
        for e in range(epochs):
            num_correct = 0
            loss_val = 0
            for _, (dat, label) in enumerate(fce_loader):
                dat, label = dat.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)

                # noise model probs:
                noise_logpdf = self.noise_logpdf(dat).view(-1, 1)
                # get ebm model probs:
                ebm_logpdf = (self.energy_MLP(dat) + logNorm)

                # define logits
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
                # compute accuracy:
                num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()

                # define loss
                loss = loss_criterion(torch.sigmoid(logits), label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

            # print some statistics
            print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_val, 4),
                                                             np.round(num_correct / (2 * n), 3)))

        self.ebm_norm = logNorm.item()

    def reset_noise(self):
        self.noise_samples = self.sample_noise(self.noise_samples.shape[0])

    def train_flow_fce(self, epochs=50, lr=1e-4, objConstant=-1.0):
        """
        FCE training of EBM model
        """
        print('Training flow contrastive noise for FCE')

        # noise data already sampled during EBM training
        n = self.data.shape[0]
        self.reset_noise()
        # define classification labels
        y = np.array([0] * n + [1] * n)

        # define
        dat_fce = FCEDataset(np.vstack((self.data, self.noise_samples)).astype(np.float32),
                             to_one_hot(y)[0].astype(np.float32))
        fce_loader = DataLoader(dat_fce, shuffle=True, batch_size=128)

        # define optimizer
        optimizer = optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=1e-5)  # todo tune WD
        # begin optimization
        loss_criterion = nn.BCELoss()

        self.flow_model.train()
        for e in range(epochs):
            num_correct = 0
            loss_val = 0
            for _, (dat, label) in enumerate(fce_loader):
                dat, label = dat.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)

                # noise model probs:
                noise_logpdf = self.noise_logpdf(dat).view(-1, 1)
                # get ebm model probs:
                ebm_logpdf = (self.energy_MLP(dat) + self.ebm_norm)

                # define logits
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
                logits *= objConstant
                # compute accuracy:
                num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()

                # define loss
                loss = loss_criterion(torch.sigmoid(logits), label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

            # print some statistics
            print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_val, 4),
                                                             np.round(1 - num_correct / (2 * n), 3)))


class ConditionalFCE(object):
    """
    train an energy based model using noise contrastive estimation
    where we assume we observe data from multiple segments/classes
    this is useful for nonlinear ICA and semi supervised learning !
    """

    def __init__(self, data, segments, energy_MLP, flow_model, verbose=False, device='cpu'):
        self.data = data
        self.segments = segments
        self.energy_MLP = energy_MLP.to(device)
        self.ebm_norm = -5.
        self.hidden_dim = self.energy_MLP.linearLast.weight.shape[0]
        self.n_segments = self.segments.shape[1]
        self.ebm_finalLayer = torch.tensor(np.ones((self.hidden_dim, self.n_segments)).astype(np.float32)).to(device)
        # self.ebm_finalLayer = torch.tensor( np.random.random(( self.hidden_dim, self.n_segments )).astype(np.float32) )
        self.flow_model = flow_model.to(device)
        self.noise_samples = None
        self.device = device
        self.verbose = verbose

    def sample_noise(self, n):
        return self.flow_model.sample(n)[-1].detach().cpu().numpy()

    def noise_logpdf(self, dat):
        """
        compute log density under flow model
        """
        zs, prior_logprob, log_det = self.flow_model(dat)
        flow_logdensity = (prior_logprob + log_det)
        return flow_logdensity

    def compute_ebm_logpdf(self, dat, seg, logNorm, augment=False):
        act_allLayer = torch.mm(self.energy_MLP(dat), self.ebm_finalLayer)
        if augment:
            # we augment the feature extractor
            act_allLayer += torch.mm(self.energy_MLP(dat) * self.energy_MLP(dat),
                                     self.ebm_finalLayer * self.ebm_finalLayer)
        # now select relevant layers by multiplying by mask matrix and reducing (and adding log norm)
        act_segment = (act_allLayer * seg).sum(1) + logNorm
        return act_segment

    def train_ebm_fce(self, epochs=500, lr=.0001, cutoff=None, augment=True, finalLayerOnly=False, useVAT=False):
        """
        FCE training of EBM model
        """
        print('Training energy based model using FCE' + useVAT * ' with VAT penalty')

        if cutoff is None:
            cutoff = 1.00  # will basically only stop with perfect classification

        # sample noise data
        n = self.data.shape[0]
        self.noise_samples = self.sample_noise(n)  # self.noise_dist.sample( n )
        # define classification labels
        y = np.array([0] * n + [1] * n)

        self.contrastSegments = (np.ones(self.segments.shape) / self.segments.shape[1])

        dat_fce = ConditionalFCEDataset(np.vstack((self.data, self.noise_samples)).astype(np.float32),
                                        to_one_hot(y)[0].astype(np.float32),
                                        np.vstack((self.segments, self.contrastSegments)).astype(np.float32))
        fce_loader = DataLoader(dat_fce, shuffle=True, batch_size=128)

        # define log normalization constant
        ebm_norm = self.ebm_norm  # -5.
        logNorm = torch.from_numpy(np.array(ebm_norm).astype(np.float32)).float().to(self.device).requires_grad_()
        self.ebm_finalLayer.requires_grad_()

        # define optimizer
        if finalLayerOnly:
            # only train the final layer, this is the equivalent of g(y) in
            # IMCA manuscript.
            optimizer = optim.Adam([self.ebm_finalLayer] + [logNorm], lr=lr)
        else:
            optimizer = optim.Adam(list(self.energy_MLP.parameters()) + [self.ebm_finalLayer] + [logNorm], lr=lr)
        self.energy_MLP.train()

        # begin optimization
        loss_criterion = nn.BCELoss()

        # import pdb; pdb.set_trace()

        for e in range(epochs):
            num_correct = 0
            loss_val = 0
            for _, (dat, label, seg) in enumerate(fce_loader):
                dat, label = dat.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)
                seg = seg.to(self.device, non_blocking=True)

                noise_logpdf = self.noise_logpdf(dat).view(-1, 1)
                ebm_logpdf = self.compute_ebm_logpdf(dat, seg, logNorm, augment=augment).view(-1, 1)

                # define logits
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1).to(self.device)
                # compute accuracy:
                num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()

                # define loss
                loss = loss_criterion(torch.sigmoid(logits), label)

                # consider adding VAT loss
                if useVAT:
                    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                    lds = vat_loss(self.energy_MLP, dat)
                    loss += 1 * lds

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

            # print some statistics
            print('epoch {}\{} \tloss: {:.4f}\taccuracy: {:.3f}'.format(e, epochs, loss_val, num_correct / (2 * n)))
            if num_correct / (2 * n) > cutoff:
                # stop training
                print('epoch {}\{}\taccuracy: {:.3f}'.format(e, epochs, num_correct / (2 * n)))
                print('cutoff value satisfied .. stopping training\n----------\n')
                break

        self.ebm_norm = logNorm.item()

    def reset_noise(self):
        self.noise_samples = self.sample_noise(self.noise_samples.shape[0])

    def train_flow_fce(self, epochs=50, lr=1e-4, objConstant=-1.0, cutoff=None):
        """
        FCE training of EBM model
        """

        print('Training flow contrastive noise for FCE')

        if cutoff is None:
            cutoff = 0.  # basically only stop for perfect misclassification

        # noise data already sampled during EBM training
        n = self.data.shape[0]
        self.reset_noise()
        # define classification labels
        y = np.array([0] * n + [1] * n)

        # define
        self.contrastSegments = (np.ones(self.segments.shape) / self.segments.shape[1])

        dat_fce = ConditionalFCEDataset(np.vstack((self.data, self.noise_samples)).astype(np.float32),
                                        to_one_hot(y)[0].astype(np.float32),
                                        np.vstack((self.segments, self.contrastSegments)).astype(np.float32))
        fce_loader = DataLoader(dat_fce, shuffle=True, batch_size=128)

        # define optimizer
        optimizer = optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=1e-5)  # todo tune WD

        # begin optimization
        loss_criterion = nn.BCELoss()

        self.flow_model.train()
        for e in range(epochs):
            num_correct = 0
            loss_val = 0
            for _, (dat, label, seg) in enumerate(fce_loader):
                dat, label = dat.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)
                seg = seg.to(self.device, non_blocking=True)

                noise_logpdf = self.noise_logpdf(dat).view(-1, 1)
                ebm_logpdf = self.compute_ebm_logpdf(dat, seg, self.ebm_norm).view(-1, 1)

                # define logits
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
                logits *= objConstant

                # compute accuracy:
                num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()

                # define loss
                loss = loss_criterion(torch.sigmoid(logits), label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

            # print some statistics
            print('epoch {}\{} \tloss: {:.4f}\taccuracy: {:.3f}'.format(e, epochs, loss_val, 1 -  num_correct / (2 * n)))

            if 1 - num_correct / (2 * n) < cutoff:
                print('epoch {}\{}\taccuracy: {:.3f}'.format(e, epochs,1 - num_correct / (2 * n)))
                print('cutoff value satisfied .. stopping training\n----------\n')
                break

    def pretrain_flow(self, epochs=50, lr=1e-4):
        optimizer = optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=1e-5)  # todo tune WD
        print("number of params: ", sum(p.numel() for p in self.flow_model.parameters()))

        dset = SingleDataset(self.data.astype(np.float32))
        train_loader = DataLoader(dset, shuffle=True, batch_size=128)

        # run optimization
        loss_vals = []

        self.flow_model.train()
        for e in range(epochs):
            loss_val = 0
            for _, dat in enumerate(train_loader):
                dat = dat.to(self.device, non_blocking=True)

                zs, prior_logprob, log_det = self.flow_model(dat)
                logprob = prior_logprob + log_det
                loss = - torch.sum(logprob)  # NLL

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

            print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
            loss_vals.append(loss_val)
        return loss_vals

    def unmixSamples(self, data, modelChoice):
        """
        perform unmixing of samples
        """
        if modelChoice == 'EBM':
            # unmix using EBM:
            recov = self.energy_MLP(torch.tensor(data.astype(np.float32))).detach().cpu().numpy()
        else:
            # unmix using flow model
            recov = self.flow_model(torch.tensor(data.astype(np.float32)))[0][-1].detach().cpu().numpy()

        return recov


class FCE2:
    def __init__(self, EBM, flow, device='cpu', dset_train=None, dset_test=None,
                 data_path='../data/', output_dir='../output/',
                 pretrain_results_file='pretrain_results.txt', pretrain_restore_file=None,
                 results_file='results.txt', restore_ebm_file=None, restore_flow_file=None):
        """
        This is written for an EBM and a MAF from nflib/maf.py
        """

        self.ebm = EBM.to(device)
        # FLOW MODEL HAS TO HAVE A log_pdf THAT RETURNS 1: pdf, AND A sample THAT RETURNS 1: samples, 2: labels, 3: pdf
        self.flow = flow.to(device)
        self.device = device

        use_cuda = device == 'cuda' and torch.cuda.is_available()
        self._loader_params = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}

        # logging and checkpoints
        self.data_path = data_path
        self.output_dir = output_dir
        self.pretrain_results_file = pretrain_results_file
        self.pretrain_restore_file = pretrain_restore_file
        self.results_file = results_file
        self.restore_ebm_file = restore_ebm_file
        self.restore_flow_file = restore_flow_file

        self.conditional_flow = self.flow.conditional
        self.cond_size = self.flow.cond_size

        assert self.cond_size == self.ebm.cond_size

        self.dset_train = dset_train
        self.dset_test = dset_test

    def sample_noise(self, n, cond_size=None):
        samples, labels, log_probs = self.flow.sample(n, cond_size)  # yields samples and labels even if unconditional!!
        return samples.detach(), labels.detach(), log_probs.detach()  # stay on device, and as torch tensor, but detach from graph

    def flow_log_pdf(self, x, y=None):
        return self.flow.log_pdf(x, y)

    def ebm_log_pdf(self, x, y, *args, **kwargs):
        return self.ebm.log_pdf(x, y, *args, **kwargs)

    def _pretrain_train_step(self, dataloader, optimizer, epoch, epochs, log_interval=100):
        for i, (x, y) in enumerate(dataloader):
            self.flow.train()

            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)

            if self.conditional_flow:
                loss = - self.flow.log_pdf(x, y).mean(0)
            else:
                loss = - self.flow.log_pdf(x).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                log_output = 'epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(epoch, epochs, i, len(dataloader),
                                                                                     loss.item())
                print(log_output)
                print(log_output, file=open(self.pretrain_results_file, 'a'))

    @torch.no_grad()
    def _pretrain_evaluate_step(self, dataloader, epoch):
        self.flow.eval()

        # conditional model
        if self.conditional_flow:
            logprior = torch.tensor(1 / self.cond_size).log().to(self.device)
            loglike = [[] for _ in range(self.cond_size)]

            for i in range(self.cond_size):
                for x, y in dataloader:
                    # make one-hot labels
                    labels = torch.zeros(x.shape[0], self.cond_size).to(self.device)
                    labels[:, i] = 1
                    x = x.view(x.shape[0], -1).to(self.device)
                    loglike[i].append(self.flow.log_pdf(x, labels))

                loglike[i] = torch.cat(loglike[i], dim=0)  # cat along data dim under this label
            loglike = torch.stack(loglike, dim=1)  # cat all data along label dim

            logprobs = logprior + loglike.logsumexp(dim=1)

        # unconditional model
        else:
            logprobs = []
            for data in dataloader:
                x = data[0].view(data[0].shape[0], -1).to(self.device)
                logprobs.append(self.flow.log_pdf(x))
            logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
        output = 'Evaluate ' + (epoch != None) * '(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(
            logprob_mean, logprob_std)
        print(output)
        print(output, file=open(self.pretrain_results_file, 'a'))
        return logprob_mean, logprob_std

    @torch.no_grad()
    def _pretrain_generate(self, step=None, n_row=10):
        self.flow.eval()

        n_samples = n_row * self.cond_size if self.conditional_flow else n_row ** 2
        samples = self.sample_noise(n_samples,
                                    cond_size=self.cond_size if self.conditional_flow else None)[0]
        samples = samples.view(samples.shape[0], 1, 28, 28)  # hardcoded mnist dims!!

        filename = 'generated_samples' + (step != None) * '_epoch_{}'.format(step) + '.png'
        save_image(samples, os.path.join(self.output_dir, filename), nrow=n_row, normalize=True)

    def pretrain_flow(self, epochs=20, batch_size=64, lr=1e-4, wd=1e-5, evaluate=False, log_interval=100):
        # pretrain the flow model using MLE to have a good approximation of the data density
        # before the contrastive phase
        print('Pretraining {} flow model'.format('conditional' if self.conditional_flow else ''))

        # Default dataset is full MNIST. This is so that it doesn't save it in memory!
        if self.dset_train is None:
            dset_train = datasets.MNIST(self.data_path, train=True, download=True,
                                        transform=transforms.ToTensor(), target_transform=mnist_one_hot_transform)
        else:
            dset_train = self.dset_train
        if self.dset_test is None:
            dset_test = datasets.MNIST(self.data_path, train=False, download=True,
                                       transform=transforms.ToTensor(), target_transform=mnist_one_hot_transform)
        else:
            dset_test = self.dset_test
        train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, **self._loader_params)
        test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=True, **self._loader_params)

        optimizer = optim.Adam(self.flow.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        if self.pretrain_restore_file:
            print('Loading model from checkpoints..')
            state = torch.load(self.pretrain_restore_file, map_location=self.device)
            self.flow.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(self.pretrain_restore_file)
        self.pretrain_results_file = os.path.join(self.output_dir, self.pretrain_results_file)

        best_eval_logprob = float('-inf')

        for i in range(start_epoch, start_epoch + epochs):
            self._pretrain_train_step(train_dataloader, optimizer, i, start_epoch + epochs, log_interval=log_interval)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': self.flow.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       os.path.join(self.output_dir, 'pretrained_flow_checkpoint.pt'))
            # save model only
            torch.save(self.flow.state_dict(), os.path.join(self.output_dir, 'pretrained_flow_state.pt'))

            # save best eval state
            if evaluate:
                eval_logprob, _ = self._pretrain_evaluate_step(test_dataloader, i)
                if eval_logprob > best_eval_logprob:
                    best_eval_logprob = eval_logprob
                    torch.save({'epoch': i,
                                'model_state': self.flow.state_dict(),
                                'optimizer_state': optimizer.state_dict()},
                               os.path.join(self.output_dir, 'best_pretrained_flow_checkpoint.pt'))
                # plot sample
                self._pretrain_generate(step=i)

    def _train_step(self, dataloader, optimizer, epoch, epochs, augment, positive, cutoff, useVAT, train_ebm,
                    log_interval=100):
        objConstant = 2. * train_ebm - 1
        loss_criterion = nn.BCELoss()
        num_correct = 0
        for i, (x, y) in enumerate(dataloader):
            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)
            # sample noise, and evaluate its log pdf through the flow
            x_tilde, y_tilde, nflp = self.sample_noise(y.shape[0],
                                                       cond_size=self.cond_size if self.conditional_flow else None)
            label = np.array([0] * x.shape[0] + [1] * x_tilde.shape[0])
            label = torch.tensor(one_hot_encode(label, 2)).to(self.device)

            if self.conditional_flow:
                flow_logpdf = self.flow_log_pdf(x, y)
            else:
                flow_logpdf = self.flow_log_pdf(x)
            flow_logpdf = torch.cat([flow_logpdf, nflp], dim=0).view(-1, 1)

            x = torch.cat([x, x_tilde], dim=0)
            y = torch.cat([y, y_tilde if self.conditional_flow else y], dim=0)

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

            if i % log_interval == 0:
                n = len(dataloader.dataset)
                accuracy = num_correct / (2 * n)
                network = train_ebm * 'ebm' + (1 - train_ebm) * 'flow'
                log_output = '{}: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}; accuracy {:.3F}'.format(network,
                                                                                                          epoch, epochs,
                                                                                                          i,
                                                                                                          len(
                                                                                                              dataloader),
                                                                                                          loss.item(),
                                                                                                          accuracy)
                print(log_output)
                print(log_output, file=open(self.results_file, 'a'))

        bk = False
        if accuracy > cutoff:
            # stop training
            log_output = 'accuracy {:.3f}/{} cutoff value satisfied .. stopping training\n----------\n'.format(accuracy,
                                                                                                               cutoff)
            print(log_output)
            print(log_output, file=open(self.results_file, 'a'))
            bk = True
        return bk

    def train(self, epochs=50, batch_size=100, lr=1e-4, wd=1e-5, network='ebm',
              augment=True, positive=False,
              cutoff=None, finalLayerOnly=False, useVAT=False, start_from_ckpt=True, log_interval=100):

        # setup
        if network.lower() == 'ebm':
            train_ebm = True
            model = self.ebm
            restore_file = self.restore_ebm_file
        elif network.lower() == 'flow':
            restore_file = self.restore_flow_file
            train_ebm = False
            model = self.flow
        else:
            raise ValueError('wrong network {}'.format(network))
        if cutoff is None:
            cutoff = 1. * train_ebm
        if not start_from_ckpt:
            restore_file = None
        network = network.lower()

        print('FCE step for the {}'.format(network))

        if self.dset_train is None:
            dset_train = datasets.MNIST(self.data_path, train=True, download=True,
                                        transform=transforms.ToTensor(), target_transform=mnist_one_hot_transform)
        else:
            dset_train = self.dset_train
        train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, **self._loader_params)

        # define optimizer
        if finalLayerOnly:
            if not train_ebm:
                raise ValueError("The EBM's g network can't be optimized when training the Flow")
            optimizer = optim.Adam(list(self.ebm.g.parameters()) + [self.ebm.log_norm], lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        if restore_file:
            state = torch.load(restore_file, map_location=self.device)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(restore_file)
        self.results_file = os.path.join(self.output_dir, self.results_file)

        # if model.is_cuda:
        #    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #    cudnn.benchmark = True

        model.train()
        for i in range(start_epoch, start_epoch + epochs):
            bk = self._train_step(train_dataloader, optimizer, i, start_epoch + epochs,
                                  augment, positive, cutoff, useVAT, train_ebm, log_interval)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       os.path.join(self.output_dir, '{}_checkpoint.pt'.format(network)))
            # save model only
            torch.save(model.state_dict(), os.path.join(self.output_dir, '{}_state.pt'.format(network)))

            if bk:
                break

    def train_ebm(self, *args, **kwargs):
        if 'network' in kwargs.keys():
            dict.pop('network')
        self.train(network='ebm', *args, **kwargs)

    def train_flow(self, *args, **kwargs):
        if 'network' in kwargs.keys():
            dict.pop('network')
        self.train(network='flow', *args, **kwargs)

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


class MNISTConditionalFCE:
    def __init__(self, EBM, flow, device='cpu', range=10,
                 data_path='../data/', output_dir='../output/',
                 pretrain_results_file='pretrain_results.txt', results_file='results.txt',
                 pretrain_save_file=None, pretrain_restore_file=None,
                 ebm_save_file=None, ebm_restore_file=None,
                 flow_save_file=None, flow_restore_file=None):
        """
        This is written for an EBM and a MAF from nflib/maf.py
        """

        self.ebm = EBM.to(device)
        self.flow = flow.to(device)
        self.device = device
        self.range = range  # number of labels to train on

        self._len_mnist = 60000
        use_cuda = device == 'cuda' and torch.cuda.is_available()
        self._loader_params = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}

        # logging and checkpoints
        self.data_path = data_path
        self.output_dir = output_dir
        self.pretrain_results_file = os.path.join(output_dir, pretrain_results_file)
        self.results_file = os.path.join(output_dir, results_file)

        self.pretrain_restore_file = os.path.join(output_dir,
                                                  pretrain_restore_file) if pretrain_restore_file is not None else None
        if pretrain_save_file is None:
            pretrain_save_file = 'pretrained_flow_checkpoint.pt'
        self.pretrain_save_file = os.path.join(output_dir, pretrain_save_file)

        self.ebm_restore_file = os.path.join(output_dir, ebm_restore_file) if ebm_restore_file is not None else None
        if ebm_save_file is None:
            ebm_save_file = 'ebm_checkpoint.pt'
        self.ebm_save_file = os.path.join(output_dir, ebm_save_file)

        self.flow_restore_file = os.path.join(output_dir, flow_restore_file) if flow_restore_file is not None else None
        if flow_save_file is None:
            flow_save_file = 'flow_checkpoint.pt'
        self.flow_save_file = os.path.join(output_dir, flow_save_file)

        self.conditional_flow = self.flow.condition_size is not None

    def sample_noise(self, n, chunk=300):
        samples = []
        nfc = n // chunk
        rem = n % chunk
        for i in range(nfc):
            samples += [self.flow.sample(chunk).detach().cpu()]
        samples += [self.flow.sample(rem).detach().cpu()]
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_noise2(self, n, output_device=None):
        self.flow.eval()
        if output_device is None:
            output_device = self.device

        # conditional model
        if self.conditional_flow:

            cond_size = self.flow.condition_size
            n_row = n // cond_size
            n_rem = n % cond_size
            samples = []
            labels_ret = []
            logpdfs = []
            labels = torch.eye(cond_size).to(self.device)

            if n_row > 0:
                for i in range(cond_size):
                    # sample model base distribution and run through backward model to sample data space
                    u = self.flow.base_dist.sample((n_row,))
                    labels_i = labels[i].expand(n_row, -1)
                    sample, _ = self.flow.backward(u, labels_i)
                    log_probs = self.flow.log_pdf(sample, labels_i)
                    logpdfs.append(log_probs)
                    samples.append(sample)
                    labels_ret.append(labels_i)

            if n_rem > 0:
                j = np.random.randint(cond_size)
                u = self.flow.base_dist.sample((n_rem,))
                labels_j = labels[j].expand(n_rem, -1)
                sample, _ = self.flow.backward(u, labels_j)
                log_probs = self.flow.log_pdf(sample, labels_j)
                logpdfs.append(log_probs)
                samples.append(sample)
                labels_ret.append(labels_j)

            samples = torch.cat(samples, dim=0)
            labels_ret = torch.cat(labels_ret, dim=0)
            logpdfs = torch.cat(logpdfs, dim=0)


        # unconditional model
        else:
            n_row = n
            u = self.flow.base_dist.sample((n_row, 1)).squeeze()
            samples, _ = self.flow.backward(u)
            logpdfs = self.flow.log_pdf(samples)
            labels_ret = None

        # # convert and save images
        # samples = samples.view(samples.shape[0], 1, 28, 28)
        # samples = (torch.sigmoid(samples) - lam) / (1 - 2 * lam)

        return samples.to(output_device), labels_ret.to(output_device), logpdfs.to(output_device)

    def flow_log_pdf(self, x, y):
        return self.flow.log_pdf(x, y)

    def ebm_log_pdf(self, x, y, *args, **kwargs):
        return self.ebm.log_pdf(x, y, *args, **kwargs)

    def _pretrain_train_step(self, dataloader, optimizer, epoch, epochs, log_interval=100):
        for i, (x, y) in enumerate(dataloader):
            self.flow.train()
            # this is written specifically for MNIST, the dataset is labeled, always.

            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)

            # the label might not be used if we want the flow to be unconditional
            loss = - self.flow.log_pdf(x, y if self.conditional_flow else None).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                log_output = 'epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(epoch, epochs, i, len(dataloader),
                                                                                     loss.item())
                print(log_output)
                print(log_output, file=open(self.pretrain_results_file, 'a'))

    @torch.no_grad()
    def _pretrain_evaluate_step(self, dataloader, epoch):
        self.flow.eval()

        # conditional model
        if self.conditional_flow:
            cond_size = self.flow.condition_size
            logprior = torch.tensor(1 / cond_size).log().to(self.device)
            loglike = [[] for _ in range(cond_size)]

            for i in range(cond_size):
                # make one-hot labels
                labels = torch.zeros(dataloader.batch_size, cond_size).to(self.device)
                labels[:, i] = 1

                for x, y in dataloader:
                    labels = labels[:x.shape[0]]  # in case last batch has smaller dim
                    x = x.view(x.shape[0], -1).to(self.device)
                    loglike[i].append(self.flow.log_pdf(x, labels))

                loglike[i] = torch.cat(loglike[i], dim=0)  # cat along data dim under this label
            loglike = torch.stack(loglike, dim=1)  # cat all data along label dim

            logprobs = logprior + loglike.logsumexp(dim=1)

        # unconditional model
        else:
            logprobs = []
            for data in dataloader:
                x = data[0].view(data[0].shape[0], -1).to(self.device)
                logprobs.append(self.flow.log_pdf(x))
            logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
        output = 'Evaluate ' + (epoch != None) * '(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(
            logprob_mean, logprob_std)
        print(output)
        print(output, file=open(self.pretrain_results_file, 'a'))
        return logprob_mean, logprob_std

    @torch.no_grad()
    def _pretrain_generate(self, lam=1e-6, step=None, n_row=10):
        self.flow.eval()

        # conditional model
        if self.conditional_flow:
            cond_size = self.flow.condition_size
            samples = []
            labels = torch.eye(cond_size).to(self.device)

            for i in range(cond_size):
                # sample model base distribution and run through backward model to sample data space
                u = self.flow.base_dist.sample((n_row,))
                labels_i = labels[i].expand(n_row, -1)
                sample, _ = self.flow.backward(u, labels_i)
                log_probs = self.flow.log_pdf(sample, labels_i).sort(0)[1].flip(
                    0)  # sort by log_prob; take argsort idxs; flip high to low
                samples.append(sample[log_probs])

            samples = torch.cat(samples, dim=0)

        # unconditional model
        else:
            u = self.flow.base_dist.sample((n_row ** 2,))
            samples, _ = self.flow.backward(u)
            log_probs = self.flow.log_pdf(samples).sort(0)[1].flip(
                0)  # sort by log_prob; take argsort idxs; flip high to low
            samples = samples[log_probs]

        # convert and save images
        samples = samples.view(samples.shape[0], 1, 28, 28)
        samples = (torch.sigmoid(samples) - lam) / (1 - 2 * lam)
        filename = 'mnist2_generated_samples' + (step != None) * '_epoch_{}'.format(step) + '.png'
        save_image(samples, os.path.join(self.output_dir, filename), nrow=n_row, normalize=True)

    def _clip_dset(self, dset):
        labels_to_consider = np.arange(self.range)
        idx = np.any([dset.targets.numpy() == i for i in labels_to_consider], axis=0).nonzero()
        dset.targets = dset.targets[idx]
        dset.data = dset.data[idx]
        dset.target_transform = lambda label: single_one_hot_encode(label, n_labels=len(labels_to_consider))
        print(len(dset))
        return dset

    def pretrain_flow(self, epochs=20, batch_size=100, lr=1e-4, wd=1e-5, evaluate=False):
        # pretrain the flow model using MLE to have a good approximation of the data density
        # before the contrastive phase
        print('Pretraining flow model')
        dset_train = datasets.MNIST(self.data_path, train=True, download=True,
                                    transform=transforms.ToTensor(),
                                    # target_transform=mnist_one_hot_transform  # comment and apply transform after clip
                                    )
        dset_test = datasets.MNIST(self.data_path, train=False, download=True,
                                   transform=transforms.ToTensor(),
                                   # target_transform=mnist_one_hot_transform  # comment and apply transform after clip
                                   )
        dset_train = self._clip_dset(dset_train)
        dset_test = self._clip_dset(dset_test)

        train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, **self._loader_params)
        test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=True, **self._loader_params)

        self._len_mnist = len(dset_train)

        optimizer = optim.Adam(self.flow.parameters(), lr=lr, weight_decay=wd)  # todo tune WD

        start_epoch = 0
        if self.pretrain_restore_file:
            print('Loading model from checkpoints..')
            state = torch.load(self.pretrain_restore_file, map_location=self.device)
            self.flow.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            # start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(self.pretrain_restore_file)

        best_eval_logprob = float('-inf')

        for i in range(start_epoch, start_epoch + epochs):
            self._pretrain_train_step(train_dataloader, optimizer, i, start_epoch + epochs)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': self.flow.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       self.pretrain_save_file)
            # save model only
            torch.save(self.flow.state_dict(), self.pretrain_save_file.split('.pt')[0] + '_state.pt')

            # save best eval state
            if evaluate:
                eval_logprob, _ = self._pretrain_evaluate_step(test_dataloader, i)
                if eval_logprob > best_eval_logprob:
                    best_eval_logprob = eval_logprob
                    torch.save({'epoch': i,
                                'model_state': self.flow.state_dict(),
                                'optimizer_state': optimizer.state_dict()},
                               self.pretrain_save_file.split('.pt')[0] + '_best.pt')
                # plot sample
                self._pretrain_generate(step=i)

    def _train_step(self, fce_loader, optimizer, epoch, epochs, augment, positive, cutoff, useVAT, train_ebm,
                    log_interval=100):
        objConstant = 2. * train_ebm - 1
        loss_criterion = nn.BCELoss()
        num_correct = 0
        n_total = 0
        loss_hist = []
        for i, (x, y, label) in enumerate(fce_loader):
            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)
            label = label.to(self.device)

            # import pdb; pdb.set_trace()

            flow_logpdf = self.flow_log_pdf(x, y).view(-1, 1)
            ebm_logpdf = self.ebm_log_pdf(x, y, augment=augment, positive=positive).view(-1, 1)

            # define logits
            logits = torch.cat((ebm_logpdf - flow_logpdf, flow_logpdf - ebm_logpdf), 1).to(self.device)
            logits *= objConstant
            # compute accuracy:
            num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()
            n_total += logits.shape[0]

            # import pdb; pdb.set_trace()

            # define loss
            loss = loss_criterion(torch.sigmoid(logits), label)
            # consider adding VAT loss when training EBM
            if useVAT and train_ebm:
                vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                lds = vat_loss(self.ebm, (x, y))
                loss += 1 * lds

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist += [loss.item]

            if i % log_interval == 0:
                accuracy = num_correct / n_total
                network = train_ebm * 'ebm' + (1 - train_ebm) * 'flow'
                log_output = '{}: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}; accuracy {:.3F}'.format(network,
                                                                                                          epoch, epochs,
                                                                                                          i,
                                                                                                          len(
                                                                                                              fce_loader),
                                                                                                          loss.item(),
                                                                                                          accuracy)
                print(log_output)
                print(log_output, file=open(self.results_file, 'a'))

        bk = False
        if accuracy > cutoff:
            # stop training
            log_output = 'accuracy {:.3f}/{} cutoff value satisfied .. stopping training\n----------\n'.format(accuracy,
                                                                                                               cutoff)
            print(log_output)
            print(log_output, file=open(self.results_file, 'a'))
            bk = True
        return bk

    def train(self, epochs=50, batch_size=100, lr=1e-4, wd=1e-5, network='ebm',
              augment=True, positive=False,
              cutoff=None, finalLayerOnly=False, useVAT=False):

        # setup
        if network.lower() == 'ebm':
            train_ebm = True
            model = self.ebm
            restore_file = self.ebm_restore_file
            save_file = self.ebm_save_file
        elif network.lower() == 'flow':
            restore_file = self.flow_restore_file
            train_ebm = False
            model = self.flow
            save_file = self.flow_save_file
        else:
            raise ValueError('wrong network {}'.format(network))
        if cutoff is None:
            cutoff = 1. * train_ebm
        network = network.lower()

        print('FCE step for the {}'.format(network))

        # generate noise data from flow, and setup contrastive dataset
        print('Generating samples from flow ...')
        st = time.time()
        noise_samples = self.sample_noise(self._len_mnist)
        dataset = ContrastiveMNIST(self.data_path, noise_samples, train=True, download=True,
                                   transform=transforms.ToTensor(),
                                   segment_transform=mnist_one_hot_transform,  # 10D one-hot for classes
                                   target_transform=contrastive_one_hot_transform)  # 2D one-hot for true or fake
        fce_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **self._loader_params)
        print('... done in {}!'.format(time.time() - st))

        # define optimizer
        if finalLayerOnly:
            if not train_ebm:
                raise ValueError("The EBM's g network can't be optimized when training the Flow")
            optimizer = optim.Adam(list(self.ebm.g.parameters()) + [self.ebm.log_norm], lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        if restore_file:
            state = torch.load(restore_file, map_location=self.device)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(restore_file)

        # if model.is_cuda:
        #    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #    cudnn.benchmark = True

        model.train()
        for i in range(start_epoch, start_epoch + epochs):
            bk = self._train_step(fce_loader, optimizer, i, start_epoch + epochs,
                                  augment, positive, cutoff, useVAT, train_ebm)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       save_file)
            # save model only
            torch.save(model.state_dict(), 'state_' + save_file)

            if bk:
                break

    def _train_step2(self, dataloader, optimizer, epoch, epochs, augment, positive, cutoff, useVAT, train_ebm,
                     log_interval=100):
        objConstant = 2. * train_ebm - 1
        loss_criterion = nn.BCELoss()
        num_correct = 0
        n_total = 0
        loss_hist = []
        for i, (x, y) in enumerate(dataloader):
            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)
            # sample noise, and evaluate its log pdf through the flow
            x_tilde, y_tilde = self.sample_noise2(x.shape[0])
            label = np.array([0] * x.shape[0] + [1] * x_tilde.shape[0])
            label = torch.tensor(one_hot_encode(label, 2)).to(self.device)

            x = torch.cat([x, x_tilde], dim=0)
            y = torch.cat([y, y_tilde], dim=0)

            flow_logpdf = self.flow_log_pdf(x, y).view(-1, 1)
            ebm_logpdf = self.ebm_log_pdf(x, y, augment=augment, positive=positive).view(-1, 1)

            # define logits
            logits = torch.cat((ebm_logpdf - flow_logpdf, flow_logpdf - ebm_logpdf), 1).to(self.device)
            logits *= objConstant
            # compute accuracy:
            num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()
            n_total += logits.shape[0]

            # define loss
            loss = loss_criterion(torch.sigmoid(logits), label)
            # consider adding VAT loss when training EBM
            if useVAT and train_ebm:
                vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                lds = vat_loss(self.ebm, (x, y))
                loss += 1 * lds

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist += [loss.item]

            if i % log_interval == 0:
                accuracy = num_correct / n_total
                network = train_ebm * 'ebm' + (1 - train_ebm) * 'flow'
                log_output = '{}: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}; accuracy {:.3F}'.format(network,
                                                                                                          epoch, epochs,
                                                                                                          i,
                                                                                                          len(
                                                                                                              dataloader),
                                                                                                          loss.item(),
                                                                                                          accuracy)
                print(log_output)
                print(log_output, file=open(self.results_file, 'a'))

        bk = False
        if accuracy > cutoff:
            # stop training
            log_output = 'accuracy {:.3f}/{} cutoff value satisfied .. stopping training\n----------\n'.format(accuracy,
                                                                                                               cutoff)
            print(log_output)
            print(log_output, file=open(self.results_file, 'a'))
            bk = True
        return bk

    def train2(self, epochs=50, batch_size=100, lr=1e-4, wd=1e-5, network='ebm',
               augment=True, positive=False,
               cutoff=None, finalLayerOnly=False, useVAT=False, start_from_ckpt=True, log_interval=100):

        # setup
        if network.lower() == 'ebm':
            train_ebm = True
            model = self.ebm
            restore_file = self.ebm_restore_file
            save_file = self.ebm_save_file
        elif network.lower() == 'flow':
            restore_file = self.flow_restore_file
            train_ebm = False
            model = self.flow
            save_file = self.flow_save_file
        else:
            raise ValueError('wrong network {}'.format(network))
        if cutoff is None:
            cutoff = 1. * train_ebm
        if not start_from_ckpt:
            restore_file = None
        network = network.lower()

        print('FCE step for the {}'.format(network))

        dset_train = datasets.MNIST(self.data_path, train=True, download=True,
                                    transform=transforms.ToTensor(),
                                    # target_transform=mnist_one_hot_transform
                                    )
        dset_train = self._clip_dset(dset_train)
        train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, **self._loader_params)

        self._len_mnist = len(dset_train)

        # define optimizer
        if finalLayerOnly:
            if not train_ebm:
                raise ValueError("The EBM's g network can't be optimized when training the Flow")
            optimizer = optim.Adam(list(self.ebm.g.parameters()) + [self.ebm.log_norm], lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        if restore_file:
            state = torch.load(restore_file, map_location=self.device)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(restore_file)
        self.results_file = os.path.join(self.output_dir, self.results_file)

        # if model.is_cuda:
        #    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #    cudnn.benchmark = True

        model.train()
        for i in range(start_epoch, start_epoch + epochs):
            bk = self._train_step2(train_dataloader, optimizer, i, start_epoch + epochs,
                                   augment, positive, cutoff, useVAT, train_ebm, log_interval)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       save_file)
            # save model only
            torch.save(model.state_dict(), 'state_' + save_file)

            if bk:
                break

    def train_ebm(self, noise_samples, epochs=50, batch_size=100, lr=1e-4, wd=1e-5, augment=True, positive=False,
                  finalLayerOnly=False, useVAT=False, start_from_ckpt=False, log_interval=100):

        # generate noise data from flow, and setup contrastive dataset

        dataset = ContrastiveMNIST(self.data_path, noise_samples, train=True, download=True,
                                   transform=transforms.ToTensor(),
                                   # segment_transform=mnist_one_hot_transform,  # 10D one-hot for classes
                                   target_transform=contrastive_one_hot_transform)  # 2D one-hot for true or fake
        fce_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **self._loader_params)

        model = self.ebm
        # define optimizer
        if finalLayerOnly:
            optimizer = optim.Adam(list(self.ebm.g.parameters()) + [self.ebm.log_norm], lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(self.ebm.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        if self.ebm_restore_file:
            state = torch.load(self.ebm_restore_file, map_location=self.device)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(self.ebm_restore_file)

        # if model.is_cuda:
        #    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #    cudnn.benchmark = True

        model.train()
        for i in range(start_epoch, start_epoch + epochs):
            bk = self._train_step(fce_loader, optimizer, i, start_epoch + epochs,
                                  augment, positive, useVAT=useVAT, log_interval=log_interval,
                                  cutoff=1, train_ebm=True)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       self.ebm_save_file)
            # save model only
            torch.save(model.state_dict(), 'state_' + self.ebm_save_file)

            if bk:
                break

        pass

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


class CleanFCE:
    def __init__(self, EBM, flow, dataset, test_dataset=None, device='cpu',
                 data_path='../data/', output_dir='../output/',
                 pretrain_results_file='pretrain_results.txt', pretrain_restore_file=None,
                 results_file='results.txt', restore_ebm_file=None, restore_flow_file=None):
        """
        This is written for an EBM and a MAF from nflib/maf.py
        """

        self.ebm = EBM.to(device)
        self.flow = flow.to(device)
        self.device = device

        self.dataset = dataset
        # self.test_dataset = test_dataset

        self._len_dset = len(dataset)
        # self._len_test_dset = len(test_dataset)

        self.input_size = self.ebm.input_size
        self.condition_size = self.ebm.condition_size

        use_cuda = device == 'cuda' and torch.cuda.is_available()
        self._loader_params = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        # logging and checkpoints
        self.data_path = data_path
        self.output_dir = output_dir
        self.pretrain_results_file = pretrain_results_file
        self.pretrain_restore_file = pretrain_restore_file
        self.results_file = results_file
        self.restore_ebm_file = restore_ebm_file
        self.restore_flow_file = restore_flow_file

        self.conditional_flow = self.flow.condition_size is not None

    def sample_noise(self, n, chunk=300):
        samples = []
        nfc = n // chunk
        rem = n % chunk
        for i in range(nfc):
            samples += [self.flow.sample(chunk).detach().cpu()]
        samples += [self.flow.sample(rem).detach().cpu()]
        return torch.cat(samples, dim=0)

    def flow_log_pdf(self, x, y):
        return self.flow.log_pdf(x, y)

    def ebm_log_pdf(self, x, y, *args, **kwargs):
        return self.ebm.log_pdf(x, y, *args, **kwargs)

    def _pretrain_train_step(self, dataloader, optimizer, epoch, epochs, log_interval=100):
        for i, (x, y) in enumerate(dataloader):
            self.flow.train()
            # this is written specifically for a labeled dataset

            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)  # in case x is in image

            # the label might not be used if we want the flow to be unconditional
            loss = - self.flow.log_pdf(x, y if self.conditional_flow else None).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                log_output = 'epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(epoch, epochs, i, len(dataloader),
                                                                                     loss.item())
                print(log_output)
                print(log_output, file=open(self.pretrain_results_file, 'a'))

    @torch.no_grad()
    def _pretrain_evaluate_step(self, dataloader, epoch):
        self.flow.eval()

        # conditional model
        if self.conditional_flow:
            cond_size = self.flow.condition_size
            logprior = torch.tensor(1 / cond_size).log().to(self.device)
            loglike = [[] for _ in range(cond_size)]

            for i in range(cond_size):
                # make one-hot labels
                labels = torch.zeros(dataloader.batch_size, cond_size).to(self.device)
                labels[:, i] = 1

                for x, y in dataloader:
                    x = x.view(x.shape[0], -1).to(self.device)
                    loglike[i].append(self.flow.log_pdf(x, labels))

                loglike[i] = torch.cat(loglike[i], dim=0)  # cat along data dim under this label
            loglike = torch.stack(loglike, dim=1)  # cat all data along label dim

            logprobs = logprior + loglike.logsumexp(dim=1)

        # unconditional model
        else:
            logprobs = []
            for data in dataloader:
                x = data[0].view(data[0].shape[0], -1).to(self.device)
                logprobs.append(self.flow.log_pdf(x))
            logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
        output = 'Evaluate ' + (epoch != None) * '(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(
            logprob_mean, logprob_std)
        print(output)
        print(output, file=open(self.pretrain_results_file, 'a'))
        return logprob_mean, logprob_std

    @torch.no_grad()
    def _pretrain_generate(self, lam=1e-6, step=None, n_row=None):
        self.flow.eval()

        if n_row is None:
            n_row = 10

        # conditional model
        if self.conditional_flow:
            cond_size = self.flow.condition_size
            samples = []
            labels = torch.eye(cond_size).to(self.device)

            for i in range(cond_size):
                # sample model base distribution and run through backward model to sample data space
                u = self.flow.base_dist.sample((n_row, 1)).squeeze()
                labels_i = labels[i].expand(n_row, -1)
                sample, _ = self.flow.backward(u, labels_i)
                log_probs = self.flow.log_pdf(sample, labels_i).sort(0)[1].flip(
                    0)  # sort by log_prob; take argsort idxs; flip high to low
                samples.append(sample[log_probs])

            samples = torch.cat(samples, dim=0)

        # unconditional model
        else:
            u = self.flow.base_dist.sample((n_row ** 2, 1)).squeeze()
            samples, _ = self.flow.backward(u)
            log_probs = self.flow.log_pdf(samples).sort(0)[1].flip(
                0)  # sort by log_prob; take argsort idxs; flip high to low
            samples = samples[log_probs]

        # convert and save images
        samples = samples.view(samples.shape[0], 1, 28, 28)  # TODO
        samples = (torch.sigmoid(samples) - lam) / (1 - 2 * lam)
        filename = 'generated_samples' + (step != None) * '_epoch_{}'.format(step) + '.png'
        save_image(samples, os.path.join(self.output_dir, filename), nrow=n_row, normalize=True)

    def pretrain_flow(self, epochs=20, batch_size=100, lr=1e-4, wd=1e-5, evaluate=False):
        # pretrain the flow model using MLE to have a good approximation of the data density
        # before the contrastive phase
        print('Pretraining flow model')

        train_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, **self._loader_params)

        optimizer = optim.Adam(self.flow.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        if self.pretrain_restore_file:
            print('Loading model from checkpoints..')
            state = torch.load(self.pretrain_restore_file, map_location=self.device)
            self.flow.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(self.pretrain_restore_file)
        self.pretrain_results_file = os.path.join(self.output_dir, self.pretrain_results_file)

        best_eval_logprob = float('-inf')

        for i in range(start_epoch, start_epoch + epochs):
            self._pretrain_train_step(train_dataloader, optimizer, i, start_epoch + epochs)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': self.flow.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       os.path.join(self.output_dir, 'pretrained_flow_checkpoint.pt'))
            # save model only
            torch.save(self.flow.state_dict(), os.path.join(self.output_dir, 'pretrained_flow_state.pt'))

            # save best eval state
            if evaluate:
                test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                             **self._loader_params)

                eval_logprob, _ = self._pretrain_evaluate_step(test_dataloader, i)
                if eval_logprob > best_eval_logprob:
                    best_eval_logprob = eval_logprob
                    torch.save({'epoch': i,
                                'model_state': self.flow.state_dict(),
                                'optimizer_state': optimizer.state_dict()},
                               os.path.join(self.output_dir, 'best_pretrained_flow_checkpoint.pt'))
                # plot sample
                self._pretrain_generate(step=i)

    def _train_step(self, fce_loader, optimizer, epoch, epochs, augment, positive, cutoff, useVAT, train_ebm):
        objConstant = 2. * train_ebm - 1
        loss_criterion = nn.BCELoss()
        num_correct = 0
        loss_hist = []
        for i, (x, y, label) in enumerate(fce_loader):
            y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)
            label = label.to(self.device)

            flow_logpdf = self.flow_log_pdf(x, y).view(-1, 1)
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

            loss_hist += [loss.item]

        # print some statistics
        n = len(fce_loader.dataset)
        accuracy = num_correct / n
        network = train_ebm * 'ebm' + (1 - train_ebm) * 'flow'
        log_output = '{}: epoch {}/{};\tloss: {:.4f};\taccuracy: {:.3f}'.format(network, epoch, epochs,
                                                                                loss.item(), accuracy)
        print(log_output)
        print(log_output, file=open(self.results_file, 'a'))

        bk = False
        if accuracy > cutoff:
            # stop training
            log_output = 'accuracy {:.3f}/{} cutoff value satisfied .. stopping training\n----------\n'.format(accuracy,
                                                                                                               cutoff)
            print(log_output)
            print(log_output, file=open(self.results_file, 'a'))
            bk = True
        return bk

    def train(self, epochs=50, batch_size=100, lr=1e-4, wd=1e-5, network='ebm',
              augment=True, positive=False, finalLayerOnly=False, useVAT=False,
              cutoff=None, start_from_ckpt=False):

        # setup
        if network.lower() == 'ebm':
            train_ebm = True
            model = self.ebm
            restore_file = self.restore_ebm_file
        elif network.lower() == 'flow':
            restore_file = self.restore_flow_file
            train_ebm = False
            model = self.flow
        else:
            raise ValueError('wrong network {}'.format(network))
        if cutoff is None:
            cutoff = 1. * train_ebm
        if not start_from_ckpt:
            restore_file = None
        network = network.lower()

        print('FCE step for the {}'.format(network))

        # generate noise data from flow, and setup contrastive dataset
        print('Generating samples from flow ...')
        st = time.time()

        images = self.sample_noise(20).detach().cpu().numpy()
        lbls = list(np.arange(52)) * 20
        self.labels = []
        for i in range(52):
            self.labels += [i] * 20

        img_transform = T.Compose([T.Resize((20, 20)), T.ToTensor()])
        target_transform = lambda t: single_one_hot_encode(t, n_labels=2)
        label_transform = lambda l: single_one_hot_encode(l, n_labels=52)
        dataset = ContrastiveAlphabet(noise_samples=(images, lbls),
                                      transform=img_transform,
                                      target_transform=target_transform,
                                      label_transform=label_transform)

        fce_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **self._loader_params)
        print('... done in {}!'.format(time.time() - st))

        # define optimizer
        if finalLayerOnly:
            if not train_ebm:
                raise ValueError("The EBM's g network can't be optimized when training the Flow")
            optimizer = optim.Adam(list(self.ebm.g.parameters()) + [self.ebm.log_norm], lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        start_epoch = 0
        if restore_file:
            state = torch.load(restore_file, map_location=self.device)
            self.flow.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
            # set up paths
            self.output_dir = os.path.dirname(restore_file)
        self.results_file = os.path.join(self.output_dir, self.results_file)

        # if model.is_cuda:
        #    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #    cudnn.benchmark = True

        model.train()
        for i in range(start_epoch, start_epoch + epochs):
            bk = self._train_step(fce_loader, optimizer, i, start_epoch + epochs,
                                  augment, positive, cutoff, useVAT, train_ebm)

            # save training checkpoint
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       os.path.join(self.output_dir, '{}_checkpoint.pt'.format(network)))
            # save model only
            torch.save(model.state_dict(), os.path.join(self.output_dir, '{}_state.pt'.format(network)))

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
