# copied shamelessly from https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

import math
import os

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [
            input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [
            input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)  # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        #        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
        #            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def backward(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y=None):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def backward(self, u, y=None):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.backward(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


# --------------------
# Models
# --------------------

class MADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu',
                 input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'lrelu':
            activation_fn = nn.LeakyReLU(0.2)
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def backward(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MAF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu',
                 input_order='sequential', batch_norm=True, device='cuda'):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.condition_size = cond_label_size
        self.conditional_flow = cond_label_size is not None

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

        self.device = device

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def backward(self, u, y=None):
        return self.net.backward(u, y)

    def log_pdf(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

    @torch.no_grad()
    def sample(self, n_samples, output_dir='./'):
        """
        If the model is conditional, generate n_samples per condition.
        If the model is NOT conditional, just generate n_samples
        """
        self.eval()
        # conditional model
        if self.conditional_flow:
            # THIS WORKS ONLY FOR DISCRETE CONDITONING VARIABLES
            cond_size = self.condition_size
            samples = []
            labels = torch.eye(cond_size).to(self.device)

            for i in range(cond_size):
                # sample model base distribution and run through backward model to sample data space
                u = self.base_dist.sample((n_samples, 1)).squeeze()
                labels_i = labels[i].expand(n_samples, -1)
                sample, _ = self.backward(u, labels_i)
                samples.append(sample)

            samples = torch.cat(samples, dim=0)

        # unconditional model
        else:
            u = self.base_dist.sample((n_samples, 1)).squeeze()
            samples, _ = self.backward(u)

        # convert and save images
        lam = 1e-6
        sz = int(math.sqrt(self.input_size))
        samples = samples.view(samples.shape[0], 1, sz, sz)
        samples = (torch.sigmoid(samples) - lam) / (1 - 2 * lam)
        filename = 'generated_samples.png'
        save_image(samples, os.path.join(output_dir, filename), nrow=n_samples, normalize=True)

        return samples


class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead,
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward


# Stuff for training generic models

def train(model, dataloader, optimizer, epoch, args):
    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(args.device)
        x = x.view(x.shape[0], -1).to(args.device)

        loss = - model.log_prob(x, y if args.cond_label_size else None).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch, args.start_epoch + args.n_epochs, i, len(dataloader), loss.item()))


@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    # conditional model
    if args.cond_label_size is not None:
        logprior = torch.tensor(1 / args.cond_label_size).log().to(args.device)
        loglike = [[] for _ in range(args.cond_label_size)]

        for i in range(args.cond_label_size):
            # make one-hot labels
            labels = torch.zeros(args.batch_size, args.cond_label_size).to(args.device)
            labels[:, i] = 1

            for x, y in dataloader:
                x = x.view(x.shape[0], -1).to(args.device)
                loglike[i].append(model.log_prob(x, labels))

            loglike[i] = torch.cat(loglike[i], dim=0)  # cat along data dim under this label
        loglike = torch.stack(loglike, dim=1)  # cat all data along label dim

        # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
        # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
        logprobs = logprior + loglike.logsumexp(dim=1)
        # TODO -- measure accuracy as argmax of the loglike

    # unconditional model
    else:
        logprobs = []
        for data in dataloader:
            x = data[0].view(data[0].shape[0], -1).to(args.device)
            logprobs.append(model.log_prob(x))
        logprobs = torch.cat(logprobs, dim=0).to(args.device)

    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
    output = 'Evaluate ' + (epoch != None) * '(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(
        logprob_mean, logprob_std)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std


@torch.no_grad()
def generate(model, dataset_lam, args, step=None, n_row=10):
    model.eval()

    # conditional model
    if args.cond_label_size:
        samples = []
        labels = torch.eye(args.cond_label_size).to(args.device)

        for i in range(args.cond_label_size):
            # sample model base distribution and run through backward model to sample data space
            u = model.base_dist.sample((n_row, 1)).squeeze()
            labels_i = labels[i].expand(n_row, -1)
            sample, _ = model.backward(u, labels_i)
            log_probs = model.log_prob(sample, labels_i).sort(0)[1].flip(
                0)  # sort by log_prob; take argsort idxs; flip high to low
            samples.append(sample[log_probs])

        samples = torch.cat(samples, dim=0)

    # unconditional model
    else:
        u = model.base_dist.sample((n_row ** 2, 1)).squeeze()
        samples, _ = model.backward(u)
        log_probs = model.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
        samples = samples[log_probs]

    # convert and save images
    samples = samples.view(samples.shape[0], 1, 28, 28)
    samples = (torch.sigmoid(samples) - dataset_lam) / (1 - 2 * dataset_lam)
    filename = 'generated_samples' + (step != None) * '_epoch_{}'.format(step) + '.png'
    save_image(samples, os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)


def train_and_evaluate(model, train_loader, test_loader, optimizer, args):
    best_eval_logprob = float('-inf')

    for i in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train(model, train_loader, optimizer, i, args)
        eval_logprob, _ = evaluate(model, test_loader, i, args)

        # save training checkpoint
        torch.save({'epoch': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                   os.path.join(args.output_dir, 'model_checkpoint.pt'))
        # save model only
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))

        # save best state
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       os.path.join(args.output_dir, 'best_model_checkpoint.pt'))

        # plot sample
        lam = 1e-6
        generate(model, lam, args, step=i)
