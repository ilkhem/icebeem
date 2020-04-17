"""
main file: chose a runner and a config file and run

usage:
    python3 main.py --dataset TCL --method iVAE --nSims 10

"""

import argparse
import os
import pickle
import shutil

import torch
import yaml

from runners import mnist_exp_runner, mnist_unconditional_exp_runner
from transfer_exp.semisupervised import semisupervised
from transfer_exp.semisupervised_cifar import semisupervised_cifar
from transfer_exp.semisupervised_fashionmnist import semisupervised_fmnist
from transfer_exp.transfer_nets import transfer


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, help='dataset to run experiments. Should be MNIST or CIFAR10')
    parser.add_argument('--method', type=str, default='dsm', help='method to employ. Should be TCL, iVAE or ICE-BeeM')
    parser.add_argument('--nSims', type=int, default=5, help='number of simulations to run')

    # following two arguments are only relevant for mnist data experiments (will be ignored otherwise)
    parser.add_argument('--config', type=str, default='mnist.yml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--SubsetSize', type=int, default=6000,
                        help='only relevant for transfer learning baseline, otherwise ignored')
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--baseline', action='store_true', help='run an unconditional baseline for EBMs')
    parser.add_argument('--semisupervised', action='store_true', help='run semi-supervised experiments')
    parser.add_argument('--transfer', action='store_true',
                        help='run the transfer learning experiments after pretraining')

    return parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def train(args):
    """
    this function trains an icebeem (or just an EBM) on MNIST, CIFAR10, FMNIST
    also works on subsets of these datasets where we omit some classes for a transfer learning experiments
    """
    args.log = os.path.join(args.run, 'logs', args.doc)

    # prepare directory to save results
    if os.path.exists(args.log):
        shutil.rmtree(args.log)
    print('saving in: ' + args.log)
    os.makedirs(args.log)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(new_config)

    if args.dataset == 'MNIST':
        pickle.dump(new_config, open('transfer_exp/config_file.p', 'wb'))
        if args.baseline:
            runner = mnist_exp_runner.mnist_runner(args, new_config)
            if not args.test:
                runner.train()
            else:
                runner.test()
        else:
            print('\nbaseline!!\n')
            runner = mnist_unconditional_exp_runner.mnist_uncond_runner(args, new_config)
            if not args.test:
                runner.train()
            else:
                runner.test()

    elif args.dataset == 'CIFAR10':
        pickle.dump(new_config, open('transfer_exp/config_file_cifar.p', 'wb'))
        if args.baseline:
            runner = mnist_exp_runner.mnist_runner(args, new_config)
            if not args.test:
                runner.train()
            else:
                runner.test()
        else:
            print('\nbaseline!!\n')
            runner = mnist_unconditional_exp_runner.mnist_uncond_runner(args, new_config)
            if not args.test:
                runner.train()
            else:
                runner.test()

    elif args.dataset == 'FashionMNIST':
        pickle.dump(new_config, open('transfer_exp/config_file_fashionMNIST.p', 'wb'))
        if args.unconditionalBaseline == 0:
            runner = mnist_exp_runner.mnist_runner(args, new_config)
            if not args.test:
                runner.train()
            else:
                runner.test()
        if args.unconditionalBaseline == 1:
            print('\nbaseline!!\n')
            runner = mnist_unconditional_exp_runner.mnist_uncond_runner(args, new_config)
            if not args.test:
                runner.train()
            else:
                runner.test()
    else:
        raise ValueError('Unknown dataset {}'.format(args.dataset))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True

    args = parse()
    train(args)

    if args.transfer:

        new_args = argparse.Namespace(**vars(args))
        for n in [500, 1000, 2000, 3000, 4000, 5000, 6000]:
            for seed in range(args.nSims):
                new_args.SubsetSize = n
                new_args.seed = seed
                new_args.config = args.dataset.lower() + '_baseline.yaml'
                new_args.doc = args.dataset.lower() + 'Baseline' + str(n)

                transfer(new_args)
                train(new_args)

    if args.semisupervised:

        print('running semi-supervised learning')
        if args.dataset == 'MNIST':
            semisupervised()
        elif args.dataset == 'CIFAR10':
            semisupervised_cifar()
        elif args.dataset == 'FashionMNIST':
            semisupervised_fmnist()
