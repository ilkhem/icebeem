import argparse
import os
import sys

import numpy as np
import torch
import yaml

from runners.real_data_runner import PreTrainer, semisupervised, transfer


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, help='dataset to run experiments. Should be MNIST or CIFAR10')
    parser.add_argument('--config', type=str, default='mnist.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='', help='A string for documentation purpose')

    parser.add_argument('--nSims', type=int, default=5, help='number of simulations to run')
    parser.add_argument('--SubsetSize', type=int, default=6000,
                        help='only relevant for transfer learning baseline, otherwise ignored')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--all', action='store_true', help='')
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


def make_dirs(args):
    os.makedirs(args.run, exist_ok=True)
    args.log = os.path.join(args.run, 'logs', args.doc)
    os.makedirs(args.log, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.doc)
    os.makedirs(args.checkpoints, exist_ok=True)


def main():
    args = parse()
    make_dirs(args)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(new_config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # These are the possible combinations of flags:

    # TRANSFER LEARNING EXPERIMENTS
    # 1- no special flag: pretrain icebeem on 0-7 // --doc should be different between datasets
    # 2- --transfer: train only g on 8-9 // --doc should be the same as in step 1
    # 3- no special flags but with baseline config: train icebeem on 8-9
    # 4- --transfer --baseline: train icebeem on 8-9 (same as above)
    # steps 2, 3 and 4 are for many seeds and many subset sizes: the user can do them manually, or add the flag --all
    # and the script will perform the loop

    if new_config.data.dataset not in ["MNIST_transferBaseline", "CIFAR10_transferBaseline"]:
        runner = PreTrainer(args, new_config)
        runner.train()

    if args.transfer and not args.baseline:
        if not args.all:
            transfer(args, new_config)
        else:
            new_args = argparse.Namespace(**vars(args))
            for n in [500, 1000, 2000, 3000, 4000, 5000, 6000]:
                for seed in range(args.nSims):
                    new_args.SubsetSize = n
                    new_args.seed = seed
                    # change random seed
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    transfer(new_args, new_config)

    if new_config.data.dataset in ["MNIST_transferBaseline", "CIFAR10_transferBaseline"]:
        # this is just here for debug, shouldn't be run, use --baseline --transfer instead
        if not args.all:
            runner = PreTrainer(args, new_config)
            runner.train()
        else:
            new_args = argparse.Namespace(**vars(args))
            for n in [500, 1000, 2000, 3000, 4000, 5000, 6000]:
                for seed in range(args.nSims):
                    new_args.SubsetSize = n
                    new_args.seed = seed
                    new_args.doc = args.dataset.lower() + 'Baseline' + str(n)
                    make_dirs(new_args)
                    # change random seed
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    runner = PreTrainer(new_args, new_config)
                    runner.train()

    if args.transfer and args.baseline:
        # update args and config
        new_args = argparse.Namespace(**vars(args))
        new_args.config = os.path.splitext(args.config)[0] + '_baseline' + os.path.splitext(args.config)[1]
        with open(os.path.join('configs', new_args.config), 'r') as f:
            config = yaml.load(f)
        new_config = dict2namespace(config)
        new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if not args.all:
            new_args.doc = args.doc + 'Baseline' + str(new_args.SubsetSize)
            make_dirs(new_args)
            runner = PreTrainer(new_args, new_config)
            runner.train()
        else:
            for n in [500, 1000, 2000, 3000, 4000, 5000, 6000]:
                for seed in range(args.nSims):
                    new_args.doc = args.doc + 'Baseline' + str(n)
                    make_dirs(new_args)
                    new_args.SubsetSize = n
                    new_args.seed = seed
                    # change random seed
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    runner = PreTrainer(new_args, new_config)
                    runner.train()

    # SEMI-SUPERVISED EXPERIMENTS
    # 1- no special flag: pretrain icebeem on 0-7 // same as 1- above
    # 2- --semisupervised: classify 8-9 using pretrained icebeem // --doc should be the same as from step 1-
    # 3- --baseline: pretrain unconditional ebm  on 0-7: IT IS VERY IMPORTANT HERE TO SPECIFY A --doc THAT IS
    # DIFFERENT FROM WHEN RUN FOR ICEBEEM
    # 4- --semisupervised --baseline: classify 8-9 using unconditional ebm // --doc should be the same as from step 3-

    if args.baseline and not args.semisupervised and not args.transfer:
        new_args = argparse.Namespace(**vars(args))
        new_args.doc = args.doc + 'Baseline'
        make_dirs(new_args)
        runner = PreTrainer(new_args, new_config)
        runner.train(conditional=False)

    if args.semisupervised and not args.baseline:
        semisupervised(args, new_config)

    if args.semisupervised and args.baseline:
        new_args = argparse.Namespace(**vars(args))
        new_args.doc = args.doc + 'Baseline'
        make_dirs(new_args)
        semisupervised(new_args, new_config)


if __name__ == '__main__':
    sys.exit(main())
