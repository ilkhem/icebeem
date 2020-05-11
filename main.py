import argparse
import os
import sys

import numpy as np
import torch
import yaml

from runners.real_data_runner import train, semisupervised, transfer, cca_representations, plot_transfer, \
    plot_representation

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', type=str, default='mnist.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='', help='A string for documentation purpose')

    parser.add_argument('--nSims', type=int, default=5, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--baseline', action='store_true', help='Run the script for the baseline')
    parser.add_argument('--transfer', action='store_true',
                        help='Run the transfer learning experiments after pretraining')
    parser.add_argument('--semisupervised', action='store_true', help='Run semi-supervised experiments')
    parser.add_argument('--representation', action='store_true',
                        help='Run CCA representation validation across multiple seeds')

    parser.add_argument('--subsetSize', type=int, default=6000,
                        help='Number of data points per class to consider -- '
                             'only relevant for transfer learning if not run with --all flag')
    parser.add_argument('--all', action='store_true',
                        help='Run transfer learning experiment for many seeds and subset sizes -- '
                             'only relevant for transfer learning')

    parser.add_argument('--plot', action='store_true',
                        help='Plot selected experiment for the selected dataset')

    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='(LEGACY) Dataset to run experiments. Should be MNIST or CIFAR10, FMNIST, or CIFAR100')

    args = parser.parse_args()
    return args


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
    args.output = os.path.join(args.run, 'output')
    os.makedirs(args.output, exist_ok=True)


def main():
    args = parse()

    with open(os.path.join('configs', args.config), 'r') as f:
        config_raw = yaml.load(f)
    config = dict2namespace(config_raw)
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(config)
    args.dataset = config.data.dataset

    if config.model.positive:
        args.doc += 'p'
    if config.model.augment:
        args.doc += 'a'
    if config.model.final_layer:
        args.doc += str(config.model.feature_size)

    make_dirs(args)
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
    # step 3 is only for debug and shouldn't be used in practice

    if not args.transfer and not args.semisupervised and not args.baseline and not args.plot and not args.representation:
        train(args, config)

    if args.transfer and not args.baseline:
        if not args.all:
            transfer(args, config)
        else:
            new_args = argparse.Namespace(**vars(args))
            for n in [500, 1000, 2000, 3000, 4000, 5000, 6000]:
                for seed in range(args.seed, args.nSims + args.seed):
                    new_args.subsetSize = n
                    new_args.seed = seed
                    # change random seed
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    transfer(new_args, config)

    if config.data.dataset in ["MNIST_transferBaseline", "CIFAR10_transferBaseline"]:
        # this is just here for debug, shouldn't be run, use --baseline --transfer instead
        if not args.all:
            train(args, config)
        else:
            new_args = argparse.Namespace(**vars(args))
            for n in [500, 1000, 2000, 3000, 4000, 5000, 6000]:
                for seed in range(args.seed, args.nSims + args.seed):
                    new_args.subsetSize = n
                    new_args.seed = seed
                    new_args.doc = config.data.dataset.lower() + 'Baseline' + str(n)
                    make_dirs(new_args)
                    # change random seed
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    train(new_args, config)

    if args.transfer and args.baseline:
        # update args and config
        new_args = argparse.Namespace(**vars(args))
        new_args.config = os.path.splitext(args.config)[0] + '_baseline' + os.path.splitext(args.config)[1]
        with open(os.path.join('configs', new_args.config), 'r') as f:
            config_raw = yaml.load(f)
        config = dict2namespace(config_raw)
        config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if not args.all:
            new_args.doc = args.doc + 'Baseline' + str(new_args.subsetSize)
            make_dirs(new_args)
            train(new_args, config)

        else:
            for n in [500, 1000, 2000, 3000, 4000, 5000, 6000]:
                for seed in range(args.seed, args.nSims + args.seed):
                    new_args.doc = args.doc + 'Baseline' + str(n)
                    make_dirs(new_args)
                    new_args.subsetSize = n
                    new_args.seed = seed
                    # change random seed
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    train(new_args, config)

    # PLOTTING TRANSFER LEARNING
    # 1- just use of the flag --plot and --transfer AND NO other flag (except --config of course)
    if args.plot and not args.baseline and not args.semisupervised and args.transfer and not args.representation:
        plot_transfer(args)

    # SEMI-SUPERVISED EXPERIMENTS
    # 1- no special flag: pretrain icebeem on 0-7 // same as 1- above
    # 2- --semisupervised: classify 8-9 using pretrained icebeem // --doc should be the same as from step 1-
    # 3- --baseline: pretrain unconditional ebm  on 0-7: IT IS VERY IMPORTANT HERE TO SPECIFY A --doc THAT IS
    # DIFFERENT FROM WHEN RUN FOR ICEBEEM
    # 4- --semisupervised --baseline: classify 8-9 using unconditional ebm // --doc should be the same as from step 3-

    if args.baseline and not args.semisupervised and not args.transfer and not args.representation:
        new_args = argparse.Namespace(**vars(args))
        new_args.doc = args.doc + 'Baseline'
        make_dirs(new_args)
        train(new_args, config, conditional=False)

    if args.semisupervised and not args.baseline:
        semisupervised(args, config)

    if args.semisupervised and args.baseline:
        new_args = argparse.Namespace(**vars(args))
        new_args.doc = args.doc + 'Baseline'
        make_dirs(new_args)
        semisupervised(new_args, config)

    # COMPARE QUALITY OF REPRESENTATIONS ON REAL DATA
    # 1- --representation: trains ICE-BeeM on train dataset, and save learnt rep of test data for
    # different seeds
    # 2- --representation --baseline: trains unconditional EBM on train dataset, and save learnt rep of test data for
    # different seeds

    if args.representation:
        # overwrite n_labels to full dataset
        config.n_labels = 10 if config.data.dataset.lower().split('_')[0] != 'cifar100' else 100
        if not args.baseline:
            for seed in range(args.seed, args.nSims + args.seed):
                new_args = argparse.Namespace(**vars(args))
                new_args.seed = seed
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                new_args.doc = args.doc + config.data.dataset + '_Representation' + str(new_args.seed)
                print(new_args.doc)
                make_dirs(new_args)
                cca_representations(new_args, config)

        else:
            # train unconditional EBMs
            for seed in range(args.seed, args.nSims + args.seed):
                new_args = argparse.Namespace(**vars(args))
                new_args.seed = seed
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                new_args.doc = args.doc + config.data.dataset + '_RepresentationBaseline' + str(new_args.seed)
                print(new_args.doc)
                make_dirs(new_args)
                cca_representations(new_args, config, conditional=False)

    # PLOTTING REPRESENTATIONS BOXPLOT
    # 1- just use of the flag --plot and representation AND NO other flag (except --config of course) to
    # compute MCC before and after applying a CCA to the rep and display as boxplot
    if args.plot and not args.baseline and not args.semisupervised and not args.transfer and args.representation:
        plot_representation(args)


if __name__ == '__main__':
    sys.exit(main())
