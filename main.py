import argparse
import os
import sys

import numpy as np
import torch
import yaml

from runners.real_data_runner import PreTrainer, semisupervised, transfer


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, help='Dataset to run experiments. Should be MNIST or CIFAR10, or FMNIST')
    parser.add_argument('--config', type=str, default='mnist.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='', help='A string for documentation purpose')

    parser.add_argument('--nSims', type=int, default=5, help='Number of simulations to run')
    parser.add_argument('--SubsetSize', type=int, default=6000,
                        help='Number of data points per class to consider -- only relevant for transfer learning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--all', action='store_true',
                        help='Run transfer learning experiment for many seeds and subset sizes -- only relevant for transfer learning')
    parser.add_argument('--baseline', action='store_true', help='Run the script for the baseline')
    parser.add_argument('--semisupervised', action='store_true', help='Run semi-supervised experiments')
    parser.add_argument('--transfer', action='store_true',
                        help='Run the transfer learning experiments after pretraining')

    parser.add_argument('--plot', action='store_true',
                        help='Plot transfer learning experiment for the selected dataset')

    args = parser.parse_args()
    args.dataset = args.dataset.upper()
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

    if not args.transfer and not args.semisupervised and not args.baseline and not args.plot:
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

    # PLOTTING TRANSFER LEARNING
    # 1- just use of the flag --plot AND NO other flag (except --dataset of course)
    if args.plot and not args.baseline and not args.semisupervised and not args.transfer:
        plot(args)


def plot(args):
    import pickle
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette('deep')

    # collect results for transfer learning
    samplesSizes = [500, 1000, 2000, 3000, 5000, 6000]

    resTransfer = {x: [] for x in samplesSizes}
    resBaseline = {x: [] for x in samplesSizes}

    # load transfer results
    for x in samplesSizes:
        files = [f for f in os.listdir(args.run) if args.dataset.lower() + 'TransferCDSM_Size' + str(x) + '_' in f]
        for f in files:
            resTransfer[x].append(np.median(pickle.load(open(os.path.join(args.run, f), 'rb'))))

        files = [f for f in os.listdir(args.run) if args.dataset + '_Baseline_Size' + str(x) + '_' in f]
        for f in files:
            resBaseline[x].append(np.median(pickle.load(open(os.path.join(args.run, f), 'rb'))))

        print(
            'Transfer: ' + str(np.median(resTransfer[x]) * 1e4) + '\tBaseline: ' + str(np.median(resBaseline[x]) * 1e4))

    resTsd = np.array([np.std(resTransfer[x]) * 1e4 for x in samplesSizes])

    resT = np.array([np.median(resTransfer[x]) * 1e4 for x in samplesSizes])
    resBas = np.array([np.median(resBaseline[x]) * 1e4 for x in samplesSizes])

    f, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
    ax1.plot(samplesSizes, resT, label='Transfer', linewidth=2, color=sns.color_palette()[2])
    ax1.fill_between(samplesSizes, resT + 2 * resTsd, resT - 2 * resTsd, alpha=.25, color=sns.color_palette()[2])
    ax1.plot(samplesSizes, resBas, label='Baseline', linewidth=2, color=sns.color_palette()[4])
    ax1.legend()
    ax1.set_xlabel('Train dataset size')
    ax1.set_ylabel('DSM Objective (scaled)')
    ax1.set_title('Conditional DSM Objective')
    f.tight_layout()
    plt.savefig(os.path.join(args.run, 'transfer_{}.pdf'.format(args.dataset.lower())))


if __name__ == '__main__':
    sys.exit(main())
