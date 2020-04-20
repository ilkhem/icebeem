import argparse
import os
import pickle

import torch
import yaml

from runners.simulation_runner import run_icebeem_exp, run_ivae_exp, run_tcl_exp


def parse_sim():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='TCL', help='Dataset to run experiments. Should be TCL or IMCA')
    parser.add_argument('--method', type=str, default='icebeem',
                        help='Method to employ. Should be TCL, iVAE or ICE-BeeM')
    parser.add_argument('--config', type=str, default='imca.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run/', help='Path for saving running related data.')
    parser.add_argument('--nSims', type=int, default=10, help='Number of simulations to run')

    parser.add_argument('--test', action='store_true', help='Whether to evaluate the models from checkpoints')
    parser.add_argument('--plot', action='store_true', help='Plot comparison of performances')

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


def make_dirs_simulations(args):
    os.makedirs(args.run, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.method)
    os.makedirs(args.checkpoints, exist_ok=True)


def plot(args):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    marker_dict = {'ICEBEEM': 'v', 'iVAE': 'o', 'TCL': 's'}
    line_dict = {'ICEBEEM': 'solid', 'iVAE': '--', 'TCL': ':'}
    legend_dict = {'ICEBEEM': 'ICE-BeeM', 'iVAE': 'iVAE', 'TCL': 'TCL'}
    n_obs_ = [100, 500, 1000, 2000]
    algos = ['ICEBEEM', 'iVAE', 'TCL']

    L = [2, 4]
    # load results
    res = {}
    for a in algos:
        fname = os.path.join(args.run, a + 'res_' + args.dataset + 'exp_' + str(args.nSims) + '.p')
        res[a] = pickle.load(open(fname, 'rb'))['CorrelationCoef']

    f, (ax1) = plt.subplots(1, 1, figsize=(4, 4))

    for l in L:
        for a in algos:
            ax1.plot(n_obs_, [np.mean(res[a][l][n]) for n in n_obs_], label=str(a) + ' (L=' + str(l) + ')',
                     marker=marker_dict[a], color=sns.color_palette()[l], linestyle=line_dict[a], linewidth=2)

    ax1.set_xlabel('Observations per segment')
    ax1.set_ylabel('Mean Correlation Coefficient')

    ax1.set_title(args.dataset)
    ax1.legend(loc='best', fontsize=7)
    f.tight_layout()
    ax1.set_ylim([0, 1])
    plt.savefig(os.path.join(args.run, 'ExpsResults_' + args.dataset + '.pdf'), dpi=300)
    print(os.getcwd())


if __name__ == '__main__':
    args = parse_sim()
    print('Running {} experiments using {}'.format(args.dataset, args.method))
    # make checkpoint and log folders
    make_dirs_simulations(args)

    if not args.plot:
        if args.dataset.lower() in ['tcl', 'imca']:
            with open(os.path.join('configs', args.config), 'r') as f:
                config = yaml.load(f)
            new_config = dict2namespace(config)
            new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            if args.method.lower() == 'tcl':
                r = run_tcl_exp(args, new_config)
            elif args.method.lower() == 'ivae':
                r = run_ivae_exp(args, new_config)
            elif args.method.lower() in ['ice-beem', 'icebeem']:
                r = run_icebeem_exp(args, new_config)
            else:
                raise ValueError('Unsupported method {}'.format(args.method))

            # save results
            # Each of the runners loops over many seeds, so the saved file contains results from multiple runs
            fname = os.path.join(args.run, args.method + 'res_' + args.dataset + 'exp_' + str(args.nSims) + '.p')
            pickle.dump(r, open(fname, "wb"))

        else:
            raise ValueError('Unsupported dataset {}'.format(args.dataset))
    else:
        plot(args)
