"""
main file: chose a runner and a config file and run

usage:
    python3 run_simulations.py --dataset TCL --method iVAE --nSims 10


"""

import argparse
import os
import pickle

import torch
import yaml

# from runners import ivae_exp_runner, icebeem_exp_runner, tcl_exp_runner
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


if __name__ == '__main__':
    args = parse_sim()
    print('Running {} experiments using {}'.format(args.dataset, args.method))
    # make checkpoint and log folders
    make_dirs_simulations(args)

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
