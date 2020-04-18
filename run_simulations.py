"""
main file: chose a runner and a config file and run

usage:
    python3 main.py --dataset TCL --method iVAE --nSims 10

"""

import argparse
import os
import pickle

import torch
import yaml

from runners import ivae_exp_runner, icebeem_exp_runner# , tcl_exp_runner

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='TCL', help='dataset to run experiments. Should be TCL or IMCA')
parser.add_argument('--method', type=str, default='icebeem', help='method to employ. Should be TCL, iVAE or ICE-BeeM')
parser.add_argument('--nSims', type=int, default=10, help='number of simulations to run')
parser.add_argument('--config', type=str, default='imca.yaml', help='Path to the config file')
parser.add_argument('--run', type=str, default='run/', help='Path for saving running related data.')
parser.add_argument('--test', action='store_true', help='Whether to test the model')

args = parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == '__main__':

    print('Running {} experiments using {}'.format(args.dataset, args.method))
    os.makedirs(args.run, exist_ok=True)
    fname = os.path.join(args.run, args.method + 'res_' + args.dataset + 'exp.p')
    print(fname)

    if args.dataset.lower() in ['tcl', 'imca']:
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.load(f)
        new_config = dict2namespace(config)
        new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if args.method.lower() == 'tcl':
            r = tcl_exp_runner.runTCLexp(args, new_config)
        elif args.method.lower() == 'ivae':
            r = ivae_exp_runner.runiVAEexp(args, new_config)
        elif args.method.lower() in ['ice-beem', 'icebeem']:
            r = icebeem_exp_runner.runICEBeeMexp(args, new_config)
        else:
            raise ValueError('Unsupported method {}'.format(args.method))

        # save results
        fname = os.path.join(args.run, args.method + 'res_' + args.dataset + 'exp_' + str(args.nSims) + '.p')
        pickle.dump(r, open(fname, "wb"))

    else:
        raise ValueError('Unsupported dataset {}'.format(args.dataset))
