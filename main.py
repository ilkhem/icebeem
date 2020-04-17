"""
main file: chose a runner and a config file and run

usage:
    python3 main.py --dataset TCL --method iVAE --nSims 10

"""

import argparse 
from runners import ivae_exp_runner, icebeem_exp_runner, mnist_exp_runner, mnist_unconditional_exp_runner #, tcl_exp_runner

import os 
import pickle
import yaml 
import torch 
import shutil

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help='dataset to run experiments. Should be TCL, IMCA or MNIST')
parser.add_argument('--method', type=str, default='dsm', help='method to employ. Should be TCL, iVAE or ICE-BeeM')
parser.add_argument('--nSims', type=int, default=5, help='number of simulations to run')
parser.add_argument('--lr_flow', type=float, default=1e-5, help='learning rate for flow in FCE (should be smaller than lr for EBM as suggested in Gao et al (2019))')
parser.add_argument('--lr_ebm', type=float, default=0.0003, help='learning rate for EBM')
parser.add_argument('--n_layer_flow', type=int, default=10, help='depth of flow network in FCE')
# following two arguments are only relevant for mnist data experiments (will be ignored otherwise)
parser.add_argument('--config', type=str, default='mnist.yml',  help='Path to the config file')
parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
parser.add_argument('--test', action='store_true', help='Whether to test the model')
parser.add_argument('--nSegments', type=int, default=7)
parser.add_argument('--SubsetSize', type=int, default=6000) # only relevant for transfer learning baseline, otherwise ignored
parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--unconditionalBaseline', type=int, default=0, help='should we run an unconditional baseline for EBMs - default no')


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

    fname = 'results/' + args.method + 'res_' + args.dataset +'exp.p'
    print(fname)

    if args.dataset in ['TCL', 'IMCA']:
        if args.method.lower() == 'tcl':
            r = tcl_exp_runner.runTCLexp( nSims=args.nSims, simulationMethod=args.method )
        if args.method.lower() == 'ivae':
            r = ivae_exp_runner.runiVAEexp( nSims=args.nSims , simulationMethod=args.method )
        if args.method.lower() in ['ice-beem', 'icebeem']:
            r = icebeem_exp_runner.runICEBeeMexp( nSims=args.nSims , simulationMethod=args.method, lr_flow=args.lr_flow, lr_ebm=args.lr_ebm, n_layers_flow=args.n_layer_flow ) 

        # save results
        fname = 'results/' + args.method + 'res_' + args.dataset + 'exp_' + str(args.nSims) + '.p'
        pickle.dump( r, open( fname, "wb" ))


    if args.dataset == 'MNIST':
        if args.unconditionalBaseline == 0:
            # train conditional EBM
            args.log = os.path.join(args.run, 'logs', args.doc)

            # prepare directory to save results
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            print('saving in: ' + args.log )
            os.makedirs(args.log)

            with open(os.path.join('configs', args.config), 'r') as f:
                config = yaml.load(f)
            new_config = dict2namespace(config)
            new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            #config_file = yaml.load( args.config )
            print(new_config)

            torch.backends.cudnn.benchmark = True

            runner = mnist_exp_runner.mnist_runner( args, new_config, nSeg = args.nSegments, subsetSize=args.SubsetSize, seed=args.seed )
            if not args.test:
                runner.train()
            else:
                runner.test()
        if args.unconditionalBaseline == 1:
            print('\n\n\n\n\nunconditional baseline')
            # train an unconditional EBM using DSM
            args.log = os.path.join(args.run, 'logs', args.doc)

            # prepare directory to save results
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            print('saving in: ' + args.log )
            os.makedirs(args.log)

            with open(os.path.join('configs', args.config), 'r') as f:
                config = yaml.load(f)
            new_config = dict2namespace(config)
            new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            #config_file = yaml.load( args.config )
            print(new_config)

            torch.backends.cudnn.benchmark = True

            runner = mnist_unconditional_exp_runner.mnist_ucond_runner( args, new_config, nSeg = args.nSegments, subsetSize=args.SubsetSize, seed=args.seed )
            if not args.test:
                runner.train()
            else:
                runner.test()


    if args.dataset == 'CIFAR10':
        if args.unconditionalBaseline == 0:
            # train conditional EBM
            args.log = os.path.join(args.run, 'logs', args.doc)

            # prepare directory to save results
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            print('saving in: ' + args.log )
            os.makedirs(args.log)

            with open(os.path.join('configs', args.config), 'r') as f:
                config = yaml.load(f)
            new_config = dict2namespace(config)
            new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            #config_file = yaml.load( args.config )
            print(new_config)
            pickle.dump( new_config, open('transfer_exp/config_file_cifar.p', 'wb'))

            torch.backends.cudnn.benchmark = True

            runner = mnist_exp_runner.mnist_runner( args, new_config, nSeg = args.nSegments, subsetSize=args.SubsetSize, seed=args.seed )
            if not args.test:
                runner.train()
            else:
                runner.test()
        if args.unconditionalBaseline == 1:
            print('\n\n\n\n\nunconditional baseline')
            # train an unconditional EBM using DSM
            args.log = os.path.join(args.run, 'logs', args.doc)

            # prepare directory to save results
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            print('saving in: ' + args.log )
            os.makedirs(args.log)

            with open(os.path.join('configs', args.config), 'r') as f:
                config = yaml.load(f)
            new_config = dict2namespace(config)
            new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            #config_file = yaml.load( args.config )
            print(new_config)

            torch.backends.cudnn.benchmark = True

            runner = mnist_unconditional_exp_runner.mnist_ucond_runner( args, new_config, nSeg = args.nSegments, subsetSize=args.SubsetSize, seed=args.seed )
            if not args.test:
                runner.train()
            else:
                runner.test()

# runner for cifar:
# python3 main.py --dataset CIFAR10 --config cifar.yaml --doc cifarPreTrain



