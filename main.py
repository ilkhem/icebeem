"""
main file: chose a runner and a config file and run

usage:
    python3 main.py --dataset TCL --method iVAE --nSims 10

"""

import argparse 
from runners import ivae_exp_runner, icebeem_exp_runner
#from runners import tcl_exp_runner 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help='dataset to run experiments. Should be TCL, IMCA or MNIST')
parser.add_argument('--method', type=str, help='method to employ. Should be TCL, iVAE or ICE-BeeM')
parser.add_argument('--nSims', type=int, help='number of simulations to run')

args = parser.parse_args()

if __name__ == '__main__':
    print('Running {} experiments using {}'.format(args.dataset, args.method))

    if args.dataset in ['TCL', 'IMCA']:
        if args.method.lower() == 'tcl':
            print(tcl_exp_runner.runiVAEexp( nSims=args.nSims, simulationMethod=args.method ) )
        if args.method.lower() == 'ivae':
            print(ivae_exp_runner.runiVAEexp( nSims=args.nSims , simulationMethod=args.method ) )
        if args.method.lower() in ['ice-beem', 'icebeem']:
            print( icebeem_exp_runner.runICEBeeMexp( nSims=args.nSims , simulationMethod=args.method) ) 
