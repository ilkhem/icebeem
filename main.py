"""
main file: chose a runner and a config file and run

usage:
    python3 main.py --dataset TCL --method iVAE --nSims 10

"""

import argparse 
from runners import ivae_exp_runner, icebeem_exp_runner, mnist_exp_runner #, tcl_exp_runner

import pickle 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help='dataset to run experiments. Should be TCL, IMCA or MNIST')
parser.add_argument('--method', type=str, help='method to employ. Should be TCL, iVAE or ICE-BeeM')
parser.add_argument('--nSims', type=int, help='number of simulations to run')
# following two arguments are only relevant for mnist data experiments (will be ignored otherwise)
parser.add_argument('--config', type=str, default='mnist.yml',  help='Path to the config file')
parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
parser.add_argument('--test', action='store_true', help='Whether to test the model')
parser.add_argument('--nSegments', type=int, default=7)
parser.add_argument('--SubsetSize', type=int, default=1000) # only relevant for transfer learning baseline, otherwise ignored

args = parser.parse_args()

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
            r = icebeem_exp_runner.runICEBeeMexp( nSims=args.nSims , simulationMethod=args.method) 

        # save results
        fname = 'results/' + args.method + 'res_' + args.dataset + 'exp_' + str(args.nSims) + '.p'
        pickle.dump( r, open( fname, "wb" ))


    if args.dataset == 'MNIST':
        args.log = os.path.join(args.run, 'logs', args.doc)

        runner = eval(args.runner)( args, config, nSeg = args.nSegments, subsetSize=args.SubsetSize )
        if not args.test:
            runner.train()
        else:
            runner.test()

