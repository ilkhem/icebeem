### run remaining scripts
#
#

import argparse 

import os 
import pickle
import yaml 
import torch 
import shutil

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help='dataset - should MNIST, CIFAR10 or FashionMNIST')
parser.add_argument('--run_semisupervised', type=int, default=0, help='run semi-supervised learning on unseen classes')
parser.add_argument('--run_transfer', type=int, default=0, help='run transfer learning experiments ')


args = parser.parse_args()

if __name__=='__main__':
    print('Running experiments on {}'.format(args.dataset))

    if args.dataset=='MNIST':
        if args.run_semisupervised==1:
            print('\n\nrunning semi-supervised learning\n\n')
            os.system( 'python3 transfer_exp/semisupervised.py' )
        if args.run_transfer==1:
            print('\n\nrunning transfer learning\n\n')
            os.system('sh transfer_exp/run_DSMtransfer.sh'
            os.system('sh transfer_exp/run_DSMtransfer_baseline.sh')


    if args.dataset=='CIFAR10':
        print('\n\nrunning semi-supervised learning\n\n')
        if args.run_semisupervised==1:
            os.system( 'python3 transfer_exp/semisupervised_cifar.py' )
        if args.run_transfer==1:
            print('\n\nrunning transfer learning\n\n')

    if args.dataset=='FashionMNIST':
        print('\n\nrunning semi-supervised learning\n\n')
        if args.run_semisupervised==1:
            os.system( 'python3 transfer_exp/semisupervised_fashionmnist.py' )
        if args.run_transfer==1:
            pass 