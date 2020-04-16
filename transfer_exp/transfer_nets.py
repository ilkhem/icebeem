### transfer learning on estimated networks !
#
#

import numpy as np 
import os 
import torch 
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pickle
from torch.utils.data.dataloader import default_collate

# load in the required modules
from models.refinenet_dilated_baseline import RefineNetDilated

import sys
args = sys.argv
SUBSET_SIZE = int(args[1])
SEED        = int(args[2])

print('SUBSET SIZE: ' + str(SUBSET_SIZE) +'\tSEED: ' + str(SEED))

# load config
config = pickle.load( open('transfer_exp/config_file.p', 'rb' ) )

expFolder = 'mnistPreTrain'
checkpoint = '' # '1000'

check_path = expFolder + '/' +'checkpoint' + checkpoint + '.pth'

# load in states
onCluster = True 
if onCluster:
	ckp_path = 'run/logs/' + check_path
	states = torch.load( ckp_path, map_location='cuda:0')

	# define score
	score = RefineNetDilated(config).to('cuda:0')
	score = torch.nn.DataParallel(score)
	score.load_state_dict(states[0])
else:
	ckp_path = '/home/projects/IMCA/ncsn-master/run/logs/' + check_path
	states = torch.load( ckp_path, map_location='cpu')

	# define score
	score = RefineNetDilated(config).to('cpu')
	score = torch.nn.DataParallel(score)
	score.load_state_dict(states[0])

print('loaded energy network')
# load the config

# now load in the data 
if config.data.random_flip is False:
    tran_transform = test_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor()
    ])
else:
    tran_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor()
    ])

test_dataset = MNIST('datasets/mnist_test', train=False, download=True, transform=test_transform)

# define the subset

id_range = list(range(SUBSET_SIZE))
testset_1 = torch.utils.data.Subset(test_dataset, id_range)

#dataloader = DataLoader( trainset_1, batch_size=config.training.batch_size, shuffle=True, num_workers=1 )

def my_collate(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label in range(8,10):
            modified_batch.append(item)
    return default_collate(modified_batch)


test_loader = DataLoader(testset_1, batch_size=config.training.batch_size, shuffle=True, num_workers=1, drop_last=True, collate_fn = my_collate )

print('loaded test data')

if onCluster == False:
	config.device = 'cpu'

energy_net_finalLayer = torch.ones((  config.data.image_size * config.data.image_size, 2 )).to(config.device)
energy_net_finalLayer.requires_grad_()

from losses.dsm import  conditional_dsm # dsm_score_estimation
import torch.optim as optim
import logging

# define the optimizer
parameters = [ energy_net_finalLayer ]
optimizer = optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                              betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad)

# start optimizing!
step = 0
eCount = 10
for epoch in range( eCount ):
    print('epoch: ' + str(epoch))
    counter = 0
    loss_track = []
    for i, (X, y) in enumerate(test_loader):
        #print(step)
        step += 1

        X = X.to(config.device)
        X = X / 256. * 255. + torch.rand_like(X) / 256.

        # replace this with either dsm or dsm_conditional_score_estimation function !!
        y = y - y.min() # make zero indexed for conditional_dsm function
        loss = conditional_dsm(score, X, y, energy_net_finalLayer,  sigma=0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #tb_logger.add_scalar('loss', loss, global_step=step)
        #print(loss.item())
        #logging.info("step: {}, loss: {}, maxLabel: {}".format(step, loss.item(), y.max()))
        loss_track.append( loss.item() )



# save loss_track somewhere! 
generateLangevinSamples = False

if generateLangevinSamples==False:
    os.chdir('transfer_exp/transferRes')
    pickle.dump( loss_track, open('TransferCDSM_Size' + str(SUBSET_SIZE) + "_Seed" + str(SEED) + '.p', 'wb' ))
else:
    pass




