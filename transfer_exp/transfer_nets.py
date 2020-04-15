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
from models.cond_refinenet_dilated import CondRefineNetDilated
from models.refinenet_dilated_baseline import RefineNetDilated

import sys
args = sys.argv
SUBSET_SIZE = int(args[1])
SEED        = int(args[2])

print('SUBSET SIZE: ' + str(SUBSET_SIZE) +'\tSEED: ' + str(SEED))

# load config
config = pickle.load( open('config_file_mnist.p', 'rb' ) )

expFolder = 'CDSM_8class_v1'
checkpoint = '' # '1000'

check_path = expFolder + '/' +'checkpoint' + checkpoint + '.pth'

# load in states
onCluster = True 
if onCluster:
	ckp_path = '/nfs/ghome/live/ricardom/IMCA/ncsn-master/run/logs/' + check_path
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

from losses.dsm import dsm_score_estimation, conditional_dsm
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
    os.chdir('/nfs/ghome/live/ricardom/IMCA/ncsn-master/transferExperiments/transferRes')
    pickle.dump( loss_track, open('TransferCDSM_Size' + str(SUBSET_SIZE) + "_Seed" + str(SEED) + '.p', 'wb' ))
else:
    print('dumping weights')
    import pickle
    torch.save( [energy_net_finalLayer], 'finalLayerweights_'+ str(eCount)+'.pth')
    pickle.dump( energy_net_finalLayer, open('finalLayerweights.p', 'wb') )
    print('RUNNING LANGEVIN SAMPLING')
    energy_net_finalLayer_1class = energy_net_finalLayer[:,0].view(784,1)
    energy_net_finalLayer_1class.requires_grad_()

    import torch.autograd as autograd

    def Langevin_dynamics(x_mod, scorenet, energy_net_finalLayer_1class, n_steps=1000, step_lr=0.00002):
        #images = []
        print(n_steps)

        #with torch.no_grad():
        for i in range(n_steps):
            #images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
            x_mod.requires_grad_()
            noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
            noise.requires_grad_()

            d = x_mod.shape[-1]
            logp = -scorenet(x_mod).view(-1, d*d)
            logp = torch.mm( logp, energy_net_finalLayer_1class )
            grad = autograd.grad(logp.sum(), x_mod, create_graph=False)[0]

            #grad = scorenet(x_mod)
            x_mod = x_mod + step_lr * grad + noise
            print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            if i % 50 == 0:
                # periodically save results
                os.chdir('/nfs/ghome/live/ricardom/IMCA/ncsn-master/transferExperiments/transferRes')
                pickle.dump({'samples': torch.clamp( x_mod, 0.0, 1.0 ).detach().numpy() }, open('SamplesIter' + str(i)  + 'seed' + str(SEED) +'.p', 'wb'))
                #pickle.dump({'samples': torch.sigmoid( x_mod ).detach().numpy() }, open('SamplesIter' + str(i) + 'seed' + str(SEED) +'.p', 'wb'))

        return None

    data_iter = iter(test_loader)
    samples, _ = next(data_iter)
    samples = torch.rand_like(samples)
    samples.to( config.device )
    print( samples[:3,:,:,:].shape )
    all_samples = Langevin_dynamics( samples[:,:,:,:], score, energy_net_finalLayer_1class, 10000, 0.00002 )



