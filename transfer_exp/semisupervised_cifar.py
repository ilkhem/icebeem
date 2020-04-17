## evaluate trained energy networks
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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from warnings import filterwarnings
filterwarnings('ignore')

# load in the required modules
from models.refinenet_dilated_baseline import RefineNetDilated

class_model = LinearSVC #LogisticRegression
test_size = .15

### ------ TRANSFER -----
# load config
config = pickle.load( open('transfer_exp/config_file_cifar.p', 'rb' ) )

expFolder = 'cifarPreTrain'
checkpoint = '' # '1000'

check_path = expFolder + '/' +'checkpoint' + checkpoint + '_5000.pth'

# load in states
ckp_path = 'run/logs/' + check_path
states = torch.load( ckp_path, map_location='cuda:0')

# define score
score = RefineNetDilated(config).to('cuda:0')
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

test_dataset = CIFAR10('datasets/cifar10_test', train=False, download=True, transform=test_transform)

print('loaded test data')

def my_collate(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label in range(8,10):
            modified_batch.append(item)
    return default_collate(modified_batch)


test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=1, drop_last=True, collate_fn = my_collate )

representations = np.zeros(( 10000, 32*32*3 ))
labels = np.zeros((10000,))
counter = 0

for i, (X, y) in enumerate(test_loader):
    #print( y )
    #print( X.shape )
    rep_i = score( X ).view(-1,32*32*3).data.cpu().numpy()
    #print( rep_i.shape )
    #print( rep_i.shape[0] )
    representations[ counter:(counter+rep_i.shape[0]),: ] = rep_i
    labels[ counter:(counter+rep_i.shape[0]) ] = y.data.numpy()
    counter += rep_i.shape[0]


representations = representations[:counter, :]
labels = labels[:counter]
print('loaded representations')


labels -= 8
rep_train, rep_test, lab_train , lab_test = train_test_split( (representations), labels, test_size=test_size, random_state=42)

print('training linear SVM')
clf = class_model(random_state=0, max_iter=10000).fit( rep_train, lab_train )
acc = accuracy_score( lab_test, clf.predict( rep_test )) *100

print('Accuracy of transfer representation: acc={}'.format( np.round(acc,2)))

### ------ BASELINE -----
# now repeat with the baseline of unconditional EBM

expFolder = 'cifarUncondBaseline' #'mnistBaseline' + str(ns)
checkpoint = '' # '1000'

check_path = expFolder + '/' +'checkpoint' + checkpoint + '_5000.pth'

# load in states
ckp_path = 'run/logs/' + check_path
states = torch.load( ckp_path, map_location='cuda:0')

# define score
score_base = RefineNetDilated(config).to('cuda:0')
score_base = torch.nn.DataParallel(score_base)
score_base.load_state_dict(states[0])

test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=1, drop_last=True, collate_fn = my_collate )

representations_base = np.zeros(( 10000, 32*32*3 ))
labels_base = np.zeros((10000,))
counter = 0

for i, (X, y) in enumerate(test_loader):
    rep_i = score_base( X ).view(-1,32*32*3).data.cpu().numpy()
    representations_base[ counter:(counter+rep_i.shape[0]),: ] = rep_i
    labels_base[ counter:(counter+rep_i.shape[0]) ] = y.data.numpy()
    counter += rep_i.shape[0]

representations_base = representations_base[:counter, :]
labels_base = labels_base[:counter]
print('loaded representations')

labels_base -= 8
rep_train_b, rep_test_b, lab_train_b , lab_test_b = train_test_split( (representations_base), labels_base, test_size=test_size, random_state=42)

print('training linear SVM')
clf_b = class_model(random_state=0, max_iter=10000).fit( rep_train_b, lab_train_b )
acc_b = accuracy_score( lab_test_b, clf_b.predict( rep_test_b )) *100

print('Accuracy of nonconditional representation learnt: acc={}'.format(np.round(acc_b,2)))

















