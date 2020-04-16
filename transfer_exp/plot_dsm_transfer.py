### plot DSM transfer learning results
#
#

import numpy as np 
import pickle
import os 
import pylab as plt; plt.ion()
import seaborn as sns 

sns.set_style("whitegrid")
sns.set_palette('deep')

# load transfer results
os.chdir('transfer_exp/transferRes')

# collect results for transfer learning
samplesSizes = [500, 1000, 2000, 3000, 5000, 6000]

resTransfer = {x: [] for x in samplesSizes}
resBaseline = {x: [] for x in samplesSizes}

for x in samplesSizes:
	files = [f for f in os.listdir(os.getcwd()) if 'TransferCDSM_Size'+str(x)+'_' in f ]
	for f in files:
		resTransfer[x].append( np.median( pickle.load(open(f, 'rb')) )) 

	files = [f for f in os.listdir(os.getcwd()) if 'Baseline_Size'+str(x)+'_' in f ]
	for f in files:
		resBaseline[x].append( np.median( pickle.load(open(f, 'rb')))) 

	print('Transfer: ' + str(np.median( resTransfer[x]) * 1e4) + '\tBaseline: ' + str(np.median( resBaseline[x]) * 1e4)) 


resTsd = np.array( [ np.std(resTransfer[x])*1e4 for x in samplesSizes])

resT   = [np.median(resTransfer[x])*1e4 for x in samplesSizes]
resBas = [np.median(resBaseline[x])*1e4 for x in samplesSizes]


f, (ax1) = plt.subplots(1,1, sharey=True, figsize = (4,4))
ax1.plot( samplesSizes, resT, label='Transfer', linewidth=2, color = sns.color_palette()[2])
ax1.fill_between( samplesSizes, resT+ 2*resTsd, resT-2*resTsd, alpha=.25, color=sns.color_palette()[2])
ax1.plot( samplesSizes, resBas, label='Baseline', linewidth=2, color = sns.color_palette()[4])
ax1.legend()
ax1.set_xlabel('Train dataset size')
ax1.set_ylabel('DSM Objective (scaled)')
ax1.set_title('Conditional DSM Objective')
f.tight_layout()

#plt.savefig('DSMtransferObjective_.pdf', dpi=300)


