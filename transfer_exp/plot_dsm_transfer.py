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

# collect results for transfer learning
samplesSizes = [250, 500, 750, 1000, 2000, 3000, 4000, 6000] # + [x*1000 for x in range(1,4)]

resTransfer = {x: [] for x in samplesSizes}
resBaseline = {x: [] for x in samplesSizes}

# load transfer results
os.chdir('/home/projects/IMCA/ncsn-master/transferExperiments/transferRes')

for x in samplesSizes:
	files = [f for f in os.listdir(os.getcwd()) if 'TransferCDSM_Size'+str(x) in f ]
	for f in files:
		resTransfer[x].append( np.mean( pickle.load(open(f, 'rb'))) )

	files = [f for f in os.listdir(os.getcwd()) if 'Baseline_Size'+str(x) in f ]
	for f in files:
		resBaseline[x].append( np.mean( pickle.load(open(f, 'rb'))) )

	print('Transfer: ' + str(np.median( resTransfer[x]) * 1e4) + '\tBaseline: ' + str(np.median( resBaseline[x]) * 1e4)) 




resTsd = np.array( [ np.std(resTransfer[x])*1e4 for x in samplesSizes])

# actually skip and just reload
results = pickle.load(open('../TransferResults2.p', 'rb'))

resT = results['transfer']
resBas = results['Baseline']


f, (ax1) = plt.subplots(1,1, sharey=True, figsize = (4,4))
ax1.plot( list(resTransfer.keys()), resT, label='Transfer', linewidth=2, color = sns.color_palette()[2])
ax1.fill_between( list(resTransfer.keys()), resT+ 2*resTsd, resT-2*resTsd, alpha=.25, color=sns.color_palette()[2])
ax1.plot( list(resTransfer.keys()), resBas, label='Baseline', linewidth=2, color = sns.color_palette()[4])
ax1.legend()
ax1.set_xlabel('Train dataset size')
ax1.set_ylabel('')
ax1.set_title('Conditional DSM Objective')
f.tight_layout()
plt.savefig('../DSMtransferObjective_.pdf', dpi=300)


import pickle
pickle.dump( {'transfer': resT,'Baseline':resBas, 'n': samplesSizes}, open('TransferResults2.p', 'wb'))
