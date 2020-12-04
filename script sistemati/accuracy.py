#!/usr/bin/env python3

'''
Take nexp experiments. Each of them is repeated nreps times.

For each experiment, we can calculate the accuracy of that experiment.

It is interesting to run with -data='antinoise', since it clearly shows that if the real accuracy we can be far off in our estimate

'''

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-data', default='random', choices=['random','accurate','anti','anti-noise'], help="How to generate the artificial data.")
parser.add_argument('-nexp',  type=int, default=10, help="How many experiments.")
parser.add_argument('-nreps', type=int, default=30, help="How many times each experiment is replicated.")
args=parser.parse_args()



## Create ground truth
nexp = args.nexp # Number of experiments
ycon = np.random.rand(nexp)
ybin = np.array([0 if y<0.5 else 1 for y in ycon])

## Create artificial data
nreps = args.nreps # Number of times each experiment is repeated
if args.data=='random':
	zcon = np.random.rand( nreps,nexp) 
elif args.data=='accurate': # A perturbation around a true value
	zcon = np.array([ycon+(0.5-np.random.rand(nexp))/3. for irep in range(nreps)])
	zcon = np.abs(zcon)
	zcon /= zcon.max()
elif args.data=='anti': #Anticorrelates with ycon
	zcon = np.array([(ycon+.5)%1 for irep in range(nreps)]) 
elif args.data=='anti-noise': #Anticorrelates with ycon but with some noise 
	zcon = np.array([(ycon+.5)%1+(0.5-np.random.rand(nexp))/3. for irep in range(nreps)])
	zcon = np.abs(zcon)
	zcon /= zcon.max()

zbin = np.array([[ (0 if zcon[irep][iexp]<0.5 else 1) for irep in range(nreps)] for iexp in range(nexp)])




def delta(a,b):
	''' kronecker delta'''
	if a==b:
		return 1
	return 0

def Acc(truth, guess):
	''' Compares two lists '''
	np.mean([list(map(lambda x: delta(ybin[x[0]], x[1]), enumerate(zbin[:,irep]) )) for irep in range(nreps)], axis=0)
	return


### Accuracies

## With ground truth
accs = np.mean([list(map(lambda x: delta(ybin[x[0]], x[1]), enumerate(zbin[:,irep]) )) for irep in range(nreps)], axis=0)


## Without ground truth
# Choose first measurement as ground truth
accs0 = np.mean([list(map(lambda x: delta(zbin[:,0]   [x[0]], x[1]), enumerate(zbin[:,irep]) )) for irep in range(1,nreps)], axis=0)


# Choose j-th measurement as ground truth
reps=np.arange(nreps)
reps=np.ma.array(reps, mask=False, hard_mask=True)
j=0
reps[j] = np.ma.masked
accsj = np.mean([list(map(lambda x: delta(zbin[:,j]   [x[0]], x[1]), enumerate(zbin[:,irep]) )) for irep in reps[~reps.mask]], axis=0)
reps.mask = np.ma.nomask


# Average over every repetition of the experiment as ground truth
reps=np.arange(nreps)
reps=np.ma.array(reps, mask=False)
accstot=np.zeros(nexp)
for j in reps.data:
	reps[j] = np.ma.masked
	accstot += np.mean([list(map(lambda x: delta(zbin[:,j]   [x[0]], x[1]), enumerate(zbin[:,irep]) )) for irep in reps[~reps.mask]], axis=0)
	reps.mask = np.ma.nomask
accstot/=nreps


# Assuming that the most common value is the ground truth (only a fool would assume that)
ybin_fools = np.array([int(1.999999999999999*zbin[iexp,:].sum()/nreps) for iexp in range(nexp)])
accs_fools = np.mean([list(map(lambda x: delta(ybin_fools[x[0]], x[1]), enumerate(zbin[:,irep]) )) for irep in range(nreps)], axis=0)





print('True Accuracy:',accs.mean(), np.std(accs)/np.sqrt(nexp-1))
print('accsj:',accsj.mean(), np.std(accsj)/np.sqrt(nexp-1))
print('accs0:',accs0.mean(), np.std(accs0)/np.sqrt(nexp-1))
print('accstot:',accstot.mean(), np.std(accstot)/np.sqrt(nexp-1))
print('accs_fools:',accs_fools.mean(), np.std(accs_fools)/np.sqrt(nexp-1))
