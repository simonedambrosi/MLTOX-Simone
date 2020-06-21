# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:45:49 2020

@author: Simone
"""
import numpy as np
def logbinning_mod(values, xmin=-1, xmax=-1, n=10, align='center'):
	'''
	Logarithmic binning of values
	'''
	#Parameters
	if(xmin==-1):
	    xmin=values.min()
	if(xmax==-1):
	    xmax=values.max()
	assert(xmin>0 and xmax>xmin)

	#Grandezze derivate
	ymin=np.log(xmin)
	ymax=np.log(xmax+0.001*(xmax-xmin/n))
	delta=np.double(ymax-ymin)/n
	histo_unfair=np.zeros(n)
	nval=len(values)

	for val in values:
	    yi=np.log(val)
	    ibin=int((yi-ymin)/delta)
	    histo_unfair[ibin]+=1

    # Left centered bins
	bins =np.array([np.exp(ymin+ibin*delta) for ibin in range(n+1)]) # Bins centered on the left side

	# Since bins are of variable size, the correct histogram divides by the bin width
	histo=np.array( [histo_unfair[i]/(bins[i+1]-bins[i]) for i in range(n) ])
	density=histo/nval # Normalized histogram

	if align=='center':
		bins = np.array([np.sqrt(bins[ibin]*bins[ibin+1]) for ibin in range(n)]) # Bins centered in the geometric center
	    
	out = np.array([(ibin, bins[ibin], histo[ibin], density[ibin]) for ibin in range(n)])

	return out