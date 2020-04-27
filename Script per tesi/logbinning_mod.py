# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:45:49 2020

@author: Simone
"""
import numpy as np
def logbinning_mod(values, xmin=-1, xmax=-1, n=10):
    #Parameters
    if(xmin==-1):
        xmin=values.min()
    if(xmax==-1):
        xmax=values.max()
    assert(xmin>0 and xmax>xmin)
        
    #Grandezze derivate
    ymin=np.log(xmin)
    ymax=np.log(xmax+0.01*(xmax-xmin/n))
    delta=np.double(ymax-ymin)/n
    histo=np.zeros(n)

    for val in values:
        yi=np.log(val)
        ibin=int((yi-ymin)/delta)
     #   print(xmin, xmax)
     #   print(ymax,ymin,delta,ibin)
        histo[ibin]+=1

 #   print("Logarithmic binning")
    out = list()
    for ibin in range(len(histo)):
       #print "LOG",ibin,np.exp(ymin+ibin*delta), histo[ibin], histo[ibin]/len(values)
       out.append(np.exp(ymin+ibin*delta))
    return out