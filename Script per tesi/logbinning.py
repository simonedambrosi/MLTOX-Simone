import numpy as np




nvalues=500
expList=np.random.exponential(1,nvalues)

def linbinning(values, xmin=-1, xmax=-1, n=10):
    #Parameters
    if(xmin==-1):
        xmin=values.min()
    if(xmax==-1):
        xmax=values.max()
    assert(xmin>0 and xmax>xmin)
        
    delta=np.double(xmax-xmin)/(n-1)
    histo=np.zeros(n)

    for val in values:
        ibin=int((val-xmin)/delta)
        histo[ibin]+=1

    print("Linear binning")
    for ibin in range(len(histo)):
        print "LIN",ibin,xmin+ibin*delta, histo[ibin], histo[ibin]/len(values)

    return


def logbinning(values, xmin=-1, xmax=-1, n=10):
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
        print xmin, xmax
        print ymax,ymin,delta,ibin
        histo[ibin]+=1

    print("Logarithmic binning")
    for ibin in range(len(histo)):
        print "LOG",ibin,np.exp(ymin+ibin*delta), histo[ibin], histo[ibin]/len(values)
        
    return

linbinning(expList)
logbinning(expList)
