import numpy as np 
import matplotlib.pyplot as plt 
import seaborn 
from collections import namedtuple


class H1:
    '''Wrapper around numpy histogram
    '''
    def __init__(self,hist):
        self.bin_edges = hist[1]
        self.n_bins = self.bin_edges.shape[0]-1
        self.content = hist[0]
    def find_bin(self,x):
        if x < self.bin_edges[0]:
            return -1 
        for ib in self.xrange(self.n_bins):
            if x>= self.bin_edges[ib]:
                return ib 
        return self.n_bins
    def get_bin(self,ib):
        if ib<0 or ib>=self.n_bins:
            return 0 
        return self.content[ib]
    def integral(self,lo=None,hi=None):
        if not lo:
            lo = 0 
        if not hi:
            hi = self.n_bins 
        widths = np.diff(self.bin_edges[lo:hi+1])
        return np.sum(self.content[lo:hi] * widths)


Tagger = namedtuple('Tagger',['response','name','lo','hi','flip'])

def create_roc(taggers, labels, weights, output, nbins=50):
    colors = ['k','r','g','b']
    for t in taggers:
        color = colors[0]
        del colors[0]
        h_sig = H1(np.histogram(t.response[labels==1],
                                weights=weights[labels==1],
                                bins=nbins,range=(t.lo,t.hi),
                                density=True))
        h_bkg = H1(np.histogram(t.response[labels==0],
                                weights=weights[labels==0],
                                bins=nbins,range=(t.lo,t.hi),
                                density=True))
        epsilons_sig = []
        epsilons_bkg = []
        for ib in xrange(nbins):
            if t.flip:
                esig = h_sig.integral(hi=ib)
                ebkg = h_bkg.integral(hi=ib)
            else:
                esig = h_sig.integral(lo=ib)
                ebkg = h_bkg.integral(lo=ib)
            epsilons_sig.append(esig)
            epsilons_bkg.append(ebkg)
    
        plt.plot(epsilons_sig, epsilons_bkg, color+'-',label=t.name)
    plt.axis([0,1,0.001,1])
    plt.yscale('log')
    plt.legend(loc=0)
    plt.ylabel('Background fake rate')
    plt.xlabel('Signal efficiency')
    plt.savefig(output+'.png',bbox_inches='tight',dpi=300)
    plt.savefig(output+'.pdf',bbox_inches='tight')

