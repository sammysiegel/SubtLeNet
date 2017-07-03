import numpy as np 
# import seaborn 
from collections import namedtuple
from keras import backend as K
from keras.engine.topology import Layer
from scipy.interpolate import interp1d

## Loss functions 

dice_smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + dice_smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + dice_smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

## Layers and ops 

## plotting tools 

# class H1:
#     '''Wrapper around numpy histogram
#     '''
#     def __init__(self,hist):
#         self.bin_edges = hist[1]
#         self.n_bins = self.bin_edges.shape[0]-1
#         self.content = hist[0]
#     def find_bin(self,x):
#         if x < self.bin_edges[0]:
#             return -1 
#         for ib in self.xrange(self.n_bins):
#             if x>= self.bin_edges[ib]:
#                 return ib 
#         return self.n_bins
#     def get_bin(self,ib):
#         if ib<0 or ib>=self.n_bins:
#             return 0 
#         return self.content[ib]
#     def integral(self,lo=None,hi=None):
#         if not lo:
#             lo = 0 
#         if not hi:
#             hi = self.n_bins 
#         widths = np.diff(self.bin_edges[lo:hi+1])
#         return np.sum(self.content[lo:hi] * widths)
# 
# 
# def plot_hists(props, hists):
#     plt.clf() 
#     bins = props['bins']
#     for h in hists:
#         plt.hist(h['vals'], bins=bins, weights=h['weights']/np.sum(h['weights']),
#                  histtype='step', # fill=False, 
#                  color=h['color'], label=h['label'])
#     if 'xlabel' in props:
#         plt.xlabel(props['xlabel'])
#     if 'ylabel' in props:
#         plt.ylabel(props['ylabel'])
#     plt.legend(loc=0)
#     plt.savefig(props['output']+'.png',bbox_inches='tight',dpi=300)
#     plt.savefig(props['output']+'.pdf',bbox_inches='tight')
# 
# 
# 
# Tagger = namedtuple('Tagger',['response','name','lo','hi','flip'])
# 
# def create_roc(taggers, labels, weights, output, nbins=50):
#     colors = ['k','r','g','b']
#     plt.clf()
#     wps = []
#     for t in taggers:
#         color = colors[0]
#         del colors[0]
#         h_sig = H1(np.histogram(t.response[labels==1],
#                                 weights=weights[labels==1],
#                                 bins=nbins,range=(t.lo,t.hi),
#                                 density=True))
#         h_bkg = H1(np.histogram(t.response[labels==0],
#                                 weights=weights[labels==0],
#                                 bins=nbins,range=(t.lo,t.hi),
#                                 density=True))
# 
#         epsilons_sig = []
#         epsilons_bkg = []
#         for ib in xrange(nbins):
#             if t.flip:
#                 esig = h_sig.integral(hi=ib)
#                 ebkg = h_bkg.integral(hi=ib)
#             else:
#                 esig = h_sig.integral(lo=ib)
#                 ebkg = h_bkg.integral(lo=ib)
#             epsilons_sig.append(esig)
#             epsilons_bkg.append(ebkg)
#         
#         interp = interp1d(epsilons_bkg,
#                           np.arange(t.lo,t.hi,float(t.hi-t.lo)/nbins))
#         wps.append(interp(0.05))
# 
#         plt.plot(epsilons_sig, epsilons_bkg, color+'-',label=t.name)
#     plt.axis([0,1,0.001,1])
#     plt.yscale('log')
#     plt.legend(loc=0)
#     plt.ylabel('Background fake rate')
#     plt.xlabel('Signal efficiency')
#     plt.savefig(output+'.png',bbox_inches='tight',dpi=300)
#     plt.savefig(output+'.pdf',bbox_inches='tight')
# 
#     return wps
