import numpy as np 
from collections import namedtuple
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.use('cairo')
from matplotlib import pyplot as plt
import seaborn

## general layout                                                                                                                      
seaborn.set(style="ticks")
seaborn.set_context("poster")
mpl.rcParams['axes.linewidth'] = 1.25
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 10
fig_size[1] = 9
plt.rcParams['figure.figsize'] = fig_size

## plotting

def sanitize_mask(x):
    return x==x

class NH1(object):
    def __init__(self, bins=[]):
        self.bins = np.array(bins, dtype=np.float64)
        self._content = np.array([0 for x in range(len(bins)+1)], dtype=np.float64)
    def find_bin(self, x):
        for ix in xrange(len(self.bins)):
            if x < self.bins[ix]:
                return ix 
        return len(self.bins)
    def get_content(self, ix):
        return self._content[ix]
    def set_content(self, ix, val):
        self._content[ix] = val
    def fill(self, x, y=1):
        self._content[self.find_bin(x)] += y
    def fill_array(self, x, weights=None):
        mask = sanitize_mask(x)
        mask &= sanitize_mask(weights)
        x_masked = x[mask]
        weights_masked = None if (weights is None) else weights[mask]
        hist = np.histogram(x_masked, bins=self.bins, weights=weights_masked, density=False)[0]
        self._content += np.concatenate([[0],hist,[0]])
    def add_array(self, arr):
        self._content += arr.astype(np.float64)
    def save(self, fpath):
        save_arr = np.array([
                np.concatenate([[0],self.bins,[0]]), 
                self._content
            ])
        np.save(fpath, save_arr)
    def load(self, fpath):
        load_arr = np.load(fpath)
        self.bins = load_arr[0][1:-1]
        self._content = load_arr[1]
    def add_from_file(self, fpath):
        load_arr = np.load(fpath)
        try:
            assert(np.array_equal(load_arr[0][1:-1], self.bins))
        except AssertionError as e:
            print fpath 
            print load_arr[0]
            print self.bins 
            raise e
        add_content = load_arr[1].astype(np.float64)
        self._content += add_content
    def integral(self):
        return np.sum(self._content)
    def scale(self, scale=None):
        norm = float(scale if scale else 1./self.integral())
        self._content *= norm 
    def invert(self):
        for ix in range(self._content.shape[0]):
            val = self._content[ix]
            if val:
                self._content[ix] = 1000./val
    def eval_array(self, arr):
        def f(x):
            return self.get_content(self.find_bin(x))
        f = np.vectorize(f)
        return f(arr)
        # ret = np.empty(arr.shape)
        # for ix in xrange(arr.shape[0]):
        #     ret[ix] = self.get_content(self.find_bin(arr[ix]))
        # return ret 
    def plot(self, opts):
        plt.hist(self.bins[:-1], bins=self.bins, weights=self._content[1:-1],
                 histtype='step',
                 color=opts['color'],
                 label=opts['label'],
                 linewidth=2)


class Plotter(object):
    def __init__(self):
        self.hists = []
    def add_hist(self, hist, label, plotstyle):
        self.hists.append((hist, label, plotstyle))
    def clear(self):
        plt.clf()
        self.hists = [] 
    def plot(self, opts):
        plt.clf()
        for hist, label, plotstyle in self.hists:
            hist.plot({'color':plotstyle, 'label':label})
        if 'xlabel' in opts:
            plt.xlabel(opts['xlabel'])
        if 'ylabel' in opts:
            plt.ylabel(opts['ylabel'])
        plt.legend(loc=0)
        if 'output' in opts:
            print 'Creating',opts['output']
            plt.savefig(opts['output']+'.png',bbox_inches='tight',dpi=300)
            plt.savefig(opts['output']+'.pdf',bbox_inches='tight')


p = Plotter()


class Roccer(object):
    def __init__(self):
        self.ROCs = []
    def addROCs(self, sig_hists, bkg_hists, labels, plotstyles):
        try:
            for h in sig_hists:
                self.ROCs.append((sig_hists[h], bkg_hists[h], labels[h], plotstyles[h]))
        except:#only one sig_hist was handed over - not iterable
            self.ROCs.append((sig_hists,bkg_hists,labels,plotstyles))
    def plotROCs(self, opts, nbins = 100):
        fig, ax = plt.subplots(1)
        ax.get_xaxis().set_tick_params(which='both',direction='in')
        ax.get_yaxis().set_tick_params(which='both',direction='in')
        ax.grid(True,ls='-.',lw=0.4,zorder=-99,color='gray',alpha=0.7,which='both')

        min_value = 1

        for sig_hist, bkg_hist, label, plotstyle in self.ROCs:
            h_sig = sig_hist
            h_bkg = bkg_hist
            rmin = h_sig.bins[0]
            rmax = h_sig.bins[len(h_sig.bins)-1]

            epsilons_sig = []
            epsilons_bkg = []

            for ib in xrange(nbins):
                if True:
                    esig = h_sig.integral(hi=ib)
                    ebkg = h_bkg.integral(hi=ib)
                else:
                    esig = h_sig.integral(lo=ib)
                    ebkg = h_bkg.integral(lo=ib)
                epsilons_sig.append(esig)
                epsilons_bkg.append(ebkg)
                if ebkg < min_value and ebkg > 0:
                    min_value = ebkg

            #print epsilons_sig
            plt.plot(epsilons_sig[1:], epsilons_bkg[1:], plotstyle,label=label,linewidth=2)

        plt.axis([0,1,0.005,1])
        plt.yscale('log', nonposy='clip')
        plt.legend(loc=4, fontsize=22)
        plt.ylabel('Background fake rate', fontsize=24)
        plt.xlabel('Signal efficiency', fontsize=24)
        ax.set_yticks([0.01,0.1,1])
        ax.set_yticklabels(['0.01','0.1','1'])

        plt.savefig(opts['output']+'.png',bbox_inches='tight',dpi=300)
        plt.savefig(opts['output']+'.pdf',bbox_inches='tight')


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
# def plot_hists(opts, hists):
#     plt.clf() 
#     bins = opts['bins']
#     for h in hists:
#         plt.hist(h['vals'], bins=bins, weights=h['weights']/np.sum(h['weights']),
#                  histtype='step', # fill=False, 
#                  color=h['color'], label=h['label'])
#     if 'xlabel' in opts:
#         plt.xlabel(opts['xlabel'])
#     if 'ylabel' in opts:
#         plt.ylabel(opts['ylabel'])
#     plt.legend(loc=0)
#     plt.savefig(opts['output']+'.png',bbox_inches='tight',dpi=300)
#     plt.savefig(opts['output']+'.pdf',bbox_inches='tight')
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
