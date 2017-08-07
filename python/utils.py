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
    def __init__(self, bins=[0,1]):
        assert(len(bins) > 1)
        self.bins = np.array(bins )
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
    def integral(self, lo=None, hi=None):
        if lo is None:
            lo = 0
        if hi is None:
            hi = self._content.shape[0]
        return np.sum(self._content[lo:hi])
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
    def plot(self, opts):
        plt.hist(self.bins[:-1], bins=self.bins, weights=self._content[1:-1],
                 histtype='step',
                 color=opts['color'],
                 label=opts['label'],
                 linewidth=2)
    def mean(self):
        sumw = 0 
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        for ix in xrange(bin_centers.shape[0]):
            sumw += bin_centers[ix] * self._content[ix+1]
        return sumw / self.integral()
    def quantile(self, threshold):
        acc = 0 
        threshold *= np.sum(self._content[1:-1])
        for ix in xrange(1, self._content.shape[0]-1):
            acc += self._content[ix]
            if acc >= threshold:
                return 0.5 * (self.bins[ix-1] + self.bins[ix])
    def median(self):
        return self.quantile(threshold = 0.5)
    def stdev(self, sheppard = False):
        # sheppard = True applies Sheppard's correction, assuming constant bin-width
        mean = self.mean()
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        integral = self.integral()
        variance = np.sum(bin_centers * bin_centers * self._content[1:-1])
        variance -= integral * mean * mean
        variance /= (integral - 1)
        if sheppard:
            variance -= pow(self.bins[1] - self.bins[0], 2) / 12 
        return np.sqrt(max(0, variance))



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
        self.cfgs = []
    def add_vars(self, sig_hists, bkg_hists, labels, plotstyles):
        try:
            for h in sig_hists:
                try:
                    self.cfgs.append((sig_hists[h], bkg_hists[h], labels[h], plotstyles[h]))
                except KeyError:
                    pass # something wasn't provided - skip!
        except TypeError:#only one sig_hist was handed over - not iterable
            self.cfgs.append((sig_hists,bkg_hists,labels,plotstyles))
    def clear(self):
        self.cfgs = []
    def plot(self, opts, nbins = 100):
        fig, ax = plt.subplots(1)
        ax.get_xaxis().set_tick_params(which='both',direction='in')
        ax.get_yaxis().set_tick_params(which='both',direction='in')
        ax.grid(True,ls='-.',lw=0.4,zorder=-99,color='gray',alpha=0.7,which='both')

        min_value = 1

        for sig_hist, bkg_hist, label, plotstyle in self.cfgs:
            h_sig = sig_hist
            h_bkg = bkg_hist
            rmin = h_sig.bins[0]
            rmax = h_sig.bins[len(h_sig.bins)-1]

            epsilons_sig = []
            epsilons_bkg = []

            inverted = h_sig.median() < h_bkg.median()

            total_sig = h_sig.integral()
            total_bkg = h_bkg.integral()

            for ib in xrange(nbins+1):
                if inverted:
                    esig = h_sig.integral(hi=ib) / total_sig
                    ebkg = h_bkg.integral(hi=ib) / total_bkg
                else:
                    esig = h_sig.integral(lo=ib) / total_sig
                    ebkg = h_bkg.integral(lo=ib) / total_bkg
                epsilons_sig.append(esig)
                epsilons_bkg.append(ebkg)
                if ebkg < min_value and ebkg > 0:
                    min_value = ebkg

            plt.plot(epsilons_sig, epsilons_bkg, plotstyle,label=label,linewidth=2)

        plt.axis([0,1,0.001,1])
        plt.yscale('log', nonposy='clip')
        plt.legend(loc=4, fontsize=22)
        plt.ylabel('Background fake rate', fontsize=24)
        plt.xlabel('Signal efficiency', fontsize=24)
        ax.set_yticks([0.01,0.1,1])
        ax.set_yticklabels(['0.01','0.1','1'])

        print 'Creating',opts['output']
        plt.savefig(opts['output']+'.png',bbox_inches='tight',dpi=300)
        plt.savefig(opts['output']+'.pdf',bbox_inches='tight')

