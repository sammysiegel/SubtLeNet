import numpy as np 
from collections import namedtuple

import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pylab as pl 
from matplotlib.colors import LogNorm 
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
        self._sumw2 = np.array([0 for x in range(len(bins)+1)], dtype=np.float64)
    def find_bin(self, x):
        for ix in xrange(len(self.bins)):
            if x < self.bins[ix]:
                return ix 
        return len(self.bins)
    def get_content(self, ix):
        return self._content[ix]
    def get_error(self, ix):
        return np.sqrt(self._sumw2[ix])
    def set_content(self, ix, val):
        self._content[ix] = val
    def fill(self, x, y=1):
        ix = self.find_bin(x)
        self._content[ix] += y
        self._sumw2[ix] = pow(y, 2)
    def fill_array(self, x, weights=None):
        mask = sanitize_mask(x)
        mask &= sanitize_mask(weights)
        x_masked = x[mask]
        weights_masked = None if (weights is None) else weights[mask]
        w2 = None if (weights_masked is None) else np.square(weights_masked)
        hist = np.histogram(x_masked, bins=self.bins, weights=weights_masked, density=False)[0]
        herr = np.histogram(x_masked, bins=self.bins, weights=w2, density=False)[0]
        self._content += np.concatenate([[0],hist,[0]])
        self._sumw2 += np.concatenate([[0],herr,[0]])
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
        self._sumw2 *= (norm ** 2)
    def invert(self):
        for ix in range(self._content.shape[0]):
            val = self._content[ix]
            if val:
                relerr = np.sqrt(self._sumw2[ix])/val 
                self._content[ix] = 1000./val
                self._sumw2[ix] = relerr * self._content[ix]
    def eval_array(self, arr):
        def f(x):
            return self.get_content(self.find_bin(x))
        f = np.vectorize(f)
        return f(arr)
    def plot(self, color, label, errors=False):
        if errors: 
            bin_centers = 0.5*(self.bins[1:] + self.bins[:-1])
            errs = np.sqrt(self._sumw2)
            plt.errorbar(bin_centers, 
                         self._content[1:-1],
                         yerr = errs[1:-1],
                         marker = '.',
                         drawstyle = 'steps-mid',
                         color=color,
                         label=label,
                         linewidth=2)
        else:
            plt.hist(self.bins[:-1], bins=self.bins, weights=self._content[1:-1],
                     histtype='step',
                     color=color,
                     label=label,
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


class NH2(object):
    def __init__(self, binsx, binsy):
        self.binsx = binsx 
        self.binsy = binsy 
        self._content = np.zeros([len(binsx)+1, len(binsy)+1], dtype=np.float64)
        self._sumw2 = np.zeros([len(binsx)+1, len(binsy)+1], dtype=np.float64)
    def _find_bin(self, val, axis):
        bins = self.binsx if (axis == 0) else self.binsy 
        for ix,x in enumerate(bins):
            if val < x:
                return ix 
        return len(bins)
    def find_bin_x(self, val):
        return self._find_bin(val, 0)
    def find_bin_y(self, val):
        return self._find_bin(val, 1)
    def _project(self, onto_axis, min_bin=None, max_bin=None):
        bins = self.binsx if (onto_axis == 0) else self.binsy 
        integrate_axis = int(not(onto_axis))
        h1 = NH1(bins)
        if integrate_axis == 0:
            s = self._content[min_bin:max_bin,:]
            e = self._sumw2[min_bin:max_bin,:]
        else:
            s = self._content[:,min_bin:max_bin]
            e = self._sumw2[:,min_bin:max_bin]
        proj = np.sum(s, axis=integrate_axis)
        proj_e = np.sum(e, axis=integrate_axis)
        h1._content = proj
        h1._sumw2 = proj_e
        return h1
    def _project_by_val(self, onto_axis, min_bin=None, min_cut=None, max_bin=None, max_cut=None):
        integrate_axis = int(not(onto_axis))
        if min_cut:
            min_bin = self._find_bin(min_cut, integrate_axis)
        if max_cut:
            max_bin = self._find_bin(max_cut, integrate_axis)
        return self._project(onto_axis, min_bin, max_bin)
    def project_onto_x(self, *args, **kwargs):
        return self._project_by_val(0, *args, **kwargs)
    def project_onto_y(self, *args, **kwargs):
        return self._project_by_val(1, *args, **kwargs)
    def fill(self, x, y, z=1):
        self._content[self.find_bin_x(x), self.find_bin_y(y)] += z 
        self._sumw2[self.find_bin_x(x), self.find_bin_y(y)] += pow(z, 2)
    def fill_array(self, x, y, weights=None):
        mask = sanitize_mask(x)
        mask &= sanitize_mask(y)
        mask &= sanitize_mask(weights)
        x_masked = x[mask]
        y_masked = y[mask]
        weights_masked = None if (weights is None) else weights[mask]
        hist = np.histogram2d(x_masked, y_masked, 
                              bins=(self.binsx, self.binsy), 
                              weights=weights_masked, 
                              normed=False)[0]
        w2 = None if (weights_masked is None) else np.square(weights_masked)
        herr = np.histogram2d(x_masked, y_masked, 
                              bins=(self.binsx, self.binsy), 
                              weights=w2, 
                              normed=False)[0]
        # print hist 
        # over/underflow bins are zeroed out
        self._content += np.lib.pad(hist, (1,1), 'constant', constant_values=0) 
        self._sumw2 += np.lib.pad(herr, (1,1), 'constant', constant_values=0) 
    def integral(self):
        return np.sum(self._content)
    def scale(self, val=None):
        if val is None:
            val = self.integral()
        self._content /= val 
    def plot(self, xlabel=None, ylabel=None, output=None, cmap=pl.cm.hot, norm=LogNorm()):
        plt.clf()
        ax = plt.gca()
        ax.grid(True,ls='-.',lw=0.4,zorder=-99,color='gray',alpha=0.7,which='both')
        plt.imshow(self._content[1:-1,1:-1].T, 
                   extent=(self.binsx[0], self.binsx[-1], self.binsy[0], self.binsy[-1]),
                   aspect=(self.binsx[-1]-self.binsx[0])/(self.binsy[-1]-self.binsy[0]),
                  # cmap=cmap,
                   )
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if output:
            print 'Creating',output
            plt.savefig(output+'.png',bbox_inches='tight',dpi=100)
            plt.savefig(output+'.pdf',bbox_inches='tight')
        else:
            plt.show()



class Plotter(object):
    def __init__(self):
        self.hists = []
    def add_hist(self, hist, label, plotstyle):
        self.hists.append((hist, label, plotstyle))
    def clear(self):
        plt.clf()
        self.hists = [] 
    def plot(self, xlabel=None, ylabel=None, output=None, errors=True):
        plt.clf()
        for hist, label, plotstyle in self.hists:
            hist.plot(color=plotstyle, label=label, errors=errors)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.legend(loc=0)
        if 'output':
            print 'Creating',output
            plt.savefig(output+'.png',bbox_inches='tight',dpi=300)
            plt.savefig(output+'.pdf',bbox_inches='tight')
        else:
            plt.show()


p = Plotter()


class Roccer(object):
    def __init__(self):
        self.cfgs = []
    def add_vars(self, sig_hists, bkg_hists, labels, plotstyles=None):
        try:
            for h in sorted(sig_hists):
                try:
                    if plotstyles is None:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], labels[h], None))
                    else:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], labels[h], plotstyles[h]))
                except KeyError:
                    pass # something wasn't provided - skip!
        except TypeError:#only one sig_hist was handed over - not iterable
            self.cfgs.append((sig_hists,bkg_hists,labels,plotstyles))
    def clear(self):
        self.cfgs = []
    def plot(self, output):
        fig, ax = plt.subplots(1)
        ax.get_xaxis().set_tick_params(which='both',direction='in')
        ax.get_yaxis().set_tick_params(which='both',direction='in')
        ax.grid(True,ls='-.',lw=0.4,zorder=-99,color='gray',alpha=0.7,which='both')

        min_value = 1

        colors = pl.cm.tab10(np.linspace(0,1,len(self.cfgs)))

        for i, (sig_hist, bkg_hist, label, plotstyle) in enumerate(self.cfgs):
            h_sig = sig_hist
            h_bkg = bkg_hist
            rmin = h_sig.bins[0]
            rmax = h_sig.bins[len(h_sig.bins)-1]

            epsilons_sig = []
            epsilons_bkg = []

            inverted = h_sig.median() < h_bkg.median()

            total_sig = h_sig.integral()
            total_bkg = h_bkg.integral()

            nbins = h_sig.bins.shape[0]
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
            color = colors[i]
            if plotstyle is not None:
                color += plotstyle 
            plt.plot(epsilons_sig, epsilons_bkg, color=color, label=label,linewidth=2)

        plt.axis([0,1,0.0001,1])
        plt.yscale('log', nonposy='clip')
        plt.legend(loc=4, fontsize=22)
        plt.ylabel('Background fake rate', fontsize=24)
        plt.xlabel('Signal efficiency', fontsize=24)
        ax.set_yticks([0.001, 0.01,0.1,1])
        ax.set_yticklabels(['0.001','0.01','0.1','1'])

        print 'Creating',output
        plt.savefig(output+'.png',bbox_inches='tight',dpi=300)
        plt.savefig(output+'.pdf',bbox_inches='tight')

