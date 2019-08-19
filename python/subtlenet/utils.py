import numpy as np 
from os import environ, getenv
import sys

import matplotlib as mpl
#mpl.use('cairo')
#import matplotlib.pylab as pl 
from matplotlib.colors import LogNorm 
from matplotlib import pyplot as plt
import seaborn

DOPDF = True

## turn on/off gpu

def set_processor(name):
    name = name.lower()
    if name == 'cpu':
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"         
        environ["CUDA_VISIBLE_DEVICES"] = ""                
        return
    if name == 'gpu':
        del environ['CUDA_DEVICE_ORDER']
        del environ['CUDA_VISIBLE_DEVICES']
        return
    sys.stderr.write('Unknown processor "%s"!\n'%name)

def get_processor():
    if (getenv('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
        and getenv('CUDA_VISIBLE_DEVICES') == ''):
        return 'cpu'
    else:
        return 'gpu'


## freeze layers of an NN
def freeze(model, on=False, to_skip = []):
    def _act(l,o):
        print ('un-frozen' if o else 'frozen')
        l.trainable = o
    for l in model.layers:
        print l.name ,
        _act(l, on == (l not in to_skip))


## general layout                                                                                                                      
seaborn.set(style="ticks")
seaborn.set_context("poster")
mpl.rcParams['axes.linewidth'] = 1.25
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 10
fig_size[1] = 9
plt.rcParams['figure.figsize'] = fig_size

## plotting

_epsilon = np.finfo(float).eps
def _clip(x):
    return np.sign(x + _epsilon) * np.clip(np.abs(x), _epsilon, np.inf)

default_colors = seaborn.color_palette('hls',10)
def sanitize_mask(x):
    return x==x

lognorm = LogNorm()

class NH1(object):
    __slots__ = ['bins','_content','_sumw2']
    def __init__(self, bins=[0,1]):
        assert(len(bins) > 1)
        self.bins = np.array(bins )
        self._content = np.zeros(len(self.bins) - 1, dtype=np.float64)
        self._sumw2 = np.zeros(len(self.bins) - 1, dtype=np.float64)
    def iter(self):
        for x in xrange(self.bins.shape[0]-1):
            yield x
    def find_bin(self, x):
        for ix,edge in enumerate(self.bins):
            if x <= edge:
                return max(0, ix - 1)
        return len(self.bins) - 1
    def get_content(self, ix):
        return self._content[ix]
    def get_error(self, ix):
        return np.sqrt(self._sumw2[ix])
    def set_content(self, ix, val):
        self._content[ix] = val
    def set_error(self, ix, val):
        self._sumw2[ix] = val * val;
    def clear(self):
        self._content *= 0
        self._sumw2 *= 0
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
        self._content += hist
        self._sumw2 += herr
    def add_array(self, arr):
        self._content += arr.astype(np.float64)
    def save(self, fpath):
        save_arr = np.array([
                self.bins, 
                np.concatenate([self._content, [0]])
            ])
        np.save(fpath, save_arr)
    def _load(self, fpath):
        load_arr = np.load(fpath)
        self.bins = load_arr[0]
        self._content = load_arr[1][:-1]
    @classmethod
    def load(x, fpath):
        if isinstance(x, NH1):
            x._load(fpath)
        else:
            h = NH1()
            h._load(fpath)
            return h
    def add_from_file(self, fpath):
        load_arr = np.load(fpath)
        try:
            assert(np.array_equal(load_arr[0], self.bins))
        except AssertionError as e:
            print fpath 
            print load_arr[0]
            print self.bins 
            raise e
        add_content = load_arr[1][:-1].astype(np.float64)
        self._content += add_content
    def clone(self):
        new = NH1(self.bins)
        new._content = np.array(self._content, copy=True)
        new._sumw2 = np.array(self._sumw2, copy=True)
        return new
    def add(self, rhs, scale=1):
        assert(self._content.shape == rhs._content.shape)
        self._content += scale * rhs._content
        self._sumw2 += scale * rhs._sumw2
    def multiply(self, rhs):
        assert(self._content.shape == rhs._content.shape)
        self_rel = self._sumw2 / _clip(self._content)
        rhs_rel = rhs._sumw2 / _clip(rhs._content)
        self._content *= rhs._content 
        self._sumw2 = (np.power(self_rel, 2) + np.power(rhs_rel, 2)) * self._content
    def divide(self, den, clip=False):
        inv = den.clone()
        inv.invert()
        self.multiply(inv)
        if clip:
            self._content[den._content <= _epsilon] = 1
    def integral(self, lo=None, hi=None):
        if lo is None:
            lo = 0
        if hi is None:
            hi = self._content.shape[0]
        return np.sum(self._content[lo:hi])
    def scale(self, scale=None):
        norm = float(scale if (scale is not None) else 1./self.integral())
        self._content *= norm 
        self._sumw2 *= (norm ** 2)
    def invert(self):
        for ix in range(self._content.shape[0]):
            val = self._content[ix]
            if val != 0:
                relerr = np.sqrt(self._sumw2[ix])/val 
                self._content[ix] = 1./val
                self._sumw2[ix] = relerr * self._content[ix]
            else:
                self._content[ix] = _epsilon
                self._sumw2[ix] = 0
    def quantile(self, eff, interp=False):
        den = 1. / self.integral()
        threshold = eff * self.integral()
        for ib,b1 in enumerate(self.bins):
            frac1 = self.integral(hi=ib) 
            if frac1 >= threshold:
                if not interp or ib == 0:
                    return b1

                frac2 = self.integral(hi=(ib-1)) 
                b2 = self.bins[ib-1]
                b0 = (b1 + 
                      ((threshold - frac1) * 
                       (b2 - b1) / (frac2 - frac1)))
                return b0

    def eval_array(self, arr):
        def f(x):
            return self.get_content(self.find_bin(x))
        f = np.vectorize(f)
        return f(arr)
    def plot(self, color, label, errors=False):
        bin_centers = 0.5*(self.bins[1:] + self.bins[:-1])
        if errors and np.max(np.abs(self._sumw2)) > 0:
            errs = np.sqrt(self._sumw2)
        else:
            errs = None
        plt.errorbar(bin_centers, 
                     self._content,
                     yerr = errs,
                     drawstyle = 'steps-mid',
                     color=color,
                     label=label,
                     linewidth=2)
    def mean(self):
        sumw = 0 
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        for ix in xrange(bin_centers.shape[0]):
            sumw += bin_centers[ix] * self._content[ix+1]
        return sumw / self.integral()
#    def quantile(self, threshold):
#        acc = 0 
#        threshold *= self.integral()
#        for ix in xrange(self._content.shape[0]):
#            acc += self._content[ix]
#            if acc >= threshold:
#                return 0.5 * (self.bins[ix] + self.bins[ix+1])
    def median(self):
        return self.quantile(eff = 0.5)
    def stdev(self, sheppard = False):
        # sheppard = True applies Sheppard's correction, assuming constant bin-width
        mean = self.mean()
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        integral = self.integral()
        variance = np.sum(bin_centers * bin_centers * self._content)
        variance -= integral * mean * mean
        variance /= (integral - 1)
        if sheppard:
            variance -= pow(self.bins[1] - self.bins[0], 2) / 12 
        return np.sqrt(max(0, variance))


class NH2(object):
    __slots__ = ['binsx','binsy','_content','_sumw2']
    def __init__(self, binsx, binsy):
        self.binsx = binsx 
        self.binsy = binsy 
        self._content = np.zeros([len(binsx)-1, len(binsy)-1], dtype=np.float64)
        self._sumw2 = np.zeros([len(binsx)-1, len(binsy)-1], dtype=np.float64)
    def _find_bin(self, val, axis):
        bins = self.binsx if (axis == 0) else self.binsy 
        for ix,x in enumerate(bins):
            if val <= x:
                return ix 
        return len(bins) - 1
    def find_bin_x(self, val):
        return self._find_bin(val, 0)
    def find_bin_y(self, val):
        return self._find_bin(val, 1)
    def _project(self, onto_axis, min_bin=None, max_bin=None):
        bins = self.binsx if (onto_axis == 0) else self.binsy 
        integrate_axis = 1 - onto_axis
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
        integrate_axis = 1 - onto_axis
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
        self._content += hist 
        self._sumw2 += herr 
    def integral(self):
        return np.sum(self._content)
    def scale(self, val=None):
        if val is None:
            val = self.integral()
        self._content /= val 
    def plot(self, xlabel=None, ylabel=None, output=None, cmap=None, norm=None):#cmap=pl.cm.hot, norm=None):
        plt.clf()
        ax = plt.gca()
        ax.grid(True,ls='-.',lw=0.4,zorder=-99,color='gray',alpha=0.7,which='both')
        plt.imshow(self._content.T, 
                   extent=(self.binsx[0], self.binsx[-1], self.binsy[0], self.binsy[-1]),
                   aspect=(self.binsx[-1]-self.binsx[0])/(self.binsy[-1]-self.binsy[0]),
                   cmap=cmap,
                   norm=norm,
                  )
        plt.colorbar()
        if xlabel:
            plt.xlabel(xlabel, fontsize=24)
        if ylabel:
            plt.ylabel(ylabel, fontsize=24)
        ax.set_ylim(bottom=0)
        plt.draw()
        if output:
            print 'Creating',output
            plt.savefig(output+'.png',bbox_inches='tight',dpi=100)
            if DOPDF:
                plt.savefig(output+'.pdf',bbox_inches='tight')
        else:
            plt.show()



class Plotter(object):
    def __init__(self):
        self.hists = []
        self.ymin = None
        self.ymax = None
        self.auto_yrange = False
    def add_hist(self, hist, label='', plotstyle='b'):
        if type(plotstyle) == int:
            plotstyle = default_colors[plotstyle]
        self.hists.append((hist, label, plotstyle))
    def clear(self):
        plt.clf()
        self.hists = [] 
        self.ymin = None
        self.ymax = None
    def plot(self, xlabel=None, ylabel=None, output=None, errors=True, logy=False):
        plt.clf()
        ax = plt.gca()
        for hist, label, plotstyle in self.hists:
            hist.plot(color=plotstyle, label=label, errors=errors)
        if xlabel:
            plt.xlabel(xlabel, fontsize=24)
        if ylabel:
            plt.ylabel(ylabel, fontsize=24)
        if logy:
            plt.yscale('log', nonposy='clip')
        plt.legend(loc=0, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if not self.auto_yrange:
            if self.ymax is not None:
                ax.set_ylim(top=self.ymax)
            if self.ymin is not None:
                ax.set_ylim(bottom=self.ymin)
            elif not logy:
                ax.set_ylim(bottom=0)
        plt.draw()
        if 'output':
            print 'Creating',output
            plt.savefig(output+'.png',bbox_inches='tight',dpi=100)
            if DOPDF:
                plt.savefig(output+'.pdf',bbox_inches='tight')
        else:
            plt.show()


p = Plotter()


class Roccer(object):
    def __init__(self, y_range=range(-5,1), axis=[0.2,1,0.0005,1]):
        self.cfgs = []
        self.axis = axis
        self.yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1]#[10**x for x in y_range]
        self.yticklabels = map(str, self.yticks)# [('1' if x==0 else r'$10^{%i}$'%x) for x in y_range]
        self.xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
        self.xticklabels = map(str, self.xticks)
    def add_vars(self, sig_hists, bkg_hists, labels, order=None):
        if order is None:
            order = sorted(sig_hists)
        try:
            for h in order: 
                try:
                    label = labels[h]
                    if type(label) == str:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label, None, '-'))
                    elif len(label) == 1:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label[0], None, '-'))
                    elif len(label) == 2:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label[0], label[1], '-'))
                    else:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label[0], label[1], label[2]))
                except KeyError:
                    pass # something wasn't provided - skip!
        except TypeError as e :#only one sig_hist was handed over - not iterable
            if type(labels) == str:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels, None, '-'))
            elif len(labels) == 1:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels[0], None, '-'))
            elif len(labels) == 2:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels[0], labels[1], '-'))
            else:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels[0], labels[1], labels[2]))
    def clear(self):
        self.cfgs = []
    def plot(self, output):
        fig, ax = plt.subplots(1)
        ax.get_xaxis().set_tick_params(which='both',direction='in')
        ax.get_yaxis().set_tick_params(which='both',direction='in')
        ax.grid(True,ls='-.',lw=0.4,zorder=-99,color='gray',alpha=0.7,which='major')

        min_value = 1

        colors = seaborn.color_palette('hls',10) #pl.cm.tab10(np.linspace(0,1,len(self.cfgs)))

        for i, (sig_hist, bkg_hist, label, customcolor, linestyle) in enumerate(self.cfgs):
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
            if customcolor is None:
                color = colors[i]
            elif type(customcolor) == int:
                color = default_colors[customcolor]
            else:
                color = customcolor
            plt.plot(epsilons_sig, epsilons_bkg, color=color, label=label, linewidth=2, ls=linestyle)

        plt.axis(self.axis)
        ax = plt.gca()
        #plt.set_xlim(self.axis[:2])
        #plt.set_ylim(self.axis[-2:])
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        #plt.yscale('log', nonposy='clip')
        #plt.xscale('log', nonposx='clip')
        plt.legend(loc=2, fontsize=22)
        plt.ylabel('Background fake rate', fontsize=24)
        plt.xlabel('Signal efficiency', fontsize=24)
        ax.set_yticks(self.yticks)
        ax.set_yticklabels(self.yticklabels)
        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels)

        print 'Creating',output
        plt.savefig(output+'.png',bbox_inches='tight',dpi=300)
        if DOPDF:
            plt.savefig(output+'.pdf',bbox_inches='tight')

