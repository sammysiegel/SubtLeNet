import numpy as np 
import config
from os.path import isfile 
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from glob import glob 
from keras.utils import np_utils
from re import sub 
from utils import *
from sys import stdout, stderr
from math import floor
from time import time 
from multiprocessing import Process, Pool


_partitions = ['train', 'test', 'validate']

_singletons = ['pt','eta','mass','msd','rho','tau32','tau21','flavour',
               'nbHadrons','nProngs','nResonanceProngs','resonanceType',
               'nB','nC','partonPt','partonEta','partonPhi','partonM']
singletons = {_singletons[x]:x for x in xrange(len(_singletons))}

_gen_singletons = ['pt', 'eta', 'mass', 't21', 't32', 
                   'msd', 't21sd', 't32sd',
                   'parton_mass', 'parton_t21', 'parton_t32']
gen_singletons = {_gen_singletons[x]:x for x in xrange(len(_gen_singletons))}

limit = None



'''data format for training

Implements a "file system" that allows for efficient caching of large datasets in small 
pieces, selective loading of columns, and on-the-fly analysis (e.g. reweighting, visualization, 
etc). Note that this "file system" is read-only. Writing is treated separately, although 
it ought to be on my TODO to integrate these. The data itself sits on disk, but a datastore is
built in memory when the dataset is accessed (only once).

Author: S. Narayanan 
'''

def _global_inference_target(args):
    counter = args[0]
    ldata = args[1]
    f = args[2]
    stdout.write('%i    \r'%(counter)); stdout.flush() 
    for _,v in ldata.iteritems():
        v()
    data = {k:v.data for k,v in ldata.iteritems()} 
    inference = f(data)
    out_name = ldata['singletons'].fpath.replace('singletons', name)
    np.save(out_name, inference)

class LazyData(object):
    def __init__(self, fpath=None, data=None, lazy=False):
        self.fpath = fpath 
        self.data = data
        self.loaded = not(data is None)
        if not lazy and data is None:
            self.__call__()  
    def __call__(self):
        if config.DEBUG: stderr.write('Loading %s\n'%self.fpath)
        stderr.flush()
        self.data = np.nan_to_num(np.load(self.fpath))
        self.loaded = True 
    def __getitem__(self, *args):
        # not sure how to implement slicing, etc through this interface
        targs = tuple(args)
        self.data[targs]

class DataObject(object):
    def __init__(self, fpaths):
        self.inputs = fpaths
        self.loaded = set([])
        self.n_available = 0 
        self.data = None 
        self.last_loaded = None 

    def load(self, idx=-1, memory=True, dry=False, lazy=False ):
        if idx > 0:
            fpath = self.inputs[idx]
            if not dry:
                self.data = LazyData(fpath=fpath, lazy=lazy)
                if not lazy:
                    self.n_available = self.data.data.shape[0]
            else:
                self.n_available = 0
                self.data = None
            if memory:
                self.loaded.add(fpath)
            self.last_loaded = fpath 
            return 
        else:
            for fpath in self.inputs:
                if fpath not in self.loaded:
                    if not dry:
                        self.data = LazyData(fpath=fpath, lazy=lazy)
                        if not lazy:
                            self.n_available = self.data.data.shape[0]
                    else:
                        self.n_available = 0
                        self.data = None
                    if memory:
                        self.loaded.add(fpath)
                    self.last_loaded = fpath 
                    return 
        print 'DataObject.load did not load anything!'

    def is_empty(self):
        return len(self.loaded) == len(self.inputs)

    def refresh(self):
        self.loaded = set([])

    def __getitem__(self, indices=None):
        if indices and self.data.loaded:
            _data = self.data.data[indices]
            return LazyData(fpath=fpath, data=_data, lazy=True)
        else:
            return self.data 


class _DataCollection(object):
    def __init__(self):
        self.objects = {part:{} for part in _partitions}
        self._current_partition = None 
        self.weight = 'ptweight_scaled'
        self.fpath = None

    def add_categories(self, categories, fpath):
        '''load categories
        
        Arguments:
            categories {[str]} -- list of categories to load
            fpath {[str]} -- must be of the form /some/path/to/PARTITION/files_*_CATEGORY.npy, where CATEGORY gets replaced by the category and PARTITION by the partition
        '''

        names = categories + [self.weight]
        self.fpath = fpath 
        for part in _partitions:
            basefiles = glob(fpath.replace('CATEGORY',names[0]).replace('PARTITION',part))
            to_add = {n:[] for n in names}
            for f in basefiles:
                missing = False 
                for n in names:
                    if not isfile(f.replace(names[0], n)):
                        missing = True 
                        break 
                if missing:
                    continue 
                for n in names:
                    to_add[n].append(f.replace(names[0], n))
            for n,fs in to_add.iteritems():
                self.objects[part][n] = DataObject(fs)

    def load(self, partition, idx=-1, repartition=True, memory=True, components=None, lazy=False):
        self._current_partition = partition 
        objs = self.objects[partition]
        n_available = None 
        for name,obj in objs.iteritems():
            dry = (components and (name not in components))
            if obj.is_empty():
                if repartition:
                    obj.refresh()
                else:
                    return False 
            obj.load(idx=idx, memory=memory, dry=dry, lazy=lazy)
            # print obj.last_loaded, obj.n_available, (n_available is None)
            # assert that all the loaded data has the same size
            assert(not(dry) or (n_available is None) or (obj.n_available==n_available))
            n_available = obj.n_available
        return True 

    def __getitem__(self, indices=None):
        data = {}
        if not self._current_partition:
            print '_DataCollection.__getitem__: load must be called before requesting data'
        for k,v in self.objects[self._current_partition].iteritems():
            data[k] = v[indices]
        return data 

    def draw(self, components, f_vars={}, f_mask=None, 
             weighted=True, partition='test', n_batches=None, f_vars2d={}):
        '''draw generic stuff
        
        Arguments:
            components {[str]} -- list of components that must be loaded (e.g. 'singletons', 'inclusive')
            f_vars {{str:(function, np.array)} -- dict mapping label to functions that accept the output of self.__getitem__() and returns array of floats of dim (batch_size, dim1, dim2,...). second element of tuple is binning
        
        Keyword Arguments:
            f_mask {[type]} -- function that accepts the output of self.__getitem__() and returns a flat array of bools of dim (batch_size,) (default: {None})
            weighted {bool} -- whether to weight the distributions or not (default: {True})
            partition {str} -- the data partition to use (default: {'test'})
            n_batches {[type]} -- number of batches to use, default is all (default: {None})
        '''
        hists = {var:NH1(x[1]) for var,x in f_vars.iteritems()}
        hists2d = {var:NH2(x[1],x[2]) for var,x in f_vars2d.iteritems()}
        i_batches = 0 
        gen = self.generator(components+[self.weight], partition, batch=None, lazy=False)
        while True:
            try:
                data = {k:v.data for k,v in next(gen).iteritems()}
                mask = f_mask(data) if f_mask else None 
                weight = data[self.weight] if weighted else None

                for var in hists:
                    h = hists[var]
                    f = f_vars[var][0]
                    x = f(data)
                    if mask is not None: 
                        x = x[mask]
                        w = weight[mask]
                    else:
                        w = weight 
                    if len(x.shape) > 1:
                        w = np.array([w for _ in x.shape[1]]).flatten()
                        x = x.flatten() # in case it has more than one dimension
                    assert w.shape == x.shape, 'Shapes are not aligned %s %s'%(str(w.shape), str(x.shape))
                    h.fill_array(x, weights=w)

                for var in hists2d: 
                    h = hists2d[var]
                    f = f_vars2d[var][0]
                    x,y = f(data)
                    if mask is not None:
                        x = x[mask]
                        y = y[mask]
                        w = weight[mask]
                    else:
                        w = weight 
                    if len(x.shape) > 1:
                        w = np.array([w for _ in x.shape[1]]).flatten()
                        y = y.flatten()
                        x = x.flatten()
                    assert (w.shape==x.shape and w.shape==y.shape), \
                           'Shapes are not aligned %s %s %s'%(str(w.shape), str(x.shape), str(y.shape))
                    h.fill_array(x, y, weights=w)

            except StopIteration:
                break 
            if n_batches:
                i_batches += 1 
                completed = int(i_batches*20/n_batches)
                stdout.write('[%s%s] %s\r'%('#'*completed, ' '*(20-completed), self.fpath))
                stdout.flush()
                if i_batches >= n_batches:
                    break
        if n_batches:
            stdout.write('\n'); stdout.flush() # flush the screen
        self.refresh(partitions=[partition])
        if len(hists) and len(hists2d):
            return hists, hists2d 
        elif len(hists2d):
            return hists2d 
        else:
            return hists

    def infer(self, components, f, name, partition='test', ncores=1):
        gen = self.generator(components+[self.weight], partition, batch=None, lazy=(ncores>1))
        starttime = time()
        if ncores == 1:
            counter = 0 
            while True:
                try:
                    stdout.write('%i (%i s)\r'%(counter, time()-starttime)); stdout.flush(); counter += 1 
                    data = {k:v.data for k,v in next(gen).iteritems()}
                    inference = f(data)
                    out_name = self.objects[partition]['singletons'].last_loaded.replace('singletons', name)
                    np.save(out_name, inference)

                except StopIteration:
                    break 
        else:
            l = list(enumerate(list(gen)))
            l = [(ll[0], ll[1], f) for ll in l ]
            pool = Pool(ncores)
            pool.map(_global_inference_target, l)

    def refresh(self, partitions=None):
        '''refresh
        
        reset the objects to the beginning of the data stream
        
        Keyword Arguments:
            partitions {[type]} -- [description] (default: {None})
        '''
        partitions = _partitions if partitions is None else partitions
        for p in partitions:
            for o in self.objects[p].values():
                o.refresh()

    def generator(self, components=None, partition='test', batch=10, repartition=False, normalize=False, lazy=False):
        # used as a generic generator for loading data
        while True:
            if not self.load(components=components, partition=partition, repartition=repartition, lazy=lazy):
                raise StopIteration
            ldata = self.__getitem__()
            if ldata.values()[0].loaded:
                sane = True 
                for _,v in ldata.iteritems():
                    if np.isnan(np.sum(v.data)): # seems to be the fastest way
                        sane = False
                if not sane:
                    print 'ERROR - last loaded data was not sane!'
                    continue
                N = ldata[components[0]].data.shape[0]
                if normalize and self.weight in components and batch:
                    ldata[self.weight].data /= batch # normalize the weight to the size of batches
                else:
                    ldata[self.weight].data /= 100 
                if batch:
                    n_batches = int(floor(N * 1. / batch + 0.5))
                    for ib in xrange(n_batches):
                        lo = ib * batch 
                        hi = min(N, (ib + 1) * batch)
                        to_yield = {k:LazyData(data=v.data[lo:hi], lazy=True) for k,v in ldata.iteritems()}
                        yield to_yield 
                else:
                    yield ldata 
            else:
                yield ldata 


# why is this even a separate class? abstracted everything away
class GenCollection(_DataCollection):
    def __init__(self):
        super(GenCollection, self).__init__()



class PFSVCollection(_DataCollection):
    def __init__(self):
        super(PFSVCollection, self).__init__()

    def __getitem__(self, indices=None):
        '''data access
        
        Keyword Arguments:
            indices {int} -- index of data to slice, None will return entirety (default: {None})
        
        Returns:
            numpy array of data 
        '''
        data = super(PFSVCollection, self).__getitem__(indices)
        data['weight'] = data[self.weight]
        if data['singletons'].loaded:
            data['nP'] = np_utils.to_categorical(
                    data['singletons'].data[:,singletons[config.truth]].astype(np.int),
                    config.n_truth
                )
            data['nB'] = np_utils.to_categorical(
                    data['singletons'].data[:,singletons['nB']].astype(np.int),
                    10
                )
        return data 

    def old_generator(self, partition='train', batch=5, repartition=False, mask=False):
        # used as a generator for training data
        # almost totally deprecated
        while True:
            if not self.load(partition=partition, repartition=repartition):
                raise StopIteration
            data = self.__getitem__()
            input_keys = self.objects[partition].keys()
            input_keys.remove('singletons')
            input_keys.remove(self.weight)
            #inputs = [data[x] for x in ['charged', 'inclusive', 'sv']]
            if mask:
                x_mask = np.logical_and(
                            np.logical_and(
                                data['singletons'][:,singletons['msd']] > 110,
                                data['singletons'][:,singletons['msd']] < 210
                                ),
                            data['singletons'][:,singletons['pt']] > 400
                            )
                inputs = [data[x][x_mask] for x in input_keys]
                outputs = [data[x][x_mask] for x in ['nP', 'nB']]
                weights = data['weight'][x_mask]
            else:
                inputs = [data[x] for x in input_keys]
                outputs = [data[x] for x in ['nP', 'nB']]
                weights = data['weight']
            N = weights.shape[0]
            n_batches = int(N / batch + 1) 
            for ib in xrange(n_batches):
                lo = ib * batch 
                hi = min(N, (ib + 1) * batch)
                i = [x[lo:hi] for x in inputs]
                o = [x[lo:hi] for x in outputs]
                w = weights[lo:hi]
                yield i, o, w 


'''
Custom generators for different kinds of data
    -   generateTest: produces some singletons for adversarial training
    -   generateSingletons: to train on msd, tau32, tau21
    -   generatePF: inclusive PF candidate arrays for learning nProngs
    -   generatePFSV: as above, but with charged PFs and SVs added, for nP and nB
'''

def generateTest(collections, partition='train', batch=32, repartition=True, decorr_mass=True, normalize=False):
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generator(components=['singletons', c.weight],
                                        partition=partition, 
                                        batch=small_batch, 
                                        repartition=repartition,
                                        normalize=normalize) 
                    for c in collections}
    input_indices = [singletons[x] for x in ['msd','tau32','tau21']]
    prongs_index = singletons[config.truth]
    msd_index = singletons['msd']
    def xform_mass(x):
        binned = (np.minimum(x, config.max_mass) / config.max_mass * (config.n_decorr_bins - 1)).astype(np.int)
        onehot = np_utils.to_categorical(binned, config.n_decorr_bins)
        return onehot
    while True: 
        inputs = []
        outputs = []
        weights = []
        for c in collections:
            data = next(generators[c])
            inputs.append([data['singletons'][:,input_indices]])

            nprongs = np_utils.to_categorical(data['singletons'][:,prongs_index], config.n_truth)
            mass = xform_mass(data['singletons'][:,msd_index])
            outputs.append([nprongs, mass])

            # print nprongs.shape, mass.shape, data[c.weight].shape

            weights.append([data[c.weight], 
                            data[c.weight] * nprongs[:,1]]) # only unmask 1-prong QCD events
        merged_inputs = []
        for j in xrange(1):
            merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))

        merged_outputs = []; merged_weights = []
        NOUTPUTS = 2 if decorr_mass else 1 
        for j in xrange(NOUTPUTS):
            merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
            merged_weights.append(np.concatenate([v[j] for v in weights], axis=0))
        yield merged_inputs, merged_outputs, merged_weights



def generateSingletons(collections, variables, partition='train', batch=32, 
                       repartition=True):
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generator(components=['singletons', c.weight],
                                        partition=partition, 
                                        batch=small_batch, 
                                        repartition=repartition,
                                        normalize=False) 
                    for c in collections}
    prongs_index = singletons[config.truth]
    var_idx = [singletons[x] for x in variables]
    while True: 
        inputs = []
        outputs = []
        weights = []
        for c in collections:
            data = {k:v.data for k,v in next(generators[c]).iteritems()}
            inputs.append([data['singletons'][:,var_idx]])
            # need to apply osme normalization to the inputs:
            mus = np.array([0.5, 0.5, 75])
            sigmas = np.array([0.5, 0.5, 50])
            inputs[-1][0] -= mus 
            inputs[-1][0] /= sigmas 
            
            nprongs = np_utils.to_categorical(data['singletons'][:,prongs_index], config.n_truth)
            o = [nprongs]
            w = [data[c.weight]]

            outputs.append(o)
            weights.append(w)

        merged_inputs = []
        for j in xrange(1):
            merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))

        merged_outputs = []
        merged_weights = []
        NOUTPUTS = 1
        for j in xrange(NOUTPUTS):
            merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
            merged_weights.append(np.concatenate([v[j] for v in weights], axis=0))

        yield merged_inputs, merged_outputs, merged_weights


def generatePF(collections, partition='train', batch=32, 
               repartition=True, mask=False, decorr_mass=False, decorr_pt=False,
               normalize=False):
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generator(components=['singletons', 'inclusive', c.weight],
                                        partition=partition, 
                                        batch=small_batch, 
                                        repartition=repartition,
                                        normalize=normalize) 
                    for c in collections}
    prongs_index = singletons[config.truth]
    msd_index = singletons['msd']
    pt_index = singletons['pt']
    mass_norm_factor = 1. / config.max_mass 
    pt_norm_factor = 1. / (config.max_pt - config.min_pt)
    def xform_mass(x):
        binned = (np.minimum(x, config.max_mass) * mass_norm_factor * (config.n_decorr_bins - 1)).astype(np.int)
        onehot = np_utils.to_categorical(binned, config.n_decorr_bins)
        return onehot
    def xform_pt(x):
        binned = (np.minimum(x-config.min_pt, config.max_pt-config.min_pt) * pt_norm_factor * (config.n_decorr_bins - 1)).astype(np.int)
        onehot = np_utils.to_categorical(binned, config.n_decorr_bins)
        return onehot
    while True: 
        inputs = []
        outputs = []
        weights = []
        for c in collections:
            data = {k:v.data for k,v in next(generators[c]).iteritems()}
            if limit:
                inputs.append([data['inclusive'][:,:limit,:]])
            else:
                inputs.append([data['inclusive']])
            
            nprongs = np_utils.to_categorical(data['singletons'][:,prongs_index], config.n_truth)
            o = [nprongs]
            w = [data[c.weight]]

            if decorr_mass:
                mass = xform_mass(data['singletons'][:,msd_index])
                o.append(mass)
                w.append(w[0] * nprongs[:,config.adversary_mask])

            if decorr_pt:
                pt = xform_pt(data['singletons'][:,pt_index])
                o.append(pt)
                w.append(w[0] * nprongs[:,config.adversary_mask])

            outputs.append(o)
            weights.append(w)

        merged_inputs = []
        for j in xrange(1):
            merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))

        merged_outputs = []
        merged_weights = []
        NOUTPUTS = 1 + int(decorr_pt) + int(decorr_mass)
        for j in xrange(NOUTPUTS):
            merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
            merged_weights.append(np.concatenate([v[j] for v in weights], axis=0))

        if config.weights_scale is not None:
            for j in xrange(NOUTPUTS):
                merged_weights[j] *= np.dot(merged_outputs[0], config.weights_scale)
        yield merged_inputs, merged_outputs, merged_weights


def generatePFSV(collections, partition='train', batch=32):
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.old_generator(partition=partition, batch=small_batch) 
                    for c in collections}
    while True: 
        inputs = []
        outputs = []
        weights = []
        for c in collections:
            i, o, w = next(generators[c]).data
            inputs.append(i)
            outputs.append(o)
            weights.append(w)
        merged_inputs = []
        for j in xrange(3):
            merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))
        merged_outputs = []
        for j in xrange(2):
            merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
        merged_weights = np.concatenate(weights, axis=0)
        yield merged_inputs, merged_outputs[:1], [merged_weights]
        # yield merged_inputs, merged_outputs, [merged_weights, merged_weights]
