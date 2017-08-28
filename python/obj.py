import numpy as np 
import config
from os.path import isfile 
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from glob import glob 
from keras.utils import np_utils
from re import sub 
from utils import *
from sys import stdout

_partitions = ['train', 'test', 'validate']

_singletons = ['pt','eta','mass','msd','rho','tau32','tau21','flavour',
               'nbHadrons','nProngs','nResonanceProngs','resonanceType',
               'nB','nC','partonPt','partonEta','partonPhi','partonM']
singletons = {_singletons[x]:x for x in xrange(len(_singletons))}


'''data format for training

Implements a "file system" that allows for efficient caching of large datasets in small 
pieces, selective loading of columns, and on-the-fly analysis (e.g. reweighting, visualization, 
etc). Note that this "file system" is read-only. Writing is treated separately, although 
it ought to be on my TODO to integrate these. The data itself sits on disk, but a datastore is
built in memory when the dataset is accessed (only once).
'''

class DataObject(object):
    def __init__(self, fpaths):
        self.inputs = fpaths
        self.loaded = set([])
        self.n_available = 0 
        self.data = None 

    def load(self, idx=-1, memory=True, dry=False):
        if idx > 0:
            fpath = self.inputs[idx]
            if not dry:
                if config.DEBUG: print 'Loading',fpath
                self.data = np.nan_to_num(np.load(fpath))
                self.n_available = self.data.shape[0]
            else:
                self.n_available = 0
                self.data = None
            if memory:
                self.loaded.add(fpath)
            return 
        else:
            for fpath in self.inputs:
                if fpath not in self.loaded:
                    if not dry:
                        if config.DEBUG: print 'Loading',fpath
                        self.data = np.nan_to_num(np.load(fpath))
                        self.n_available = self.data.shape[0]
                    else:
                        self.n_available = 0
                        self.data = None
                    if memory:
                        self.loaded.add(fpath)
                    return 
        print 'DataObject.load did not load anything!'

    def is_empty(self):
        return len(self.loaded) == len(self.inputs)

    def refresh(self):
        self.loaded = set([])

    def __getitem__(self, indices=None):
        if indices:
            return self.data[indices]
        else:
            return self.data 


class DataCollection(object):
    def __init__(self):
        self.objects = {part:{} for part in _partitions}
        self.cached_objects = {}
        self._current_partition = None 

    def add_categories(self, names, fpath):
        if config.DEBUG: print 'Searching for files...\r',
        for part in _partitions:
            self.objects[part][name] = DataObject(glob(fpath.replace('PARTITION', part)))
            if not len(self.objects[part][name].inputs):
                print 'ERROR: class %s, partition %s has no inputs'%(name, part)
        if config.DEBUG: print 'Found files                 '

    def load(self, partition, idx=-1, repartition=True, memory=True, components=None):
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
            obj.load(idx=idx, memory=memory, dry=dry)
            # assert that all the loaded data has the same size
            assert(not(dry) or not(n_available) or obj.n_available==n_available)
            n_available = obj.n_available
        return True 

    def __getitem__(self, indices=None):
        data = {}
        if not self._current_partition:
            print 'DataCollection[]: load must be called before requesting data'
        for k,v in self.objects[self._current_partition].iteritems():
            data[k] = v[indices]
        return data 


class PFSVCollection(DataCollection):
    def __init__(self):
        super(PFSVCollection, self).__init__()
        self.pt_weight = None 
        self.fpath = None 
        self.n_entries = 0
        self.weight = 'ptweight_scaled'

    def add_categories(self, categories, fpath):
        '''load categories
        
        Arguments:
            categories {[str]} -- list of categories to load
            fpath {[str]} -- must be of the form /some/path/to/PARTITION/files_*_CATEGORY.npy, where CATEGORY gets replaced by the category and PARTITION by the partition
        '''

        names = categories + [self.weight]
        self.fpath = fpath 
        for part in _partitions:
            basefiles = glob(fpath.replace('CATEGORY','singletons').replace('PARTITION',part))
            to_add = {n:[] for n in names}
            for f in basefiles:
                missing = False 
                for n in names:
                    if not isfile(f.replace('singletons', n)):
                        missing = True 
                        break 
                if missing:
                    continue 
                for n in names:
                    to_add[n].append(f.replace('singletons', n))
            for n,fs in to_add.iteritems():
                self.objects[part][n] = DataObject(fs)

    def __getitem__(self, indices=None):
        '''data access
        
        Keyword Arguments:
            indices {int} -- index of data to slice, None will return entirety (default: {None})
        
        Returns:
            numpy array of data 
        '''
        data = super(PFSVCollection, self).__getitem__(indices)
        data['weight'] = data[self.weight]
        data['nP'] = np_utils.to_categorical(
                data['singletons'][:,singletons[config.truth]].astype(np.int),
                config.n_truth
            )
        data['nB'] = np_utils.to_categorical(
                data['singletons'][:,singletons['nB']].astype(np.int),
                10
            )
        return data 

    def draw_singletons(self, variables, partition='test', weighted=True):
        '''DEPRECATED
        '''
        hists = {var:NH1(bins) for var,bins in variables}
        while self.load(partition=partition, repartition=False, 
                        components=['singletons',self.weight], memory=True):
            data = self.__getitem__()
            if weighted:
                weight = data[self.weight]
            else:
                weight = None
            for var,_ in variables:
                hists[var].fill_array(data['singletons'][:,singletons[var]], weight)
        return hists

    def draw(self, components, f_vars, f_mask=None, weighted=True, partition='test', n_batches=None):
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
        i_batches = 0 
        gen = self.generic_generator(components+[self.weight], partition, batch=1000)
        while True:
            try:
                data = next(gen)
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
                    try:
                        assert(w.shape == x.shape)
                    except AssertionError as e :
                        print w.shape, x.shape 
                        raise e
                    h.fill_array(x, weights=w)

            except StopIteration:
                break 
            if n_batches:
                i_batches += 1 
                completed = int(i_batches*20/n_batches)
                stdout.write('[%s%s]\r'%('#'*completed, ' '*(20-completed)))
                stdout.flush()
                if i_batches >= n_batches:
                    break
        stdout.write('\n'); stdout.flush() # flush the screen
        self.refresh(partitions=[partition])
        return hists 

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

    def generic_generator(self, components=None, partition='test', batch=10, repartition=False, normalize=False):
        # used as a generic generator for loading data
        while True:
            if not self.load(components=components, partition=partition, repartition=repartition):
                raise StopIteration
            data = self.__getitem__()
            sane = True 
            for _,v in data.iteritems():
                if np.isnan(np.sum(v)): # seems to be the fastest way
                    sane = False
            if not sane:
                print 'ERROR - last loaded data was not sane!'
                continue
            N = data[components[0]].shape[0]
            if normalize and self.weight in components:
                data[self.weight] /= batch # normalize the weight to the size of batches
            n_batches = int(N / batch + 1) 
            for ib in xrange(n_batches):
                lo = ib * batch 
                hi = min(N, (ib + 1) * batch)
                to_yield = {k:v[lo:hi] for k,v in data.iteritems()}
                yield to_yield 

    def generator(self, partition='train', batch=5, repartition=False, mask=False):
        # used as a generator for training data
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

    -   generatePF: inclusive PF candidate arrays for learning nProngs
    -   generatePFSV: as above, but with charged PFs and SVs added, for nP and nB
'''

def generateTest(collections, partition='train', batch=32, repartition=True, decorr_mass=True, normalize=False):
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generic_generator(components=['singletons', c.weight],
                                        partition=partition, 
                                        batch=small_batch, 
                                        repartition=repartition,
                                        normalize=normalize) 
                    for c in collections}
    input_indices = [singletons[x] for x in ['msd','tau32','tau21']]
    prongs_index = singletons[config.truth]
    msd_index = singletons['msd']
    def xform_mass(x):
        binned = (np.minimum(x, config.max_mass) / config.max_mass * (config.n_mass_bins - 1)).astype(np.int)
        onehot = np_utils.to_categorical(binned, config.n_mass_bins)
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
    generators = {c:c.generic_generator(components=['singletons', c.weight],
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
            data = next(generators[c])
            inputs.append([data['singletons'][:,var_idx]])
            # need to apply osme normalization to the inputs:
            mus = np.array([0.5, 75])
            sigmas = np.array([0.25, 50])
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
               repartition=True, mask=False, decorr_mass=False, normalize=False):
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generic_generator(components=['singletons', 'inclusive', c.weight],
                                        partition=partition, 
                                        batch=small_batch, 
                                        repartition=repartition,
                                        normalize=normalize) 
                    for c in collections}
    prongs_index = singletons[config.truth]
    msd_index = singletons['msd']
    norm_factor = 1. / config.max_mass 
    def xform_mass(x):
        binned = (np.minimum(x, config.max_mass) * norm_factor * (config.n_mass_bins - 1)).astype(np.int)
        onehot = np_utils.to_categorical(binned, config.n_mass_bins)
        return onehot
    while True: 
        inputs = []
        outputs = []
        weights = []
        for c in collections:
            data = next(generators[c])
            inputs.append([data['inclusive'][:,:10,:]])
            
            nprongs = np_utils.to_categorical(data['singletons'][:,prongs_index], config.n_truth)
            o = [nprongs]
            w = [data[c.weight]]

            if decorr_mass:
                mass = xform_mass(data['singletons'][:,msd_index])
                o.append(mass)
                w.append(w[0] * nprongs[:,1])

            outputs.append(o)
            weights.append(w)

        merged_inputs = []
        for j in xrange(1):
            merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))

        merged_outputs = []
        merged_weights = []
        NOUTPUTS = 2 if decorr_mass else 1 
        for j in xrange(NOUTPUTS):
            merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
            merged_weights.append(np.concatenate([v[j] for v in weights], axis=0))

        if config.weights_scale is not None:
            for j in xrange(NOUTPUTS):
                merged_weights[j] *= np.dot(merged_outputs[0], config.weights_scale)
        yield merged_inputs, merged_outputs, merged_weights


def generatePFSV(collections, partition='train', batch=32):
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generator(partition=partition, batch=small_batch) 
                    for c in collections}
    while True: 
        inputs = []
        outputs = []
        weights = []
        for c in collections:
            i, o, w = next(generators[c])
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
