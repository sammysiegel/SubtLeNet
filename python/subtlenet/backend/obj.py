import numpy as np 
from os.path import isfile 
from os import environ
from glob import glob 
from re import sub 
from sys import stdout, stderr
from math import floor
from time import time 
from multiprocessing import Process, Pool

from ..  import config
from ..utils import NH1,NH2

_partitions = ['train', 'test', 'validate']


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
    __slots__ = ['fpath','data','loaded']
    def __init__(self, fpath=None, data=None, lazy=False):
        self.fpath = fpath 
        self.data = data
        self.loaded = not(data is None)
        if not lazy and data is None:
            self.__call__()  
    def __call__(self):
        if config.DEBUG: 
            stderr.write('Loading %s\n'%self.fpath)
            stderr.flush()
        self.data = np.nan_to_num(np.load(self.fpath))
        self.loaded = True 
    def __getitem__(self, *args):
        targs = tuple(args)
        self.data[targs]

class _DataObject(object):
    def __init__(self, fpaths):
        self.inputs = fpaths
        self.loaded = set([])
        self.n_available = 0 
        self.data = None 
        self.last_loaded = None 

    def load(self, idx=-1, memory=True, dry=False, lazy=False ):
        fpath = None
        if idx >= 0:
            fpath = self.inputs[idx]
        else:
            for fpath_ in self.inputs:
                if fpath_ not in self.loaded:
                    fpath = fpath_
                    break
        if fpath is None:
            print '_DataObject.load did not load anything!'
            return
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
    def __init__(self, label=-1):
        self.label = label
        self.objects = {part:{} for part in _partitions}
        self.weight = 'ptweight_scaled'
        self.fpath = None
        self.order = None
        self._counter = 0
        self.n_available = None

    def add_categories(self, categories, fpath):
        '''load categories
        
        Arguments:
            categories {[str]} -- list of categories to load
            fpath {[str]} -- must be of the form /some/path/to/PARTITION/files_*_CATEGORY.npy, 
                             where CATEGORY gets replaced by the category and PARTITION by 
                             the partition
        '''

        names = categories + [self.weight, 'truth']
        self.fpath = fpath 
        for part in _partitions:
            basefiles = glob(fpath.replace('CATEGORY',names[0]).replace('PARTITION',part))
            n_missing = 0
            to_add = {n:[] for n in names}
            for f in basefiles:
                missing = False 
                for n in names:
                    if not isfile(f.replace(names[0], n)):
                        missing = True 
                        if config.DEBUG:
                            print 'missing',f.replace(names[0], n)
                        n_missing += 1
                        break 
                if missing:
                    continue 
                for n in names:
                    to_add[n].append(f.replace(names[0], n))
            if n_missing:
                'partition %s had %i missing inputs'%(part, n_missing)
            for n,fs in to_add.iteritems():
                self.objects[part][n] = _DataObject(fs)
        self.order = range(len(self.objects.values()[0].values()[0].inputs))
        np.random.shuffle(self.order)

    def get(self, partition, indices=None):
        data = {}
        for k,v in self.objects[partition].iteritems():
            data[k] = v[indices]
        return data 

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

        self._counter = 0
        np.random.shuffle(self.order)

    def repartition(self, partition):
        # verbose refresh for a single partition
        print ''
        print '_DataCollection.load: repartitioning %s!'%(
                     self.fpath.replace('PARTITION',partition)
                )
        self.refresh(partitions=[partition])

    def load(self, partition, idx=-1, repartition=True, memory=True, components=None, lazy=False):
        objs = self.objects[partition]
        self.n_available = None 
        if idx < 0:
            if (self._counter == len(self.order)):
                if repartition:
                    self.repartition(partition)
                else:
                    return False
            idx = self.order[self._counter]
            if memory:
                self._counter += 1 
        for name,obj in objs.iteritems():
            dry = (components and (name not in components))
            obj.load(idx=idx, memory=memory, dry=dry, lazy=lazy)
            # assert that all the loaded data has the same size
            assert(not(dry) or (self.n_available is None) or (obj.n_available==self.n_available))
            self.n_available = obj.n_available
        return True 

    def draw(self, components, f_vars={}, f_mask=None, 
             weighted=True, partition='test', n_batches=None, f_vars2d={}):
        '''draw generic stuff
        
        Arguments:
            components {[str]} -- list of components that must be loaded (e.g. 'singletons', 'pf')
            f_vars {{str:(function, np.array)} -- dict mapping label to functions that accept the output 
                                                  of self.__getitem__() and returns array of floats of 
                                                  dim (batch_size, dim1, dim2,...). second element of 
                                                  tuple is binning
        
        Keyword Arguments:
            f_mask {[type]} -- function that accepts the output of self.__getitem__() and returns a flat 
                               array of bools of dim (batch_size,) (default: {None})
            weighted {bool} -- whether to weight the distributions or not (default: {True})
            partition {str} -- the data partition to use (default: {'test'})
            n_batches {[type]} -- number of batches to use, default is all (default: {None})
        '''
        hists = {var:NH1(x[1]) for var,x in f_vars.iteritems()}
        hists2d = {var:NH2(x[1],x[2]) for var,x in f_vars2d.iteritems()}
        i_batches = 0 
        gen = self.generator(components+[self.weight, 'truth'], partition, batch=None, lazy=False)
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
                        if weighted:
                            w = weight[mask]
                    else:
                        w = weight 
                    if len(x.shape) > 1:
                        if weighted:
                            w = np.array([w for _ in xrange(x.shape[1])]).flatten()
                        x = x.flatten() # in case it has more than one dimension
                    if weighted:
                        assert w.shape == x.shape, 'Shapes are not aligned %s %s'%(str(w.shape), str(x.shape))
                    else:
                        w = None
                    h.fill_array(x, weights=w)

                for var in hists2d: 
                    h = hists2d[var]
                    f = f_vars2d[var][0]
                    x,y = f(data)
                    if mask is not None:
                        x = x[mask]
                        y = y[mask]
                        if weighted:
                            w = weight[mask]
                    else:
                        w = weight 
                    if len(x.shape) > 1:
                        w = np.array([w for _ in x.shape[1]]).flatten()
                        y = y.flatten()
                        x = x.flatten()
                    if weighted:
                        assert (w.shape==x.shape and w.shape==y.shape), \
                               'Shapes are not aligned %s %s %s'%(str(w.shape), str(x.shape), str(y.shape))
                    else:
                        w = None
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
        gen = self.generator(components+[self.weight, 'truth'], partition, batch=None, lazy=(ncores>1))
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
                    print ''
                    break 
        else:
            l = list(enumerate(list(gen)))
            l = [(ll[0], ll[1], f) for ll in l ]
            pool = Pool(ncores)
            pool.map(_global_inference_target, l)

    def generator(self, components=None, partition='test', batch=10, repartition=False, normalize=False, lazy=False):
        # used as a generic generator for loading data
        while True:
            if not self.load(components=components, partition=partition, repartition=repartition, lazy=lazy):
                raise StopIteration
            ldata = self.get(partition)
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
                    ldata[self.weight].data /= ldata[self.weight].data.shape[0] 
                    ldata[self.weight].data *= 100
                        # normalize the weight to the size of batches
                else:
                    ldata[self.weight].data /= 100 
                if batch:
                    n_batches = int(floor(N * 1. / batch + 0.5))
                    for ib in xrange(n_batches):
                        lo = ib * batch 
                        hi = min(N, (ib + 1) * batch)
                        to_yield = {k:LazyData(data=v.data[lo:hi], lazy=True) for k,v in ldata.iteritems()}

                        sanity_check = {k:v.data.shape[0] for k,v in to_yield.iteritems()
                                                          if type(v.data) == np.ndarray}
                        scv = sanity_check.values()
                        if any([x != scv[0] for x in scv]):
                            print 'Found an inconsistency in %s'%( 
                                        self.obje.cts[partition]['singletons'].last_loaded
                                    )
                            print 'partition = ',partition
                            print 'We are in batch %i out of %i'%(ib, n_batches)
                            print 'lo=%i, hi=%i, N=%i'%(lo, hi, N)
                            for k,v in sanity_check.iteritems():
                                print '%s : %i / %i'%(k, v, ldata[k].data.shape[0])
                            raise ValueError

                        yield to_yield 
                else:
                    yield ldata 
            else:
                yield ldata 


# specialized classes for different types of data.
# unclear whether this is still necessary, but let's
# keep it in case we want to specialize further in the 
# future.
class GenCollection(_DataCollection):
    def __init__(self, label=-1):
        super(GenCollection, self).__init__(label)

    def get(self, partition, indices=None):
        '''data access
        
        Keyword Arguments:
            indices {int} -- index of data to slice, None will return entirety (default: {None})
        
        Returns:
            numpy array of data 
        '''
        data = super(GenCollection, self).get(partition,indices)
        data['weight'] = data[self.weight]
        if self.n_available is not None:
            data['label'] = LazyData(data = self.label * np.ones(data['weight'].data.shape))
        return data 

class PFSVCollection(_DataCollection):
    def __init__(self, label=-1):
        super(PFSVCollection, self).__init__(label)

    def get(self, partition, indices=None):
        '''data access
        
        Keyword Arguments:
            indices {int} -- index of data to slice, None will return entirety (default: {None})
        
        Returns:
            numpy array of data 
        '''
        data = super(PFSVCollection, self).get(partition,indices)
        data['weight'] = data[self.weight]
        if self.n_available is not None:
            data['label'] = LazyData(data = self.label * np.ones(data['weight'].data.shape))
        return data 


