import numpy as np 
from os.path import isfile 
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from glob import glob 
from keras.utils import np_utils
from re import sub 
from utils import *

DEBUG = False 

_singletons = ['pt','eta','mass','msd','rho','tau32','tau21','flavour',
               'nbHadrons','nProngs','nResonanceProngs','resonanceType',
               'nB','nC']
singletons = {_singletons[x]:x for x in xrange(len(_singletons))}

class DataObject(object):
    def __init__(self, fpaths):
        self.inputs = fpaths
        self.loaded = []
        self.n_available = 0 
        self.data = None 
    def load(self, idx=-1, memory=True):
        if idx > 0:
            fpath = self.inputs[idx]
            if DEBUG: print 'Loading',fpath
            self.data = np.load(fpath)
            self.n_available = self.data.shape[0]
            if memory:
                self.loaded.append(fpath)
            return 
        else:
            for fpath in self.inputs:
                if fpath not in self.loaded:
                    if DEBUG: print 'Loading',fpath
                    self.data = np.load(fpath)
                    self.n_available = self.data.shape[0]
                    if memory:
                        self.loaded.append(fpath)
                    return 
    def __getitem__(self, indices=None):
        if indices:
            return self.data[indices]
        else:
            return self.data 


class DataCollection(object):
    def __init__(self):
        self.objects = {}
        self.input_partitions = None 
        self.cached_input_partitions = None 
    def add_classes(self, names, fpath):
        if DEBUG: print 'Searching for files...\r',
        self.objects[name] = DataObject(glob(fpath))
        if not len(self.objects[name].inputs):
            print 'ERROR: class %s has no inputs'%name
        if DEBUG: print 'Found files                 '
    def partition(self, train_frac=0.5, test_frac=0.25):
        n_inputs = None 
        self.input_partitions = {'train':[], 'test':[], 'validate':[]} 
        for _,v in self.objects.iteritems():
            if n_inputs:
                assert(len(v.inputs) == n_inputs)
            else:
                n_inputs = len(v.inputs)
        n_train = train_frac * n_inputs
        n_test = (train_frac + test_frac) * n_inputs

        # figure out the subsamples in our collection
        pds = {}
        sample_inputs = self.objects.values()[0].inputs  #just a random collection
        for i,base_ in zip(xrange(len(sample_inputs)), sample_inputs):
            base = sub('Output.*', '', base_)
            if base not in pds:
                pds[base] = []
            pds[base].append(i)

        for idxs in pds.values():
            N = len(idxs)
            n_train = int(train_frac * N)
            n_test = int((train_frac + test_frac) * N)
            self.input_partitions['train'].extend(idxs[:n_train])
            self.input_partitions['test'].extend(idxs[n_train:n_test])
            self.input_partitions['validate'].extend(idxs[n_test:])

        # shuffler = range(n_inputs)
        # np.random.shuffle(shuffler)
        # for idx in xrange(n_inputs):
        #     sidx = shuffler[idx]
        #     if idx > n_test:
        #         self.input_partitions['validate'].append(sidx)
        #     elif idx > n_train:
        #         self.input_partitions['test'].append(sidx)
        #     else:
        #         self.input_partitions['train'].append(sidx)
        for k in self.input_partitions:
            np.random.shuffle(self.input_partitions[k])
        if DEBUG: 
            print 'Training sample:'
            print self.input_partitions['train']
        self.cached_input_partitions = {}
        for k,v in self.input_partitions.iteritems():
            self.cached_input_partitions[k] = v[:]
    def load(self, idx=-1, partition=None, repartition=True, memory=True, components=None):
        if partition:
            if self.input_partitions == None:
                self.partition() 
            if not len(self.input_partitions[partition]):
                if not repartition:
                    return False
                self.input_partitions[partition] = self.cached_input_partitions[partition][:]
            idx = self.input_partitions[partition].pop(0)
        for k,v in self.objects.iteritems():
            if components and k != components:
                continue
            v.load(idx, memory)
        return True 
    def n_available(self):
        ns = [v.n_available for _,v in self.objects.iteritems()]
        assert(max([abs(x - ns[0]) for x in ns]) == 0)
        return ns[0]
    def __getitem__(self, indices=None):
        data = {}
        for k,v in self.objects.iteritems():
            data[k] = v[indices]
        return data 

class PFSVCollection(DataCollection):
    def __init__(self):
        super(PFSVCollection, self).__init__()
        self.pt_weight = None 
        self.fpath = None 
        self.n_entries = 0
    def add_classes(self, names, fpath):
        '''
        fpath must be of the form /some/path/to/files_*_XXXX.npy, 
        where XXXX gets replaced by the names
        '''
        self.fpath = fpath 
        basefiles = glob(fpath.replace('XXXX','singletons'))
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
            self.objects[n] = DataObject(fs)
    def weight(self, normalize=True):
        if DEBUG: print 'Calculating weight...\r',
        self.pt_weight = NH1(
            np.arange(200.,2000.,40)
            # [0,40,80,120,160,200,250,300,350,400,450,500,600,700,800,1000,1200,1400,2000]
            )
        for o in self.objects['singletons'].inputs:
            self.pt_weight.add_from_file(sub('singletons', 'ptweight', o))
        self.n_entries = int(self.pt_weight.integral())
        self.pt_weight.save('sum_weights.npy')
        self.pt_weight.invert()
        if not normalize:
            self.pt_weight.scale(self.n_entries/1000.)
        self.pt_weight.save('inverted_weights.npy')
        if DEBUG: print 'Calculated weight         '
        print 'Loaded a total of %i samples from %s' % (self.n_entries, self.fpath)
    def __getitem__(self, indices=None):
        data = super(PFSVCollection, self).__getitem__(indices)
        data['weight'] = self.pt_weight.eval_array(data['singletons'][:,singletons['pt']])
        data['nP'] = np_utils.to_categorical(
                data['singletons'][:,singletons['resonanceType']].astype(np.int),
                5
            )
        data['nB'] = np_utils.to_categorical(
                data['singletons'][:,singletons['nB']].astype(np.int),
                10
            )
        # data['nC'] = np_utils.to_categorical(
        #         data['singletons'][:,singletons['nC']].astype(np.int),
        #         5
        #     )
        return data 
    def draw_singletons(self, vars, partition='test', weighted=True):
        if not self.pt_weight:
            self.weight()
        hists = {var:NH1(bins) for var,bins in vars}
        while self.load(partition=partition, repartition=False, components='singletons', memory=False):
            data = self.__getitem__()
            if weighted:
                weight = self.pt_weight.eval_array(data['singletons'][:,singletons['pt']])
            else:
                weight = None
            for var,_ in vars:
                hists[var].fill_array(data['singletons'][:,singletons[var]], weight)
        return hists
    def generator(self, partition='train', batch=5, repartition=False):
        # used as a generator for training data
        if not self.pt_weight:
            self.weight()
        while True:
            if not self.load(partition=partition, repartition=repartition):
                raise StopIteration
            data = self.__getitem__()
            input_keys = self.objects.keys()
            input_keys.remove('singletons')
            inputs = [data[x] for x in input_keys]
            #inputs = [data[x] for x in ['charged', 'inclusive', 'sv']]
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

def generatePF(collections, partition='train', batch=32):
    for c in collections:
        if not c.pt_weight:
            c.weight(normalize=False)
    # entry_frac = 1./sum([x.n_entries for x in collections])
    # batches = {c : entry_frac * c.n_entries * batch for c in collections}
    # generators = {c:c.generator(partition=partition, batch=max(1, int(batches[c]))) 
    #                 for c in collections}
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generator(partition=partition, batch=small_batch, repartition=True) 
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
        for j in xrange(1):
            merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))
        merged_outputs = []
        for j in xrange(2):
            merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
        merged_weights = np.concatenate(weights, axis=0)
        # print merged_inputs[0][:10]
        # print merged_outputs[0][:10]
        # print merged_weights[:10]
        yield merged_inputs[0], merged_outputs[0], merged_weights
        # yield merged_inputs, merged_outputs, [merged_weights, merged_weights]

def generatePFSV(collections, partition='train', batch=32):
    for c in collections:
        if not c.pt_weight:
            c.weight()
    entry_frac = 1./sum([x.n_entries for x in collections])
    batches = {c : entry_frac * c.n_entries * batch for c in collections}
    generators = {c:c.generator(partition=partition, batch=max(1, int(batches[c]))) 
                    for c in collections}
    # small_batch = max(1, int(batch / len(collections)))
    # generators = {c:c.generator(partition=partition, batch=small_batch) 
    #                 for c in collections}
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
