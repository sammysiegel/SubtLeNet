from _common import *
from _common import _Generator, _Reshaper

_truths = ['nprongs']
truths = {_truths[x]:x for x in xrange(len(_truths))}

def make_coll(fpath, label=-1):
    coll = obj.GenCollection(label=label)
    coll.add_categories(['singletons'], fpath)
    return coll


class Generator(_Generator):
    def __init__(self, 
                 collections, partition='train', batch=256, 
                 label=False,
                 **kwargs):
        super(Generator, self).__init__(collections,
                                        partition,
                                        batch,
                                        ['singletons','ptweight_scaled','truth'],
                                        **kwargs)
        self.label = label
    def __call__(self):
        variables = config.gen_default_variables
        mus = config.gen_default_mus
        sigmas = config.gen_default_sigmas
        
        var_idx = [config.gen_singletons[x] for x in variables]
        if (mus is not None) and (sigmas is not None):
            mus = np.array(mus)
            sigmas = np.array(sigmas)
        
        while True: 
            inputs = []
            outputs = []
            weights = []
            for c in self.collections:
                data = {k:v.data for k,v in next(self.generators[c]).iteritems()}
                i = [data['singletons'][:,var_idx]]
                if (mus is not None) and (sigmas is not None):
                    # need to apply some normalization to the inputs:
                    i[0] -= mus 
                    i[0] /= sigmas 
                
                w = [data[c.weight], data[c.weight]]
                subbatch_size = w[0].shape[0]
                o = [i[0], 
                     np.zeros((subbatch_size, len(self.collections)))]
                if self.label:
                    nprongs = data['truth'][:,truths['nprongs']].astype(np.int) 
                    o.append(nprongs)
                    w.append(data[c.weight])

                inputs.append(i)
                outputs.append(o)
                weights.append(w)

            merged_inputs = []
            for j in xrange(1):
                merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))

            merged_outputs = []
            merged_weights = []
            NOUTPUTS = 2 + int(self.label)
            for j in xrange(NOUTPUTS):
                merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
                merged_weights.append(np.concatenate([v[j] for v in weights], axis=0))

            yield merged_inputs, merged_outputs, merged_weights


def generate(*args, **kwargs):
    g = Generator(*args, **kwargs)()
    while True:
        yield next(g) 
