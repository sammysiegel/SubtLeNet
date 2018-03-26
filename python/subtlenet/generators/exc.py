from _common import *
from _common import _Generator, _Reshaper

obj.GUARANTEED = 'particles'
obj.ADD_DEFAULTS = False

EXPONENT = 2
SCALE = 1

def get_dims(coll):
    return (2,)

def make_coll(fpath, categories=['particles']):
    coll = obj.GenCollection()
    coll.add_categories(categories, fpath)
    return coll


class Generator(_Generator):
    def __init__(self, 
                 collections, partition='train', 
                 autoencode=False,
                 label=False,
                 ks=[2],
                 **kwargs):
        batch = 1
        super(Generator, self).__init__(collections,
                                        partition,
                                        batch,
                                        ['particles'],
                                        **kwargs)
        self.autoencode=autoencode # probably will remain unused
        self.label = label
        self.ks = ks[:]
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
                i = [data['particles'][0,:,:2], data['particles'][0,:,4]]
                pt = data['particles'][0,:,2] / SCALE
                
                ones = np.ones((1,))
                o = []; w = []

                if self.autoencode:
                    o.append(i[0])
                    w.append(ones)

                for k in self.ks:
                    w.append(np.power(pt,EXPONENT))
                    o.append(np.zeros((pt.shape[0], k)))

                if self.label:
                    nprongs = data['truth'][:,truths['nprongs']].astype(np.int) 
                    o.append(nprongs)
                    w.append(ones)

                yield i, o, w



def generate(*args, **kwargs):
    g = Generator(*args, **kwargs)()
    while True:
        yield next(g) 
