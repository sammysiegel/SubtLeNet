from _common import *
from _common import _Generator, _Reshaper

_truths = ['nprongs']
truths = {_truths[x]:x for x in xrange(len(_truths))}

truncate = -1

def get_dims(coll):
    coll.objects['train']['particles'].load(memory=False)
    dims = coll.objects['train']['particles'].data.data.shape 
    if truncate > 0:
        dims = (dims[0], dims[1], truncate) 
    else:
        dims = (dims[0], dims[1], dims[2]-truncate) # need to exclude the last column
    if config.limit is not None and config.limit < dims[1]:
        dims = (dims[0], config.limit, dims[2])
    return dims 

def make_coll(fpath, categories=['singletons','particles']):
    coll = obj.GenCollection()
    coll.add_categories(categories, fpath)
    return coll


class Generator(_Generator):
    def __init__(self, 
                 collections, partition='train', batch=32, 
                 learn_mass=False, learn_pt=False,
                 smear_params=None, 
                 **kwargs):
        super(Generator, self).__init__(collections,
                                        partition,
                                        batch,
                                        ['singletons', 'particles','ptweight_scaled','truth'],
                                        **kwargs)

        self._set_xforms()
        self._set_smearer(smear_params)
        self._set_mass_reshaper()
        self.learn_mass = learn_mass
        self.learn_pt = learn_pt

    def __call__(self):
        while True: 
            inputs = []
            outputs = []
            weights = []
            for c in self.collections:
                data = {k:v.data for k,v in next(self.generators[c]).iteritems()}
                i = []

                # the last element of the particle feature vector is really truth info - do not train!
                particles = data['particles'][:,slice(config.limit),:truncate]
                if self.smearer is not None:
                    particles = self.smearer(particles)
                i.append(particles)
                msd = data['singletons'][:,self.msd_index]
                pt = data['singletons'][:,self.pt_index]
                raw_weight = data[c.weight]

                if self.learn_mass:
                    i.append(msd * self.msd_norm_factor)
                if self.learn_pt:
                    i.append((pt - config.min_pt) * self.pt_norm_factor)

                inputs.append(i)
                
                nprongs = np_utils.to_categorical(
                        np.clip(
                                data['truth'][:,truths['nprongs']].astype(np.int), 
                                0, config.n_truth
                            ),
                        config.n_truth
                    )
                is_bkg = nprongs[:,config.adversary_mask]

                o = [nprongs]
                w = [raw_weight]
                if self.window:
                    w[0] = raw_weight * np.logical_and(msd > 150, msd < 200).astype(int)
                if self.reshape:
                    w[0] = w[0] * (1 + (self.mass_reshaper(msd) - 1) * is_bkg)


                if self.decorr_mass:
                    xmass = self.xform_mass(msd)
                    o.append(xmass)
                    w.append(raw_weight * nprongs[:,config.adversary_mask])

                if self.decorr_pt:
                    xpt = xform_pt(pt)
                    o.append(xpt)
                    w.append(raw_weight * nprongs[:,config.adversary_mask])

                outputs.append(o)
                weights.append(w)

            merged_inputs = []
            NINPUTS = 1 + int(self.learn_mass) + int(self.learn_pt) 
            for j in xrange(NINPUTS):
                merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))

            merged_outputs = []
            merged_weights = []
            NOUTPUTS = 1 + int(self.decorr_pt) + int(self.decorr_mass) 
            for j in xrange(NOUTPUTS):
                merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
                merged_weights.append(np.concatenate([v[j] for v in weights], axis=0))

            if config.weights_scale is not None:
                for j in xrange(NOUTPUTS):
                    merged_weights[j] *= np.dot(merged_outputs[0], config.weights_scale)
            yield merged_inputs, merged_outputs, merged_weights


# backwards compatiblity 
def generate(*args, **kwargs):
    g = Generator(*args, **kwargs)()
    while True:
        yield next(g) 
