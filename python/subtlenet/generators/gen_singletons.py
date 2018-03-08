from _common import *

_truths = ['nprongs']
truths = {_truths[x]:x for x in xrange(len(_truths))}

def make_coll(fpath, label=-1):
    coll = obj.GenCollection(label=label)
    coll.add_categories(['singletons'], fpath)
    return coll


def generate(collections, 
             partition='train', 
             batch=32, 
             repartition=True,
             decorr_mass=False,
             decorr_pt=False,
             decorr_label=None,
             kl_decorr_mass=False,
             kl_decorr_pt=False,
             kl_decorr_label=None,
             normalize=False):
    variables = config.gen_default_variables
    mus = config.gen_default_mus
    sigmas = config.gen_default_sigmas
    small_batch = max(1, int(batch / len(collections)))
    generators = {c:c.generator(components=['singletons', c.weight,'truth'],
                                partition=partition, 
                                batch=batch, 
                                #batch=small_batch, 
                                repartition=repartition,
                                normalize=normalize) 
                    for c in collections}
    variation_mask = {c:(c.label >= 0) for c in collections}
    train_mask = {c:(c.label <= 0) for c in collections}
    var_idx = [config.gen_singletons[x] for x in variables]
    if (mus is not None) and (sigmas is not None):
        mus = np.array(mus)
        sigmas = np.array(sigmas)
    msd_index = config.gen_singletons['msd']
    pt_index = config.gen_singletons['pt']
    msd_norm_factor = 1. / config.max_mass 
    pt_norm_factor = 1. / (config.max_pt - config.min_pt)
    either_decorr_label = max(decorr_label, kl_decorr_label)
    def xform_mass(x):
        binned = (np.minimum(x, config.max_mass) * msd_norm_factor * (config.n_decorr_bins - 1)).astype(np.int)
        onehot = np_utils.to_categorical(binned, config.n_decorr_bins)
        return onehot
    def xform_pt(x):
        binned = (np.minimum(x-config.min_pt, config.max_pt-config.min_pt) 
                  * pt_norm_factor 
                  * (config.n_decorr_bins - 1)
                 ).astype(np.int)
        onehot = np_utils.to_categorical(binned, config.n_decorr_bins)
        return onehot
    while True: 
        inputs = []
        outputs = []
        weights = []
        for c in collections:
            data = {k:v.data for k,v in next(generators[c]).iteritems()}
            i = [data['singletons'][:,var_idx]]
            if (mus is not None) and (sigmas is not None):
                # need to apply some normalization to the inputs:
                i[0] -= mus 
                i[0] /= sigmas 
            
            nprongs = np_utils.to_categorical(
                    np.clip(
                        data['truth'][:,truths['nprongs']].astype(np.int), 
                        0, config.n_truth
                        ),
                    config.n_truth
                )
            o = [nprongs]
            w = [data[c.weight]]
            subbatch_size = w[0].shape[0]

            # if this is just being used for the adversary, turn it off for the main loss f'n
            if either_decorr_label > 0 and not train_mask[c]:
                w[0] = np.zeros((subbatch_size,))


            if decorr_mass or kl_decorr_mass:
                mass = xform_mass(data['singletons'][:,msd_index])
                if decorr_mass:
                    o.append(mass)
                    w.append(w[0] * nprongs[:,config.adversary_mask])
                if kl_decorr_mass:
                    # inputs get [one-hot mass] + [sample weight]
                    i.append(np.concatenate([mass, np.reshape(w, (subbatch_size,1))], axis=-1))
                    # outputs get [one-hot mass] + [sample weight] + [QCD mask]
                    o.append(
                        np.concatenate(
                            [mass, 
                             np.reshape(w, (subbatch_size,1)), 
                             np.reshape(nprongs[:,config.adversary_mask], (subbatch_size,1))],
                            axis=-1
                        )
                    )
                    w.append(w[0] * nprongs[:,config.adversary_mask])

            if decorr_pt or kl_decorr_pt:
                pt = xform_pt(data['singletons'][:,pt_index])
                if decorr_pt:
                    o.append(pt)
                    w.append(w[0] * nprongs[:,config.adversary_mask])

            
            if either_decorr_label > 0:
                if either_decorr_label > 1:
                    labels = np_utils.to_categorical(
                                    np.clip(data['label'].astype(np.int), 0, either_decorr_label - 1),
                                    either_decorr_label
                                )
                else:
                    labels = data['label']
                label_weight = data[c.weight] if variation_mask[c] else np.zeros((subbatch_size,))
                if decorr_label > 0:
                    o.append(labels)
                    w.append(label_weight)
                else:
                    i.append(labels)
                    o.append(np.concatenate([np.reshape(label_weight, (subbatch_size,1)),
                                             labels],
                                            axis=-1))
                    w.append(label_weight)

            inputs.append(i)
            outputs.append(o)
            weights.append(w)

        merged_inputs = []
        for j in xrange(1 + int(kl_decorr_mass) + int(kl_decorr_label is not None)):
            merged_inputs.append(np.concatenate([v[j] for v in inputs], axis=0))

        merged_outputs = []
        merged_weights = []
        NOUTPUTS = 1 + sum(map(int, [decorr_pt, decorr_mass, kl_decorr_mass, 
                                     (either_decorr_label is not None)]))
        for j in xrange(NOUTPUTS):
            merged_outputs.append(np.concatenate([v[j] for v in outputs], axis=0))
            merged_weights.append(np.concatenate([v[j] for v in weights], axis=0))

        yield merged_inputs, merged_outputs, merged_weights

