from _common import *


class _Generator(object):
    def __init__(self, n_input, n_output, batch_size, epoch_size=-1):
        self.n_i = n_input
        self.n_o = n_output
        self.batch_size = batch_size
        self.epoch_size = epoch_size
    def _generate_iow(self):
        raise NotImplementedError
    def __call__(self):
        n_epoch = 0
        while n_epoch != self.epoch_size:
            yield self._generate_iow()


class RadialClustering(_Generator):
    n_input = 3
    n_output = 2
    def __init__(self, batch_size, epoch_size=-1, radii=[0.5, 1], sigma=0.01, label=False):
        super(RadialClustering, self).__init__(RadialClustering.n_input,
                                               RadialClustering.n_output,
                                               batch_size,
                                               epoch_size)
        self.sigma = sigma 
        self.radii = radii
        self.label = label
    def _generate_iow(self):
        ones = np.ones(self.batch_size * len(self.radii))
        batch_ones = np.ones(self.batch_size)
        
        i_ = []
        for r_ in self.radii:
            r = np.random.normal(r_, self.sigma, (self.batch_size))
            #phi = 2 * 3.14159 * np.random.rand(self.batch_size)
            phi = (np.pi * np.random.randint(0, 2, self.batch_size)) + r_ 
            cos_theta = 2 * np.random.rand(self.batch_size) - 1
            sin_theta = np.sqrt(1 - cos_theta**2)
            x = r * np.cos(phi) * sin_theta
            y = r * np.sin(phi) * sin_theta
            z = r * cos_theta
            i_.append(np.vstack([x, y, z]).T)
        
        i = [np.concatenate(i_, axis=0)]
        
        o = [i[0],
             np.zeros((self.batch_size * len(self.radii), len(self.radii)))]
        if self.label:
            o.append(np.concatenate([r_ * batch_ones for r_ in self.radii]))
        #o = [np_utils.to_categorical((np.sum(i[0][:,:2], axis=-1) > 1).astype(np.int),
        #                             2),
        #     0 * ones]

        w = [ones, ones]
        if self.label:
            w.append(ones)

        return i, o, w


class SimpleClustering(_Generator):
    n_input = 3
    n_output = 2
    def __init__(self, batch_size, epoch_size=-1, sigma=0.25, label=False):
        super(SimpleClustering, self).__init__(SimpleClustering.n_input,
                                               SimpleClustering.n_output,
                                               batch_size,
                                               epoch_size)
        self.label = label
        self.sigma = sigma 
    def _generate_iow(self):
        ones = np.ones(self.batch_size * 2)
        batch_ones = np.ones(self.batch_size)
        
        i_ = []
        for mu in [-0.5, 0.5]:
            i_.append(np.random.normal(mu, self.sigma, (self.batch_size * 2,)).reshape((self.batch_size,2)))
        
        i_ = np.concatenate(i_, axis=0)
        i = [i_,
             (np.sum(i_, axis=-1) 
              * np.random.normal(1, 0.05, (self.batch_size * 2))).reshape(self.batch_size*2,1)]
        i = [np.concatenate(i, axis=-1)]
        
        o = [i[0],
             0 * ones]
        if self.label:
            o.append(np.concatenate([x * batch_ones for x in [-0.5,0.5]]))
        #o = [np_utils.to_categorical((np.sum(i[0][:,:2], axis=-1) > 1).astype(np.int),
        #                             2),
        #     0 * ones]

        w = [ones, ones]

        return i, o, w


class SimpleSmear(_Generator):
    n_input = 3
    n_output = 2
    def __init__(self, batch_size, epoch_size=-1, smear_biases=None, smear_width=0.1, penalty=False):
        super(SimpleSmear, self).__init__(SimpleSmear.n_input, 
                                          SimpleSmear.n_output, 
                                          batch_size, 
                                          epoch_size)
        self.smears = smear_biases if smear_biases else [0]
        self.smear_width = smear_width
        self.penalty = penalty and (len(self.smears) > 1)
    def _generate_iow(self):
        inputs = []
        outputs = []
        weights = []
        ones = np.ones(self.batch_size)

        for s in self.smears:
            i_ = np.random.rand(self.batch_size, 2) * 10
            i = [np.concatenate([i_,
                                 np.sum(i_, axis=-1)[:,np.newaxis]],
                                axis=-1)]

            if self.smear_width > 0:
                res = np.random.normal(1, self.smear_width, (self.batch_size,))
                i[0][:,2] *= res

#            o = [np_utils.to_categorical(
#                        (np.sum(i[0], axis=-1) > 20).astype(np.int),
#                        2
#                    )]
            o = [np.sum(i[0], axis=-1)]

            bias = s + np.ones(self.batch_size)
            i[0][:,2] *= bias

            w = [1 * ones]
            if self.penalty:
                # o.append(np_utils.to_categorical(
                #             s * ones,
                #             len(self.smears)
                #         ))
                o.append(s * ones) 
                w.append(1 * ones)

            inputs.append(i); outputs.append(o); weights.append(w)

        m_inputs = []; m_outputs = []; m_weights = []

        NINPUTS = 1
        for j in xrange(NINPUTS):
            m_inputs.append( np.concatenate( [v[j] for v in inputs] ) )

        NOUTPUTS = 1 + int(self.penalty)
        for j in xrange(NOUTPUTS):
            m_outputs.append( np.concatenate( [v[j] for v in outputs] ) )
            m_weights.append( np.concatenate( [v[j] for v in weights] ) )


        return m_inputs, m_outputs, m_weights

