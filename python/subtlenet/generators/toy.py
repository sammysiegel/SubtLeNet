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


class SimpleSmear(_Generator):
    n_input = 3
    n_output = 2
    def __init__(self, batch_size, epoch_size=-1, smear_strengths=None, penalty=False):
        super(SimpleSmear, self).__init__(SimpleSmear.n_input, 
                                          SimpleSmear.n_output, 
                                          batch_size, 
                                          epoch_size)
        self.smears = smear_strengths if smear_strengths else [0]
        self.penalty = penalty
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
            if s > 0:
                res = np.random.normal(1, s, (self.batch_size,))
                i[0][:,2] *= res

            o = [np.sum(i[0], axis=-1)]
            w = [1 * ones]
            if self.penalty:
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

