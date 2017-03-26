import numpy as np 
import matplotlib.pyplot as plt 
import seaborn 
from collections import namedtuple
import theano 
from keras import backend as K
from keras.engine.topology import Layer

class ReverseGradient(theano.Op):
    """ theano operation to reverse the gradients
    Introduced in http://arxiv.org/pdf/1409.7495.pdf
    """

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

class GradientReversalLayer(Layer):
    """ Reverse a gradient 
    <feedforward> return input x
    <backward> return -lambda * delta
    """
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return self.gr_op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                         "lambda": self.hp_lambda}
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class H1:
    '''Wrapper around numpy histogram
    '''
    def __init__(self,hist):
        self.bin_edges = hist[1]
        self.n_bins = self.bin_edges.shape[0]-1
        self.content = hist[0]
    def find_bin(self,x):
        if x < self.bin_edges[0]:
            return -1 
        for ib in self.xrange(self.n_bins):
            if x>= self.bin_edges[ib]:
                return ib 
        return self.n_bins
    def get_bin(self,ib):
        if ib<0 or ib>=self.n_bins:
            return 0 
        return self.content[ib]
    def integral(self,lo=None,hi=None):
        if not lo:
            lo = 0 
        if not hi:
            hi = self.n_bins 
        widths = np.diff(self.bin_edges[lo:hi+1])
        return np.sum(self.content[lo:hi] * widths)


Tagger = namedtuple('Tagger',['response','name','lo','hi','flip'])

def create_roc(taggers, labels, weights, output, nbins=50):
    colors = ['k','r','g','b']
    plt.clf()
    for t in taggers:
        color = colors[0]
        del colors[0]
        h_sig = H1(np.histogram(t.response[labels==1],
                                weights=weights[labels==1],
                                bins=nbins,range=(t.lo,t.hi),
                                density=True))
        h_bkg = H1(np.histogram(t.response[labels==0],
                                weights=weights[labels==0],
                                bins=nbins,range=(t.lo,t.hi),
                                density=True))
        epsilons_sig = []
        epsilons_bkg = []
        for ib in xrange(nbins):
            if t.flip:
                esig = h_sig.integral(hi=ib)
                ebkg = h_bkg.integral(hi=ib)
            else:
                esig = h_sig.integral(lo=ib)
                ebkg = h_bkg.integral(lo=ib)
            epsilons_sig.append(esig)
            epsilons_bkg.append(ebkg)
    
        plt.plot(epsilons_sig, epsilons_bkg, color+'-',label=t.name)
    plt.axis([0,1,0.001,1])
    plt.yscale('log')
    plt.legend(loc=0)
    plt.ylabel('Background fake rate')
    plt.xlabel('Signal efficiency')
    plt.savefig(output+'.png',bbox_inches='tight',dpi=300)
    plt.savefig(output+'.pdf',bbox_inches='tight')

