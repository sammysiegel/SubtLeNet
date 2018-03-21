from _common import *
from ..generators.exc import make_coll, generate, get_dims
from ..generators import exc as generator
import time

utils.set_processor('cpu')

''' 
some global definitions
''' 

NEPOCH = 50
VERSION = 0
MODELDIR = environ.get('MODELDIR', 'models/') + '/exc/'
BASEDIR = environ['BASEDIR']
_APOSTLE = None
encoded_size = 4

# must be called!
def instantiate():
    global _APOSTLE
    _APOSTLE = 'v%s'%(str(VERSION))
    system('mkdir -p %s/%s/'%(MODELDIR,_APOSTLE))
    system('cp -v %s %s/%s/trainer.py'%(sys.argv[0], MODELDIR, _APOSTLE))
    system('cp -v %s %s/%s/lib.py'%(__file__.replace('.pyc','.py'), MODELDIR, _APOSTLE))


    with open('%s/%s/setup.py'%(MODELDIR, _APOSTLE),'w') as fsetup:
        fsetup.write('''
# auto-generated. do not edit!
from subtlenet import config
from subtlenet.generators import exc as generator
''')

    # instantiate data loaders 
    vbf = make_coll(BASEDIR + '/PARTITION/VBFHbb_*_CATEGORY.npy')
    zh = make_coll(BASEDIR + '/PARTITION/*ZvvHbb_*_CATEGORY.npy')

#    data = [vbf, zh]
    data = [vbf]
    dims = get_dims(data[0])

    return data, dims


# set up data 
def setup_data(*args, **kwargs):
    g = {
            'train' : generate(*args, partition='train',  **kwargs),
            'test' : generate(*args, partition='train',  label=True, **kwargs),
        }
    return g


def build_simple(dims, ks):
    inputs  = Input(shape=dims, name='input')
    opts = {
            'linear_unclustered' : True,
            'R' : 1,
            'etaphi' : True,
           }
    kmeans = [KMeans(k, name='kmeans%i'%k, **opts)(inputs) for k in ks]

    clusterer = Model(inputs=[inputs], outputs=kmeans)
    #clusterer.compile(optimizer=SGD(lr=20),
    clusterer.compile(optimizer=Adam(lr=1),
                      loss=min_pred)

    print '########### CLUSTERER ############'
    clusterer.summary()
    print '###################################'

    return clusterer


def build_model(dims, k, w_ae=1, w_cl=1):
    inputs  = Input(shape=dims, name='input')
    h = Dense(10, activation='tanh')(inputs)
    h = Dense(10, activation='tanh')(h)
    h = Dense(10, activation='tanh')(h)
    encoded = Dense(encoded_size, activation='linear', name='encoded')(h)
    h = Dense(10, activation='tanh')(encoded)
    h = Dense(10, activation='tanh')(h)
    h = Dense(10, activation='tanh')(h)
    decoded = Dense(dims[0], activation='linear', name='decoded')(h)

    encoded_rescaled = encoded
    # encoded_rescaled = BatchNormalization(momentum=0.6)(encoded)
    kmeans = KMeans(k)(encoded_rescaled)

    clusterer = Model(inputs=[inputs], outputs=[decoded, kmeans])
    clusterer.compile(optimizer=Adam(lr=0.0001),
                      loss=['mse', min_pred],
                      loss_weights=[w_ae, w_cl])  # use this one instead

    encoder = Model(inputs=[inputs], outputs=[encoded_rescaled])
    encoder.compile(optimizer=Adam(lr=0.0001), loss='mse') # can't actually train this one


    print '########### CLUSTERER ############'
    clusterer.summary()
    print '###################################'

    return clusterer, encoder

def cluster(model, data=None, gen=None, post_func=None, skip_train = False):
    if gen and data is None:
        data = next(gen)
    x, y, w = data
    if not skip_train:
        start = time.time()
        model.fit(x, y, sample_weight=w, batch_size=y[0].shape[0], epochs=NEPOCH, verbose=0)
        print 'fit took %.6fs for %i epochs'%(time.time() - start, NEPOCH)

    ret = {
            'weights' : {l.name:l.get_weights() for l in model.layers},
            'distances' : model.predict(x),
            'x' : x[0],
            'pt' : np.power(w[0], 1./generator.EXPONENT) * generator.SCALE
          }

    return ret
        
