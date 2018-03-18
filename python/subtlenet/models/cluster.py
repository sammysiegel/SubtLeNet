from _common import *
from ..generators import toy as generator

utils.set_processor('cpu')

''' 
some global definitions
''' 

NEPOCH = 50
VERSION = 0
MODELDIR = environ.get('MODELDIR', 'models/') + '/clustering/'
_APOSTLE = None
gen = None
encoded_size = 2

# must be called!
def instantiate(name='SimpleClustering'):
    global _APOSTLE, gen
    _APOSTLE = 'v%s'%(str(VERSION))
    system('mkdir -p %s/%s/'%(MODELDIR,_APOSTLE))
    system('cp -v %s %s/%s/trainer.py'%(sys.argv[0], MODELDIR, _APOSTLE))
    system('cp -v %s %s/%s/lib.py'%(__file__.replace('.pyc','.py'), MODELDIR, _APOSTLE))


    with open('%s/%s/setup.py'%(MODELDIR, _APOSTLE),'w') as fsetup:
        fsetup.write('''
# auto-generated. do not edit!
from subtlenet import config
from subtlenet.generators import toy as generator
gen = generator.%s
'''%name)

    gen = getattr(generator, name)

    return (gen.n_input,)


'''
first build the classifier!
'''

# set up data 
def setup_data(*args, **kwargs):
    g = {
            'train' : gen(*args, **kwargs)(),
            'test' : gen(*args, **kwargs)(),
            'validate' : gen(*args, **kwargs)()
        }
    return g


def build_model(dims, w_ae=1, w_cl=1):
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
    kmeans = KMeans(2)(encoded_rescaled)

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


def train(*args, **kwargs):
    return base_trainer(MODELDIR, _APOSTLE, NEPOCH, *args, **kwargs)
