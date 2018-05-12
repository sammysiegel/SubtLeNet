from _common import *
from ..generators.gen_auto import make_coll, generate
from ..generators import gen_auto as generator
import re

utils.set_processor('cpu')

''' 
some global definitions
''' 

NEPOCH = 50
VERSION = 0
MODELDIR = environ.get('MODELDIR', 'models/') + '/cluster_jet/'
BASEDIR = environ['BASEDIR']
_APOSTLE = None
encoded_size = 4

#config.set_gen_variables([x for x in config.gen_singletons if re.match('^[0-9]_[0-9]_[0-9]$', x)])
#config.set_gen_variables(['msd','pt','tau3','tau2','tau1','tau3sd','tau2sd','tau1sd'])
config.set_gen_variables(['msd','tau3','tau2','tau1'])

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
from subtlenet.generators import gen_auto as generator
config.gen_singletons = %s
config.gen_default_variables = %s
config.gen_default_mus = %s
config.gen_default_sigmas = %s
'''%(repr(config.gen_singletons),
     repr(config.gen_default_variables),
     repr(config.gen_default_mus),
     repr(config.gen_default_sigmas)))

    # instantiate data loaders 
    top = make_coll(BASEDIR + '/PARTITION/Top_*_CATEGORY.npy')
    qcd = make_coll(BASEDIR + '/PARTITION/QCD_*_CATEGORY.npy')

    data = [top, qcd]

    return data, (len(config.gen_default_variables),)


'''
first build the classifier!
'''

# set up data 
def setup_data(*args, **kwargs):
    g = {
            'train' : generate(*args, **kwargs),
            'test' : generate(*args, batch=100000, label=True, **kwargs),
            'validate' : generate(*args, **kwargs)
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
