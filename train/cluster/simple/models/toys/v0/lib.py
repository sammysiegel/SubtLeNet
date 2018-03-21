from _common import *
from ..generators import toy as generator

utils.set_processor('cpu')

''' 
some global definitions
''' 

NEPOCH = 50
VERSION = 0
MODELDIR = environ.get('MODELDIR', 'models/') + '/toys/'
VARIATIONS = [0, 1, 2]
_APOSTLE = None
gen = None

# must be called!
def instantiate(name='SimpleSmear'):
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


# definitely train this on a GPU or it'll take forever
# :P
def build_linear(dims):
    N = dims[0]
    inputs  = Input(shape=dims, name='input')
    outputs = Dense(1, activation='linear', name='y_hat')(inputs)

    classifier = Model(inputs=[inputs], outputs=[outputs])
    classifier.compile(optimizer=Adam(lr=0.001),
                       loss='mse',
                       metrics=['accuracy'])

    print '########### CLASSIFIER ############'
    classifier.summary()
    print '###################################'

    return classifier

def build_discrete_adv(clf, scale, w_clf, w_adv, n_class):
    inputs = clf.inputs
    y_hat = clf.outputs[0]
    pred = Adversary(n_class, n_outputs=1, scale=scale)(y_hat)
    stack = Model(inputs=inputs,
                  outputs=[y_hat]+pred)
    stack.compile(optimizer=Adam(lr=0.001),
                  loss=['mse', 'categorical_crossentropy'],
                  loss_weights=[w_clf, w_adv])

    print '########### STACK ############'
    stack.summary()
    print '###################################'

    return stack

def build_continuous_adv(clf, scale, w_clf, w_adv):
    inputs = clf.inputs
    y_hat = clf.outputs[0]
    pred = Adversary(1, n_outputs=1, scale=scale)(y_hat)
    stack = Model(inputs=inputs,
                  outputs=[y_hat]+pred)
    stack.compile(optimizer=Adam(lr=0.001),
                  loss=['mse', 'mse'],
                  loss_weights=[w_clf, w_adv])

    print '########### STACK ############'
    stack.summary()
    print '###################################'

    return stack

def train(*args, **kwargs):
    return base_trainer(MODELDIR, _APOSTLE, NEPOCH, *args, **kwargs)
