#!/usr/bin/env python2.7

from _common import *
from ..generators.gen import make_coll, Generator, get_dims
from ..generators import gen as generator
from copy import copy, deepcopy

''' 
some global definitions
''' 

NEPOCH = 50
VERSION = '6'
MODELDIR = environ.get('MODELDIR', 'models/') + '/particles/'
BASEDIR = environ['BASEDIR']
OPTIMIZER = 'Adam'
_APOSTLE = None
train_opts = {
        'learn_mass' : True,
        'learn_pt' : True,
        }


_conv_args = { 
    'activation' : 'relu',
    'padding' : 'same',
    'kernel_initializer' : 'lecun_uniform' 
    }
_dense_args = {
        'kernel_initializer' : 'lecun_uniform'
        }

# must be called!
def instantiate(trunc=4, limit=50):
    global _APOSTLE
    generator.truncate = trunc
    config.limit = limit
    _APOSTLE = 'v%s_trunc%i_limit%i'%(str(VERSION), generator.truncate, config.limit)
    system('mkdir -p %s/%s/'%(MODELDIR,_APOSTLE))
    system('cp -v %s %s/%s/trainer_%s.py'%(sys.argv[0], MODELDIR, _APOSTLE, VERSION))
    system('cp -v %s %s/%s/lib_%s.py'%(__file__.replace('.pyc','.py'), MODELDIR, _APOSTLE, VERSION))

    # instantiate data loaders 
    top = make_coll(BASEDIR + '/PARTITION/Top_*_CATEGORY.npy')
    qcd = make_coll(BASEDIR + '/PARTITION/QCD_*_CATEGORY.npy')

    data = [top, qcd]
    dims = get_dims(top)

    with open('%s/%s/setup.py'%(MODELDIR, _APOSTLE),'w') as fsetup:
        fsetup.write('''
from subtlenet import config
from subtlenet.generators import gen as generator
from subtlenet.utils import set_processor
config.limit = %i
generator.truncate = %i
set_processor("%s")
config.smear_params = %s
'''%(config.limit, generator.truncate, utils.get_processor(), repr(config.smear_params)))

    return data, dims


'''
first build the classifier!
'''

# set up data 
def setup_data(data, **kwargs):
    kwargs.update(train_opts)
    if 'smear_params' not in kwargs:
        kwargs['smear_params'] = config.smear_params
    gen = {
        'train' : Generator(data, partition='train', batch=1000, **kwargs)(),
        'validation' : Generator(data, partition='validate', batch=2000, **kwargs)(),
        'test' : Generator(data, partition='test', batch=10, **kwargs)(),
        }
    return gen


def compilation_args(name, **kwargs):
    if name == 'classifier':
        return {
            'optimizer' : getattr(keras_objects, OPTIMIZER)(lr=0.0005, decay=1.0E-05), # decay is new
            'loss' : 'categorical_crossentropy',
            'metrics' : ['accuracy']
        }
    if name == 'adversary':
        N = range(kwargs['N'])
        if type(kwargs['w_adv']) != list:
            kwargs['w_adv'] = [kwargs['w_adv'] for _ in N]
        return {
            'optimizer' : getattr(keras_objects, OPTIMIZER)(lr=0.00025, decay=1.0E-05), # decay is new
            'loss' : ['categorical_crossentropy'] + [kwargs['loss'] for _ in N],
            'loss_weights' : [kwargs['w_clf']] + kwargs['w_adv']
        }

# this is purely a discriminatory classifier
def build_classifier(dims, last_size=10, l2_penalty=0, l1_penalty=0):
    if l2_penalty + l1_penalty > 0:
        reg = lambda : L1L2(l1_penalty, l2_penalty)
    else:
        reg = lambda : None
    # reg_opts = lambda : {'bias_regularizer' : reg(), 'kernel_regularizer' : reg()}
    reg_opts = lambda : {}

    LSTMImplementation = CuDNNLSTM if (utils.get_processor() == 'gpu') else LSTM

    def conv_args():
        a = copy(_conv_args)
        a.update(reg_opts())
        return a

    def lstm_args():
        a = {'recurrent_regularizer' : reg()}
        a.update(reg_opts())
        return a

    def dense_args(act='relu'):
        a = copy(_dense_args)
        a['activation'] = act
        a.update(reg_opts())
        return a

    input_particles  = Input(shape=(dims[1], dims[2]), name='input_particles')
    input_mass = Input(shape=(1,), name='input_mass')
    input_pt = Input(shape=(1,), name='input_pt')
    inputs = [input_particles, input_mass, input_pt]

    # now build the particle network
    h = BatchNormalization(momentum=0.6, name='f_bn0')(input_particles)
    h = Conv1D(32, 2, name='f_c0', **conv_args())(h)
    h = BatchNormalization(momentum=0.6, name='f_bn1')(h)
    h = Conv1D(16, 4, name='f_c1', **conv_args())(h)
    h = BatchNormalization(momentum=0.6, name='f_bn2')(h)

    h = LSTMImplementation(100, name='f_lstm', **lstm_args())(h)

    h = BatchNormalization(momentum=0.6, name='f_bn3')(h)
    h = Dense(100, name='f_d0', **dense_args())(h)
    h = BatchNormalization(momentum=0.6, name='f_bn4')(h)
    h = Dense(50, name='f_d1', **dense_args())(h)
    h = BatchNormalization(momentum=0.6, name='f_bn5')(h)
    h = Dense(50, name='f_d2', **dense_args())(h)
    h = BatchNormalization(momentum=0.6, name='f_bn6')(h)
    h = Dense(last_size, name='f_d3', **dense_args())(h)
    h = BatchNormalization(momentum=0.6, name='f_bn7')(h)
    particles_final = h 

    # merge everything
    to_merge = [particles_final, input_mass, input_pt]
    h = concatenate(to_merge, name='f_cc0')

    for i in xrange(2):
        h = Dense(50, name='u_xd%i'%i, **dense_args('tanh'))(h)
        h = BatchNormalization(momentum=0.6, name='u_xbn%i'%i)(h)


    y_hat = Dense(config.n_truth, name='y_hat', **dense_args('softmax'))(h)

    classifier = Model(inputs=inputs, outputs=[y_hat])
    for l in classifier.layers:
        l.freezable = l.name.startswith('f_')
    #classifier.compile(optimizer=Adam(lr=0.0002),
    classifier.compile(**compilation_args('classifier'))

    print '########### CLASSIFIER ############'
    classifier.summary()
    print '###################################'

    return classifier

### TODO Finish implementing regularization for what is below

def build_kl_mass(clf, w_clf=0.0001, w_kl=1, w_adv=None, loss=sculpting_kl_penalty):
    if w_adv is not None:
        w_kl = w_adv # backwards compatibility
    kl_input = Input(shape=(config.n_decorr_bins + 1,), name='kl_input')
    y_hat = clf.outputs[0]
    tag = Lambda(lambda x : x[:,-1:], output_shape=(1,))(y_hat)
    kl_output = concatenate([kl_input, tag], axis=-1, name='kl')
    inputs = clf.inputs + [kl_input]
    outputs = clf.outputs + [kl_output]

    kl = Model(inputs=inputs, outputs=outputs)
    kl.compile(optimizer=Adam(lr=0.0001),
               loss=['categorical_crossentropy', sculpting_kl_penalty],
               loss_weights=[w_clf, w_kl])

    print '########### KL-MODIFIED ############'
    kl.summary()
    print '###################################'

    return kl


def build_adversary(clf, loss, scale, w_clf, w_adv, n_outputs=1):
    if loss == 'mean_squared_error':
        config.n_decorr_bins = 1
    y_hat = clf.outputs[0]
    inputs= clf.inputs
    kin_hats = Adversary(config.n_decorr_bins, n_outputs=n_outputs, scale=scale)(y_hat)
    adversary = Model(inputs=inputs,
                      outputs=[y_hat]+kin_hats)
    adversary.compile(**compilation_args('adversary', 
                                         w_clf=w_clf, 
                                         w_adv=w_adv, 
                                         N=n_outputs, 
                                         loss=loss))

    print '########### ADVERSARY ############'
    adversary.summary()
    print '###################################'

    return adversary


    print '########### ADVERSARY ############'
    adversary.summary()
    print '###################################'

    return adversary


def build_old_classifier(dims):
    input_particles  = Input(shape=(dims[1], dims[2]), name='input_particles')
    input_mass = Input(shape=(1,), name='input_mass')
    input_pt = Input(shape=(1,), name='input_pt')
    inputs = [input_particles, input_mass, input_pt]

    # now build the particle network
    h = BatchNormalization(momentum=0.6)(input_particles)
    h = Conv1D(32, 2, activation='relu', kernel_initializer='lecun_uniform', padding='same')(h)
    h = BatchNormalization(momentum=0.6)(h)
    h = Conv1D(16, 4, activation='relu', kernel_initializer='lecun_uniform', padding='same')(h)
    h = BatchNormalization(momentum=0.6)(h)
    h = CuDNNLSTM(100)(h)
    h = BatchNormalization(momentum=0.6)(h)
    h = Dense(100, activation='relu', kernel_initializer='lecun_uniform')(h)
    particles_final = BatchNormalization(momentum=0.6)(h)

    # merge everything
    to_merge = [particles_final, input_mass, input_pt]
    h = concatenate(to_merge)

    for i in xrange(1,5):
        h = Dense(50, activation='tanh')(h)
    #    if i%2:
    #        h = Dropout(0.1)(h)
        h = BatchNormalization(momentum=0.6)(h)


    y_hat = Dense(config.n_truth, activation='softmax', name='y_hat')(h)

    classifier = Model(inputs=inputs, outputs=[y_hat])
    #classifier.compile(optimizer=Adam(lr=0.0002),
    classifier.compile(optimizer=getattr(keras_objects, OPTIMIZER)(lr=0.0005),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    print '########### CLASSIFIER ############'
    classifier.summary()
    print '###################################'

    return classifier


def partial_freeze(model, compargs):
    clone = Model(inputs=model.inputs, outputs=model.outputs)
    for l in clone.layers:
        if hasattr(l, 'freezable') and l.freezable:
            l.trainable = False 
    clone.compile(**compargs)
    return clone


# train any model
def train(model, name, train_gen, validation_gen, save_clf_params=None):
    if save_clf_params is not None:
        callbacks = [PartialModelCheckpoint(filepath='%s/%s/%s_clf_best.h5'%(MODELDIR,_APOSTLE,name), 
                                            save_best_only=True, verbose=True,
                                            **save_clf_params)]
        save_clf = save_clf_params['partial_model']
    else:
        save_clf = model
        callbacks = []
    callbacks += [ModelCheckpoint('%s/%s/%s_best.h5'%(MODELDIR,_APOSTLE,name), 
                                  save_best_only=True, verbose=True)]

    def save_classifier(name_=name, model_=save_clf):
        model_.save('%s/%s/%s.h5'%(MODELDIR,_APOSTLE,name_))

    def save_and_exit(signal=None, frame=None):
        save_classifier()
        exit(1)

    signal.signal(signal.SIGINT, save_and_exit)

    model.fit_generator(train_gen, 
                        steps_per_epoch=3000, 
                        epochs=NEPOCH,
                        validation_data=validation_gen,
                        validation_steps=2000,
                        callbacks = callbacks,
                       )
    save_classifier()


def infer(modelh5, name):
    model = load_model(modelh5,
                       custom_objects={'DenseBroadcast':DenseBroadcast,
                                       'GradReverseLayer':GradReverseLayer})
    model.summary()

    coll = generator.make_coll(BASEDIR + '/PARTITION/*_CATEGORY.npy')

    msd_norm_factor = 1. / config.max_mass
    pt_norm_factor = 1. / (config.max_pt - config.min_pt)
    msd_index = config.gen_singletons['msd']
    pt_index = config.gen_singletons['pt']

    calo = None
    if config.smear_params is not None:
        calo = CaloSmear(*config.smear_params) 

    def predict_t(data):
        msd = data['singletons'][:,msd_index] * msd_norm_factor
        pt = (data['singletons'][:,pt_index] - config.min_pt) * pt_norm_factor
        if msd.shape[0] > 0:
            particles = data['particles'][:,:config.limit,:]
            if calo:
                particles = calo(particles)
            particles = particles[:,:,:generator.truncate]
            r_t = model.predict([particles,msd,pt])[:,config.n_truth-1]
        else:
            r_t = np.empty((0,))

        return r_t 

    print 'loaded from',modelh5,
    print 'saving to',name
    coll.infer(['singletons','particles'], f=predict_t, name=name, partition='test')
