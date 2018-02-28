#!/usr/bin/env python2.7

from _common import *

''' 
some global definitions
''' 

NEPOCH = 50
VERSION = 4
MODELDIR = environ.get('MODELDIR', 'models/particles/')
BASEDIR = environ['BASEDIR']
_APOSTLE = None
train_opts = {
        'learn_mass' : True,
        'learn_pt' : True,
        }

# must be called!
def instantiate(trunc=4, limit=50):
    global _APOSTLE
    generator.truncate = 4
    config.limit = 50
    _APOSTLE = 'v%i_trunc%i_limit%i'%(VERSION, generator.truncate, config.limit)
    system('mkdir -p %s/%s/'%(MODELDIR,_APOSTLE))
    system('cp -v %s %s/%s/trainer.py'%(sys.argv[0], MODELDIR, _APOSTLE))
    system('cp -v %s %s/%s/lib.py'%(__file__.replace('.pyc','.py'), MODELDIR, _APOSTLE))

    # instantiate data loaders 
    top = make_coll(BASEDIR + '/PARTITION/Top_*_CATEGORY.npy')
    qcd = make_coll(BASEDIR + '/PARTITION/QCD_*_CATEGORY.npy')

    data = [top, qcd]
    dims = get_dims(top)

    with open('%s/%s/setup.py'%(MODELDIR, _APOSTLE),'w') as fsetup:
        fsetup.write('''
from subtlenet import config
from subtlenet.generators import gen as generator
config.limit = %i
generator.truncate = %i
'''%(config.limit, generator.truncate))

    return data, dims


'''
first build the classifier!
'''

# set up data 
def setup_data(data):
    opts = {}; opts.update(train_opts)
    gen = {
        'train' : generate(data, partition='train', batch=500, **opts),
        'validation' : generate(data, partition='validate', batch=2000, **opts),
        'test' : generate(data, partition='test', batch=10, **opts),
        }
    return gen

def setup_adv_data(data):
    opts = {'decorr_mass':True}; opts.update(train_opts)
    gen = {
        'train' : generate(data, partition='train', batch=1000, **opts),
        'validation' : generate(data, partition='validate', batch=10000, **opts),
        'test' : generate(data, partition='test', batch=10, **opts),
        }
    return gen


# this is purely a discriminatory classifier
def build_classifier(dims):
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
    classifier.compile(optimizer=Adam(lr=0.0005),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    print '########### CLASSIFIER ############'
    classifier.summary()
    print '###################################'

    return classifier

def build_adversary(clf, loss, scale, w_clf, w_adv):
    y_hat = clf.outputs[0]
    inputs= clf.inputs
    kin_hats = Adversary(config.n_decorr_bins, n_outputs=1, scale=scale)(y_hat)
    adversary = Model(inputs=inputs,
                      outputs=[y_hat]+kin_hats)
    adversary.compile(optimizer=Adam(lr=0.00025),
                      loss=['categorical_crossentropy']+[emd for _ in kin_hats],
                      loss_weights=[w_clf]+[w_adv for _ in kin_hats])

    print '########### ADVERSARY ############'
    adversary.summary()
    print '###################################'

    return adversary


# train any model
def train(model, name, train_gen, validation_gen, callbacks=[]):
    # ctrl+C now triggers a graceful exit
    def save_classifier(name_=name, model_=model):
        model_.save('%s/%s/%s.h5'%(MODELDIR,_APOSTLE,name_))

    def save_and_exit(signal=None, frame=None):
        save_classifier()
        exit(1)

    signal.signal(signal.SIGINT, save_and_exit)

    callbacks_ = [ModelCheckpoint('%s/%s/%s_best.h5'%(MODELDIR,_APOSTLE,name), 
                                  save_best_only=True, verbose=True)]
    callbacks_.extend(callbacks)

    model.fit_generator(train_gen, 
                        steps_per_epoch=3000, 
                        epochs=NEPOCH,
                        validation_data=validation_gen,
                        validation_steps=1000,
                        callbacks = callbacks_,
                       )
    save_classifier()


def infer(modelh5, name):
    print 'loading',modelh5
    model = load_model(modelh5,
                       custom_objects={'DenseBroadcast':DenseBroadcast})
    model.summary()

    coll = generator.make_coll(BASEDIR + '/PARTITION/*_CATEGORY.npy')

    msd_norm_factor = 1. / config.max_mass
    pt_norm_factor = 1. / (config.max_pt - config.min_pt)
    msd_index = config.gen_singletons['msd']
    pt_index = config.gen_singletons['pt']

    def predict_t(data):
        msd = data['singletons'][:,msd_index] * msd_norm_factor
        pt = (data['singletons'][:,pt_index] - config.min_pt) * pt_norm_factor
        if msd.shape[0] > 0:
            particles = data['particles'][:,:config.limit,:generator.truncate]
            r_t = model.predict([particles,msd,pt])[:,config.n_truth-1]
        else:
            r_t = np.empty((0,))

        return r_t 

    print 'saving to',name
    coll.infer(['singletons','particles'], f=predict_t, name=name, partition='test')
