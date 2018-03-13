from os import environ
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] = ""

from _common import *
from ..generators.gen_singletons import make_coll, generate
from ..generators import gen_singletons as generator

''' 
some global definitions
''' 

NEPOCH = 50
VERSION = 4
MODELDIR = environ.get('MODELDIR', 'models/') + '/shallow/'
BASEDIR = environ['BASEDIR']
BASEDIR_VAR = environ.get('BASEDIR_VAR')
VARIATIONS = ['ISRRenorm']
#VARIATIONS = ['ISRRenorm', 'FSRRenorm']
_APOSTLE = None

# must be called!
def instantiate(default_variables=config.gen_default_variables,
                default_mus=config.gen_default_mus,
                default_sigmas=config.gen_default_sigmas):
    global _APOSTLE
    _APOSTLE = 'v%s'%(str(VERSION))
    system('mkdir -p %s/%s/'%(MODELDIR,_APOSTLE))
    system('cp -v %s %s/%s/trainer.py'%(sys.argv[0], MODELDIR, _APOSTLE))
    system('cp -v %s %s/%s/lib.py'%(__file__.replace('.pyc','.py'), MODELDIR, _APOSTLE))

    config.gen_default_variables = default_variables
    config.gen_default_mus = default_mus
    config.gen_default_sigmas = default_sigmas

    with open('%s/%s/setup.py'%(MODELDIR, _APOSTLE),'w') as fsetup:
        fsetup.write('''
# auto-generated. do not edit!
from subtlenet import config
from subtlenet.generators import gen_singletons as generator
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
    qcd = make_coll(BASEDIR + '/PARTITION/QCD_*_CATEGORY.npy', label=0)

    data = [top, qcd]

    if BASEDIR_VAR is not None:
        counter = 1
        for v in VARIATIONS:
            suffixes = [''] if v == 'Nominal' else ['Up', 'Down']
            for s in suffixes:
                data.append(make_coll(BASEDIR_VAR + '/PARTITION/QCD_%s_*_CATEGORY.npy'%(v + s), 
                                      counter))
                counter += 1

    return data, (len(config.gen_default_variables),)


'''
first build the classifier!
'''

# set up data 
def setup_data(data, opts=None):
    if opts is None:
        opts = {}
    gen = {
        'train' : generate(data, partition='train', batch=1000, **opts),
        'validation' : generate(data, partition='validate', batch=10000, **opts),
        'test' : generate(data, partition='test', batch=10, **opts),
        }
    return gen

def setup_adv_data(data):
    opts = {'decorr_mass':True}
    gen = {
        'train' : generate(data, partition='train', batch=1000, **opts),
        'validation' : generate(data, partition='validate', batch=10000, **opts),
        'test' : generate(data, partition='test', batch=10, **opts),
        }
    return gen

# this is purely a discriminatory classifier
def build_classifier(dims):
    N = dims[0]
    inputs  = Input(shape=dims, name='input')
    h = inputs
    h = BatchNormalization(momentum=0.6)(h)
    h = Dense(2*N, activation='tanh',kernel_initializer='lecun_uniform') (h)
    h = Dense(2*N, activation='tanh',kernel_initializer='lecun_uniform') (h)
    h = Dense(2*N, activation='tanh',kernel_initializer='lecun_uniform') (h)
    h = Dense(2*N, activation='tanh',kernel_initializer='lecun_uniform') (h)
    y_hat   = Dense(config.n_truth, activation='softmax', name='y_hat') (h)

    classifier = Model(inputs=inputs, outputs=[y_hat])
    classifier.compile(optimizer=Adam(lr=0.0001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    print '########### CLASSIFIER ############'
    classifier.summary()
    print '###################################'

    return classifier

def build_kl_classifier(clf, n_class, loss='kl'):
    kl_input = Input(shape=(n_class,), name='kl_input')
    y_hat = clf.outputs[0]
    tag = Lambda(lambda x : x[:,-1:], output_shape=(1,))(y_hat)
    kl_output = concatenate([tag, kl_input], axis=-1, name='kl')
    inputs = clf.inputs + [kl_input]
    outputs = clf.outputs + [kl_output]

    lossfn = DistCompatibility(config.n_decorr_bins, n_class, loss)

    kl = Model(inputs=inputs, outputs=outputs)
    kl.compile(optimizer=Adam(lr=0.0001),
               loss=['categorical_crossentropy', lossfn],
               loss_weights=[1,1])

    print '########### KL-MODIFIED ############'
    kl.summary()
    print '###################################'

    return kl

def build_adversary(clf, loss, scale, w_clf, w_adv, N=config.n_decorr_bins):
    y_hat = clf.outputs[0]
    inputs= clf.inputs
    kin_hats = Adversary(N, n_outputs=1, scale=scale)(y_hat)
    adversary = Model(inputs=inputs,
                      outputs=[y_hat]+kin_hats)
    adversary.compile(optimizer=Adam(lr=0.00025),
                      loss=['categorical_crossentropy']+[loss for _ in kin_hats],
                      loss_weights=[w_clf]+[w_adv for _ in kin_hats])

    print '########### ADVERSARY ############'
    adversary.summary()
    print '###################################'

    return adversary


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
                        validation_steps=100,
                        callbacks = callbacks,
                       )
    save_classifier()


def infer(modelh5, name):
    print 'loading',modelh5
    model = load_model(modelh5,
                       custom_objects={'DenseBroadcast':DenseBroadcast})
    model.summary()

    coll = generator.make_coll(BASEDIR + '/PARTITION/*_CATEGORY.npy')

    def predict_t(data):
        inputs = data['singletons'][:,[config.gen_singletons[x] 
                                       for x in config.gen_default_variables]]
        if config.gen_default_mus is not None:
            mus = np.array(config.gen_default_mus)
            sigmas = np.array(config.gen_default_sigmas)

        if inputs.shape[0] > 0:
            if config.gen_default_mus is not None:
                inputs -= mus 
                inputs /= sigmas 
            r_t = model.predict(inputs)[:,config.n_truth-1]
        else:
            r_t = np.empty((0,))

        return r_t 

    coll.infer(['singletons'], f=predict_t, name=name, partition='test')
