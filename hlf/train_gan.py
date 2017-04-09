#!/usr/bin/env python

from sys import exit 
from os import environ
environ['KERAS_BACKEND'] = 'theano'

import numpy as np
from functools import partial 
from tqdm import tqdm 

from utils import *

from keras.layers import Input, Dense, Dropout, Activation, concatenate
from keras.models import Model 
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam 
from keras.utils import np_utils
from keras.losses import categorical_crossentropy

### HELPERS ### 

def reform(arr, train_frac, val_frac, fields, weight, label, extra_fields=[]):
    n = arr.shape[0] 
    ns = {}
    ns['train'] = (0,int(train_frac*n))
    ns['val'] = (ns['train'][1],ns['train'][1]+int(val_frac*n))
    ns['test'] = (ns['val'][1],n)
    print 'label=%i, n_train=%i, n_val=%i, n_test=%i'%(label,ns['train'][1],ns['val'][1]-ns['val'][0],ns['test'][1]-ns['test'][0])
    weight_norm = 100. / np.sum(arr[weight])
    x = {}; y = {}; w = {}; extras = {}
    for subset in ['train','val','test']:
        n_ = ns[subset]
        x[subset] = arr[fields].view(np.float64).reshape(arr[fields].shape+(-1,))[n_[0]:n_[1]]
        w[subset] = arr[weight][n_[0]:n_[1]] * weight_norm 
        y[subset] = (label * np.ones(n_[1]-n_[0])).astype(int)
        for e in extra_fields:
            extras[subset+'_'+e] = arr[e][n_[0]:n_[1]]
    return {'x':x,'y':y,'w':w,'extras':extras}
    

def load_data(train_frac,val_frac,fields):
    # arr_bkg = np.load('../data/Background_selected.npy')
    # arr_sig = np.load('../data/Top_selected.npy')
    arr_bkg = np.load('../data/QCD_goodjets.npy')
    arr_sig = np.load('../data/ZpTT_goodjets.npy')

    np.random.shuffle(arr_bkg)
    np.random.shuffle(arr_sig)

    bkg = reform(arr_bkg,train_frac,val_frac,fields,'weight',0,['top_ecf_bdt','msd'])
    sig = reform(arr_sig,train_frac,val_frac,fields,'weight',1,['top_ecf_bdt','msd'])

    x = {}; y = {}; w = {}; bdt = {}; mass = {}

    for subset in ['train','val','test']:
        x[subset] = np.concatenate((bkg['x'][subset],sig['x'][subset]), axis=0)
        w[subset] = np.concatenate((bkg['w'][subset],sig['w'][subset]), axis=0)
        bdt[subset] = np.concatenate((bkg['extras'][subset+'_top_ecf_bdt'],
                                      sig['extras'][subset+'_top_ecf_bdt']), axis=0)
        mass[subset] = np.concatenate((bkg['extras'][subset+'_msd'],
                                       sig['extras'][subset+'_msd']), axis=0)
        y_vec = np.concatenate((bkg['y'][subset],sig['y'][subset]), axis=0)
        y[subset] = np_utils.to_categorical(y_vec, 2)
        mass[subset] = mass[subset].reshape((mass[subset].shape[0],1))
        bdt[subset] = bdt[subset].reshape((bdt[subset].shape[0],1))

    return x,y,w,bdt,mass 


### ACQUIRE DATA ###

fields = ['tau32sd','frec'] + ['ecf%i'%i for i in xrange(11)]
x,y,w,bdt,mass = load_data(0.5,0.25,fields)
for subset in bdt:
    bdt[subset] = bdt[subset].reshape((bdt[subset].shape[0],))
dim = x['train'].shape[1]
y_gan = {subset:np.concatenate([y[subset],mass[subset]], axis=1)
         for subset in y}


### BUILD THE MODELS ### 

# Discrimination model 
d_input = Input(shape=(dim,), name='hlf')

l = Dense(64, activation='relu')(d_input)
l = Dense(64, activation='relu')(l)
l = Dense(32, activation='relu')(l)
d_output = Dense(2, activation='softmax', name='hlf_disc')(l)

d_model = Model(inputs=d_input, outputs=d_output)
d_model.compile(optimizer=Adam(),
               loss='categorical_crossentropy')

d_model.summary()


# Generation model
g_input = Input(shape=(2,),name='disc')
# l = GradientReversalLayer(hp_lambda=100, name='reversal')(g_input)
l = Dense(32, activation='relu')(g_input)
l = Dense(32, activation='sigmoid')(l)
g_output = Dense(1, activation='linear', name='hlf_gen')(l)

g_model = Model(inputs=g_input, outputs=g_output)
g_model.compile(optimizer=Adam(lr=1.),
                loss='mse')

g_model.summary() 


# Add the models
gan_input = Input(shape=(dim,), name='hlf_gan')
gan_d = d_model(gan_input)
gan_reverse_1 = GradientReversalLayer(hp_lambda=1, name='reversal_1')(gan_d)
gan_g = g_model(gan_reverse_1)
gan_reverse_2 = GradientReversalLayer(hp_lambda=1, name='reversal_2')(gan_g)
gan_output = concatenate([gan_d,gan_reverse_2],axis=1)

my_adversarial_loss = partial(adversarial_loss, g_weight=1.)
my_adversarial_loss.__name__ = "my_adversarial_loss" # partial doesn't do this for some reason
gan_model = Model(inputs=gan_input, outputs=gan_output)
gan_model.compile(optimizer=Adam(lr=0.001),
                  loss=my_adversarial_loss)


### PRE-TRAIN DISCRIMINATOR ### 

d_model.fit(x['train'], y['train'], sample_weight=w['train'],
            batch_size=500, epochs=1, verbose=1,
            shuffle=True)
y_pred_v0 = d_model.predict(x['test'])


### PRE-TRAIN GENERATOR ###

y_pred = d_model.predict(x['train'])
bkg_mask = y['train'][:,0]==1 
g_model.fit(y_pred[bkg_mask], mass['train'][bkg_mask], 
            sample_weight=w['train'][bkg_mask], 
            batch_size=32, epochs=1, verbose=1, 
            shuffle=True)


### TRAIN THE ADVERSARIAL STACK ###

n_test_fast = 20
test_idx = np.random.random_integers(low=0,high=x['test'].shape[0],size=n_test_fast)
# y_pred = gan_model.predict(x['test'][test_idx])
# for i in range(n_test_fast): 
#     print 'tag: %i -> %4.3f, mass: %6.3f -> %6.3f'%(y_gan['test'][test_idx[i]][1],
#                                                     y_pred[i][1],
#                                                     y_gan['test'][test_idx[i]][2],
#                                                     y_pred[i][2],)
 

# checkpoint = ModelCheckpoint(filepath='simple_disc.h5', save_best_only=True)
for big_epoch in range(1):
    batch_size = 500
    n_train = x['train'].shape[0]
    n_batch = n_train / batch_size 
    order = range(n_train)
    np.random.shuffle(order)
    for batch in tqdm(range(n_batch)):
        idxs = order[batch*batch_size : (batch+1)*batch_size]

        w_ = w['train'][idxs]
        x_ = x['train'][idxs]
        y_ = y['train'][idxs]
        y_gan_ = y_gan['train'][idxs]
        mass_ = mass['train'][idxs]

        bkg_mask = y_[:,0]==1

        # # now train the stack 
        # make_trainable(g_model,False)
        gan_loss = gan_model.train_on_batch(x_, y_gan_, sample_weight=w_)
        # make_trainable(g_model,True)

        # run the discriminator 
        y_pred = d_model.predict(x_)
        d_loss = d_model.evaluate(x_, y_,
                                  verbose=0, sample_weight=w_)
        
        # train the generator 
        g_loss = g_model.train_on_batch(y_pred[bkg_mask], mass_[bkg_mask], 
                                        sample_weight=w_[bkg_mask])

        # if batch%1000==0:
        #     print d_loss, g_loss, gan_loss 

    # y_pred = d_model.predict(x['val'])
    # print d_model.evaluate(x['val'],y['val'],
    #                  verbose=1, sample_weight=w['val'])
    # print g_model.evaluate(y_pred,mass['val'],
    #                  verbose=1, sample_weight=w['val'])
    # print gan_model.evaluate(x['val'],y_gan['val'],
    #                    verbose=1, sample_weight=w['val'])





y_pred_v1 = gan_model.predict(x['test'])


dnn_v0_t = Tagger(y_pred_v0[:,1], 'DNN v0', 0, 1, False)
dnn_v1_t = Tagger(y_pred_v1[:,1], 'DNN v1', 0, 1, False)
bdt_t = Tagger(bdt['test'], 'BDT', -1, 1, False)
create_roc([dnn_v0_t,dnn_v1_t,bdt_t],
           np.argmax(y['test'],axis=1),
           w['test'],'gan')


mask = np.logical_and(110<mass['test'], mass['test']<210).reshape((y['test'].shape[0],))
dnn_v0_t_mass = Tagger(y_pred_v0[:,1][mask], 'DNN v0', 0, 1, False)
dnn_v1_t_mass = Tagger(y_pred_v1[:,1][mask], 'DNN v1', 0, 1, False)
bdt_t_mass = Tagger(bdt['test'][mask], 'BDT', -1, 1, False)
wps = create_roc([dnn_v0_t_mass, dnn_v1_t_mass, bdt_t_mass],
           np.argmax(y['test'][mask],axis=1),
           w['test'][mask],'gan_mass')

print wps

mask_v0 = np.logical_and(y_pred_v0[:,1]>wps[0], y['test'][:,0]==1)
mask_v1 = np.logical_and(y_pred_v1[:,1]>wps[1], y['test'][:,0]==1)
mask_bdt = np.logical_and(bdt['test']>wps[2], y['test'][:,0]==1)
mask_bkg = y['test'][:,0]==1
mass_test = mass['test'].reshape((mass['test'].shape[0],))
props = {'xlabel' : '$m_{SD}$ [GeV]',
         'bins' : np.arange(0,500,20),
         'output' : 'sculpt'}
h_inc = {'vals':mass_test[mask_bkg],
        'weights':w['test'][mask_bkg],
        'color':'b', 'label':'Inclusive'}
h_v0 = {'vals':mass_test[mask_v0],
        'weights':w['test'][mask_v0],
        'color':'k', 'label':'DDN v0'}
h_v1 = {'vals':mass_test[mask_v1],
        'weights':w['test'][mask_v1],
        'color':'r', 'label':'DDN v1'}
h_bdt = {'vals':mass_test[mask_bdt],
        'weights':w['test'][mask_bdt],
        'color':'g', 'label':'BDT'}
plot_hists(props, [h_inc, h_v0, h_v1])