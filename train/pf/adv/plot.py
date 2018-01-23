#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import utils
import adversarial


import obj 
import config 

#config.DEBUG = True
#config.n_truth = 5
#config.truth = 'resonanceType'

n_batches = 200
#n_batches = 1
partition = 'test'


p = utils.Plotter()
r = utils.Roccer()

APOSTLE = 'panda_3'
OUTPUT = environ['BADNET_FIGSDIR'] + '/' + APOSTLE + '/'
system('mkdir -p %s'%OUTPUT)

#components=['singletons', 'inclusive', 'nn1', 'nn2']
components=['singletons', APOSTLE+'_conv', 'panda_3_shallow']


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(components, fpath) 
    return coll 

basedir = '/fastscratch/snarayan/pandaarrays/v1//PARTITION/'

colls = {
        't' : make_coll(basedir + '/ZpTT_*_CATEGORY.npy'),
        'q' : make_coll(basedir + '/QCD_*_CATEGORY.npy'),
        }

'''
colls = {
#  't' : make_coll('/fastscratch/snarayan/baconarrays/v12_repro/PARTITION/ZprimeToTTJet_4_*_CATEGORY.npy'),
#  'q' : make_coll('/fastscratch/snarayan/baconarrays/v12_repro/PARTITION/QCD_0_*_CATEGORY.npy') 
    't' : make_coll('/fastscratch/snarayan/baconarrays/v13_repro/PARTITION/ZprimeToTTJet_3_*_CATEGORY.npy'),
#    'h' : make_coll('/fastscratch/snarayan/baconarrays/v13_repro/PARTITION/ZprimeToA0hToA0chichihbb_2_*_CATEGORY.npy'),
    'q' : make_coll('/fastscratch/snarayan/baconarrays/v13_repro/PARTITION/QCD_1_*_CATEGORY.npy') 

}
'''


# run DNN
def predict(data, model=0):
    return data['panda_3_shallow']

def predict_conv(data, model):
    #return data[APOSTLE+'_conv'][:,model]
    return data[APOSTLE+'_conv']


f_vars = {
    'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01), r'$\tau_{32}$'),
    'tau21' : (lambda x : x['singletons'][:,obj.singletons['tau21']], np.arange(0,1.2,0.01), r'$\tau_{21}$'),
    'tau32SD' : (lambda x : x['singletons'][:,obj.singletons['tau32SD']], np.arange(0,1.2,0.01), r'$\tau_{32}^\mathrm{SD}$'),
    'tau21SD' : (lambda x : x['singletons'][:,obj.singletons['tau21SD']], np.arange(0,1.2,0.01), r'$\tau_{21}^\mathrm{SD}$'),
    'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5), 'Parton mass [GeV]'),
    'msd'     : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.), r'$m_\mathrm{SD}$ [GeV]'),
    'pt'        : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.), r'$p_\mathrm{T}$ [GeV]'),
    'top_ecf_bdt' : (lambda x : x['singletons'][:,obj.singletons['top_ecf_bdt']], np.arange(-1.2,1,0.05), 'ECF classifier'),
    'shallow_t' : (lambda x : predict(x, 0), np.arange(0,1.2,0.001), 'Shallow classifier'),
#    'shallow_h' : (lambda x : predict(x, 1), np.arange(0,1.2,0.001), 'Shallow classifier'),
    'classifier_conv_t'     : (lambda x : predict_conv(x, 0), np.arange(0,1.2,0.001), 'CLSTM'),
#    'classifier_conv_h'     : (lambda x : predict_conv(x, 1), np.arange(0,1.2,0.001), 'CLSTM'),
    'regularized_conv_t'     : (lambda x : predict_conv(x, 1), np.arange(0,1.2,0.001), r'$m_\mathrm{SD}$-ecorr. CLSTM'),
#    'regularized_conv_h'     : (lambda x : predict_conv(x, 3), np.arange(0,1.2,0.001), 'Decorrelated CLSTM'),
}

# f_vars2d = {
#     'correlation_reg' : (lambda x : (x['singletons'][:,obj.singletons['msd']], predict_conv(x, 2)),
#                          np.arange(40,400,10.),
#                          np.arange(0,1,0.01)),
#     'correlation_class' : (lambda x : (x['singletons'][:,obj.singletons['msd']], predict_conv(x, 0)),
#                            np.arange(40,400,10.),
#                            np.arange(0,1,0.01)),
# }


# unmasked first
hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                                 f_vars=f_vars,
                                 n_batches=n_batches, partition=partition)



#hists2d['q']['correlation_reg'].scale()
#hists2d['q']['correlation_class'].scale()
#hists2d['q']['correlation_reg'].plot(xlabel=r'$m_{SD}$', ylabel='Regularized NN', 
#                                     output=OUTPUT+'correlation_reg', norm=utils.lognorm)
#hists2d['q']['correlation_class'].plot(xlabel=r'$m_{SD}$', ylabel='Classifier NN', 
#                                       output=OUTPUT+'correlation_class', norm=utils.lognorm)



for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
#    hh = hists['h'][k]
    for h in [ht, hq]:
   # for h in [ht, hq, hh]:
        h.scale()
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
#    p.add_hist(hh, '2-prong Higgs', 'b')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot(output=OUTPUT+'unmasked_'+k, xlabel=f_vars[k][2])

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           {'tau32':r'$\tau_{32}$', 'tau32SD':r'$\tau_{32}^\mathrm{SD}$', 
            'tau21':r'$\tau_{21}$', 'tau21SD':r'$\tau_{21}^\mathrm{SD}$', 
            'classifier_t':'classifier', 
            'regularized_t':'regularized', 'msd':r'$m_\mathrm{SD}$',
            'top_ecf_bdt':'ECF classifier',
            'classifier_conv_t':'CLSTM',
            'regularized_conv_t':r'$m_\mathrm{SD}$-decorr. CLSTM',
            'shallow_t':r'Shallow NN'},
           )
r.plot(**{'output':OUTPUT+'unmasked_top_roc'})




# get the cuts
thresholds = [0, 0.5, 0.75, 0.9, 0.99, 0.999]

def sculpting(name, f_pred):
    h = hists['q'][name]
    tmp_hists = {t:{} for t in thresholds}
    f_vars2d = {
      'msd' : (lambda x : (x['singletons'][:,obj.singletons['msd']], f_pred(x)),
               np.arange(40,400,20.),
               np.arange(0,1,0.001)),
      'pt' : (lambda x : (x['singletons'][:,obj.singletons['pt']], f_pred(x)),
               np.arange(400,1000,50.),
               np.arange(0,1,0.001)),
      'partonM' : (lambda x : (x['singletons'][:,obj.singletons['partonM']], f_pred(x)),
               np.arange(0,400,20.),
               np.arange(0,1,0.001)),
      }

    h2d = colls['q'].draw(components=components,
                          f_vars={}, f_vars2d=f_vars2d,
                          n_batches=n_batches, partition=partition)

    for t in thresholds:
        cut = 0
        for ib in xrange(h.bins.shape[0]):
           frac = h.integral(lo=0, hi=ib) / h.integral()
           if frac >= t:
               cut = h.bins[ib]
               break
    
        print 'For classifier=%s, threshold=%.3f reached at cut=%.3f'%(name, t, cut )
    
        for k,h2 in h2d.iteritems():
            tmp_hists[t][k] = h2.project_onto_x(min_cut=cut)

    
    colors = utils.pl.cm.tab10(np.linspace(0,1,len(thresholds)))
    for k in tmp_hists[thresholds[0]]:
        p.clear()
        for i,t in enumerate(thresholds):
            p.add_hist(tmp_hists[t][k], 'Acceptance=%.3f'%(1-t), colors[i])
        p.plot(output=OUTPUT+name+'_progression_'+k, xlabel=f_vars[k][2], logy=True)


sculpting('regularized_conv_t', f_pred = lambda d : predict_conv(d, 1))
sculpting('shallow_t', f_pred = predict)
sculpting('classifier_conv_t', f_pred = lambda d : predict_conv(d, 0))
scuplting('tau32', f_pred = lambda x : x['singletons'][:,obj.singletons['tau32']])

# mask the top mass
def f_mask(data):
    mass = data['singletons'][:,obj.singletons['msd']]
    return (mass > 110) & (mass < 210)



hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars, n_batches=n_batches, partition=partition, f_mask=f_mask)


for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    # hh = hists['h'][k]
    for h in [ht, hq]:
        h.scale()
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    # p.add_hist(hh, '3-prong Higgs', 'b')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot(output=OUTPUT+'topmass_'+k, xlabel=f_vars[k][2])

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           {'tau32':r'$\tau_{32}$', 'tau32SD':r'$\tau_{32}^\mathrm{SD}$', 
            'tau21':r'$\tau_{21}$', 'tau21SD':r'$\tau_{21}^\mathrm{SD}$', 
            'classifier_t':'classifier', 
            'regularized_t':'regularized', 'msd':r'$m_\mathrm{SD}$',
            'top_ecf_bdt':'ECF classifier',
            'classifier_conv_t':'CLSTM',
            'regularized_conv_t':r'$m_\mathrm{SD}$-decorr. CLSTM',
            'shallow_t':r'Shallow NN'},
           )
r.plot(**{'output':OUTPUT+'topmass_top_roc'})
