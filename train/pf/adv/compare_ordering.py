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

APOSTLE = 'panda_4'
OUTPUT = environ['BADNET_FIGSDIR'] + '/' + APOSTLE + '/'
system('mkdir -p %s'%OUTPUT)

#components=['singletons', 'inclusive', 'nn1', 'nn2']
components={ 'pt' : ['panda_3_shallow', APOSTLE+'_conv'],
             'kt' : [APOSTLE+'_kt_conv'],
             'akt' : [APOSTLE+'_akt_conv',APOSTLE+'b_akt_conv']}
for _,c in components.iteritems():
    c.append('singletons')


def make_coll(fpath,l):
    coll = obj.PFSVCollection()
    coll.add_categories(components[l], fpath) 
    return coll 

basedir = '/fastscratch/snarayan/pandaarrays/v1%s//PARTITION/'

colls = {}

colls['pt'] = {'t' : make_coll(basedir%'' + '/Top_*_CATEGORY.npy','pt'),
               'q' : make_coll(basedir%'' + '/QCD_*_CATEGORY.npy','pt')}
colls['kt'] = {'t' : make_coll(basedir%'_kt' + '/Top_*_CATEGORY.npy','kt'),
               'q' : make_coll(basedir%'_kt' + '/QCD_*_CATEGORY.npy','kt')}
colls['akt'] = {'t' : make_coll(basedir%'_akt' + '/Top_*_CATEGORY.npy','akt'),
                'q' : make_coll(basedir%'_akt' + '/QCD_*_CATEGORY.npy','akt')}


# run DNN
def predict_shallow(data, model=0):
    return data['panda_3_shallow']

def predict_pt(data):
    return data[APOSTLE+'_conv']

def predict_kt(data):
    return data[APOSTLE+'_kt_conv']

def predict_akt(data, model):
    return data[APOSTLE+'_akt_conv'][:,model]

def predict_akt_decorr(data):
    return data[APOSTLE+'b_akt_conv']

f_vars = {}

f_vars['pt'] = {
    'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01), r'$\tau_{32}$'),
    'tau21' : (lambda x : x['singletons'][:,obj.singletons['tau21']], np.arange(0,1.2,0.01), r'$\tau_{21}$'),
    'tau32SD' : (lambda x : x['singletons'][:,obj.singletons['tau32SD']], np.arange(0,1.2,0.01), r'$\tau_{32}^\mathrm{SD}$'),
    'tau21SD' : (lambda x : x['singletons'][:,obj.singletons['tau21SD']], np.arange(0,1.2,0.01), r'$\tau_{21}^\mathrm{SD}$'),
    'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5), 'Parton mass [GeV]'),
    'msd'     : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.), r'$m_\mathrm{SD}$ [GeV]'),
    'rawrho'     : (lambda x : x['singletons'][:,obj.singletons['rawrho']], np.arange(-5.,7.5,.5), r'$\rho$'),
    'rawpt'        : (lambda x : x['singletons'][:,obj.singletons['rawpt']], np.arange(250.,1000.,50.), r'$p_\mathrm{T}$ [GeV]'),
    'top_ecf_bdt' : (lambda x : x['singletons'][:,obj.singletons['top_ecf_bdt']], np.arange(-1.2,1,0.05), 'ECF classifier'),
    'shallow_t' : (lambda x : predict_shallow(x), np.arange(0,1.2,0.001), 'Shallow NN'),
    'classifier_pt'     : (lambda x : predict_pt(x), np.arange(0,1.2,0.001), 'CLSTM'),
}

f_vars['kt'] = {
    'classifier_kt'     : (lambda x : predict_kt(x), np.arange(0,1.2,0.001), 'CLSTM'),
}

f_vars['akt'] = {
    'classifier_akt'     : (lambda x : predict_akt(x,0), np.arange(0,1.2,0.001), r'CLSTM'),
    'classifier_akt_decorr'     : (lambda x : predict_akt_decorr(x), np.arange(0,1.2,0.001), r'$m_\mathrm{SD}$-decorr CLSTM'),
}



# unmasked first
hists = {'q':{}, 't':{}}
for name,c in colls.iteritems():
    for k,v in c.iteritems():
        hists[k].update(v.draw(components=components[name],
                               f_vars=f_vars[name],
                               n_batches=n_batches, partition=partition))



for _,f in f_vars.iteritems():
   for k,args in f.iteritems():
       ht = hists['t'][k]
       hq = hists['q'][k]
       for h in [ht, hq]:
           h.scale()
       p.clear()
       p.add_hist(ht, '3-prong top', 'r')
       p.add_hist(hq, '1-prong QCD', 'k')
       p.plot(output=OUTPUT+'unmasked_'+k, xlabel=args[2])

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           {'tau32':r'$\tau_{32}$', 'tau32SD':r'$\tau_{32}^\mathrm{SD}$', 
            'classifier_pt':r'$p_\mathrm{T}$-ordered', 
            'classifier_kt':r'$k_\mathrm{T}$-ordered', 
            'classifier_akt':r'anti-$k_\mathrm{T}$-ordered', 
            'shallow_t':r'Shallow NN'},
           )
r.plot(**{'output':OUTPUT+'unmasked_top_roc'})




# get the cuts
thresholds = [0, 0.5, 0.75, 0.9, 0.99, 0.999]

def sculpting(name, comp, f_pred, logy=True):
    h = hists['q'][name]
    tmp_hists = {t:{} for t in thresholds}
    f_vars2d = {
         'msd' : (lambda x : (x['singletons'][:,obj.singletons['msd']], f_pred(x)),
                  np.arange(40,400,20.),
                  np.arange(0,1,0.001)),
         'rawpt' : (lambda x : (x['singletons'][:,obj.singletons['rawpt']], f_pred(x)),
                  np.arange(400,1000,50.),
                  np.arange(0,1,0.001)),
         'rawrho' : (lambda x : (x['singletons'][:,obj.singletons['rawrho']], f_pred(x)),
                  np.arange(-5,7.5,.5),
                  np.arange(0,1,0.001)),
      }

    h2d = colls[comp]['q'].draw(components=components[comp],
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
            if not logy:
                tmp_hists[t][k].scale()
            p.add_hist(tmp_hists[t][k], 'Acceptance=%.3f'%(1-t), colors[i])
        if logy:
            p.plot(output=OUTPUT+name+'_progression_'+k, xlabel=f_vars['pt'][k][2], logy=logy)
        else:
            p.plot(output=OUTPUT+name+'_progstacked_'+k, xlabel=f_vars['pt'][k][2], logy=logy)


sculpting('classifier_akt', 'akt', f_pred = lambda d : predict_akt(d,0))
sculpting('classifier_akt', 'akt', f_pred = lambda d : predict_akt(d,0), logy=False)
sculpting('classifier_akt_decorr', 'akt', f_pred = predict_akt_decorr)
sculpting('classifier_akt_decorr', 'akt', f_pred = predict_akt_decorr, logy=False)

# mask the top mass
def f_mask(data):
    mass = data['singletons'][:,obj.singletons['msd']]
    return (mass > 130) & (mass < 180)



hists = {'q':{}, 't':{}}
for name,c in colls.iteritems():
    for k,v in c.iteritems():
        hists[k].update(v.draw(components=components[name],
                               f_vars=f_vars[name],
                               f_mask=f_mask,
                               n_batches=n_batches, partition=partition))



for _,f in f_vars.iteritems():
   for k,args in f.iteritems():
       ht = hists['t'][k]
       hq = hists['q'][k]
       for h in [ht, hq]:
           h.scale()
       p.clear()
       p.add_hist(ht, '3-prong top', 'r')
       p.add_hist(hq, '1-prong QCD', 'k')
       p.plot(output=OUTPUT+'topmass_'+k, xlabel=args[2])

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           {'tau32':r'$\tau_{32}$', 'tau32SD':r'$\tau_{32}^\mathrm{SD}$', 
#            'classifier_pt':r'$p_\mathrm{T}$-ordered', 
#            'classifier_kt':r'$k_\mathrm{T}$-ordered', 
            'classifier_akt':r'anti-$k_\mathrm{T}$-ordered', 
            'classifier_akt_decorr':r'decorr. anti-$k_\mathrm{T}$-ordered', 
            'shallow_t':r'Shallow NN'},
           )
r.plot(**{'output':OUTPUT+'topmass_top_roc'})
