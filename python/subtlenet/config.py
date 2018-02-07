DEBUG = False            # fill your screen with debugging output 

truth = 'nPartons'       # obsolete kinda
n_truth = 4              # size of one-hot
n_btruth = 4              # size of one-hot

weights_scale = None     # honestly don't remember

limit = None             # whether to limit the number of input particles

# decorrelation
n_decorr_bins = 15       # granularity of binning
max_mass = 300.          # maximum msd
max_rho = 7.5            # maximum rho
min_rho = -5.            # minimum rho
max_pt = 1000.           # maximum pT
min_pt = 450.            # minimum pT
adversary_mask = 1       # which category we should apply the decorrelation to, e.g. only on QCD(=1prong)

## reco singletons ## 
_singletons = ['msd','pt', 'rawpt', 'eta', 'phi',  'eventNumber',
               'partonM', 'partonPt', 'partonEta', 'nPartons',
               'nBPartons', 'nCPartons',
               'rho','rawrho','rho2','rawrho2',
               'tau32','tau32SD','tau21','tau21SD',
               'maxcsv','mincsv','doubleb', # uncomment in a bit
               'top_ecf_bdt']

singletons = {_singletons[x]:x for x in xrange(len(_singletons))}

default_variables=['msd','rawpt','tau32SD','tau32', 'top_ecf_bdt', 'maxcsv', 'mincsv', 'doubleb'] 
default_mus      =[  150,    1000, 0.5,0.5, 0, 0.5, 0.5, 0.5]
default_sigmas   =[   50,    500, 0.5,0.5, 1, 0.5, 0.5, 0.5]

## gen singletons ## 
_gen_singletons = ['pt', 'eta', 'phi', 'm', 'msd',
                   'tau3', 'tau2', 'tau1',
                   'tau3sd', 'tau2sd', 'tau1sd',
                   'nprongs', 'partonpt', 'partonm']

gen_singletons = {_gen_singletons[x]:x for x in xrange(len(_gen_singletons))}

gen_default_variables=['eta', 'phi', 'm', 'msd', 'tau3', 'tau2', 'tau1', 'tau3sd', 'tau2sd', 'tau1sd']
gen_default_mus      =[   0,    0,   50,     50,    0.5,    0.5,    0.5,      0.5,      0.5,      0.5]
gen_default_sigmas   =[   3,    2,   50,     50,    0.5,    0.5,    0.5,      0.5,      0.5,      0.5]
#gen_default_variables=['pt', 'eta', 'phi', 'm', 'msd', 'tau3', 'tau2', 'tau1', 'tau3sd', 'tau2sd', 'tau1sd']
#gen_default_mus      =[ 500,    0,    0,   50,     50,    0.5,    0.5,    0.5,      0.5,      0.5,      0.5]
#gen_default_sigmas   =[ 500,    3,    2,   50,     50,    0.5,    0.5,    0.5,      0.5,      0.5,      0.5]
