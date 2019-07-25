import argparse
import os
import json
import uproot
import numpy as np
import pandas as pd
import re

# grabbing command line args
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='dir where output files will be stored')
parser.add_argument('--name', type=str, help='name of sample to process')
parser.add_argument('--json', type=str, help='json file controlling which files/samples/features are used')

args = parser.parse_args()

# making the output direrctory in case it doesn't already exist
try:
    os.makedirs(args.out)
except:
    pass

# parsing the json file
with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    basedir = payload['base']
    features = payload['features']
    weight = payload['weight']
    cut_vars = payload['cut_vars']
    cut = payload['cut']
    substructure_vars = payload['substructure_vars']
    default = payload['default']
    per_part = bool(payload['per_part'])
    print "per_part: ", per_part
    nparticles = payload['nparticles']

    y = 0
    for sample in payload['samples']:
        y += 1
        if sample['name'] == args.name:
            samples = sample['samples']
    
    in_cut_vars_only = list(set(cut_vars) - set(features))


# reading from the root files
filenames = [basedir+'/'+sample+'.root' for sample in samples]
files = [uproot.open(f) for f in filenames]
trees = [f['Events'] for f in files]
keys = trees[0].keys()
#uncomment the line below to see what keys will be accepted for features/defualts in the json
#print '\n Keys: \n', keys[:20], type(keys)

#if no features are provided, want to grab features w same number of entries as the default
if features == []:
    try:
        default
    except ValueError:
        print 'no default provided'
    as_dict = trees[0].arrays(keys)
    default_shape = as_dict[default].shape
    #print '\n\ndefault_shape: ', default_shape
    for k in keys:
        shape = as_dict[k].shape
        #print shape
        if shape == default_shape:
            #print 'appending^'
            features.append(k)

# takes a dict produced by tree.arrays and reformats it
# the input dict has var names for keys, and 2d (evt and particle) arrays for values
# the output dict has keys like 'fj_cpf_pt[0]' and 1d arrays (evt) for values
def reformat_dict(d):
    ks = []
    vs = []

    for k, v in d.iteritems():
        if v.ndim == 1:
            ks.append(k)
            vs.append(v)
        elif v.ndim == 2:
            for i in range(nparticles):
                ks.append(k+"[{}]".format(i))
                vs.append([v[j][i] for j in range(len(v))])
        else:
            raise ValueError("Too many dimensions")

    new_dict = dict(zip(ks, vs))
    return new_dict


def get_branches_as_df(branches, mode):
    dicts = [tree.arrays(branches=branches) for tree in trees] 
    if per_part:
        dicts = [reformat_dict(d) for d in dicts]

    dfs = [pd.DataFrame.from_dict(d) for d in dicts]
    df = pd.concat(dfs)
    #print mode, '\n', df.head()
    if mode=='features': 
        #the first call to this function must have mode=features for everything to work
        df = df[eval(cut)]
    return df.reset_index(drop=True)


# unpacking the branches into x, y, weight and substructure vars       
X = get_branches_as_df(features + in_cut_vars_only, 'features')
#print "X before: ", X.shape, '\n', X.head()
#print list(X.columns)

def is_extra(s):
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0)) >= nparticles
    return True

if per_part:
    extra_columns = list(filter(is_extra, list(X.columns)))
    weird_columns = list(filter(lambda s: "fj_pt" in s, list(X.columns)))
    X = X.drop(in_cut_vars_only + weird_columns + extra_columns, axis=1)
else:
    X = X.drop(in_cut_vars_only, axis=1)

#print '\n\n\n', list(X.columns)
#print "dtypes: ", X.dtypes
W = get_branches_as_df([weight], 'weight')
Y = pd.DataFrame(data=y*np.ones(shape=W.shape))
substructure = get_branches_as_df(substructure_vars, 'substructure')

# writing to .pkl files
def save(df, label):
    fout = args.out+'/'+args.name+'_'+label+'.pkl'
    df.to_pickle(fout)
    print '\n'+label+'\n', df.shape, '\n'
    print df.head()

save(X, 'x')
save(Y, 'y')
save(W, 'w')
save(substructure, 'ss_vars')

