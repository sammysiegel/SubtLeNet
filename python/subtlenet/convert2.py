import argparse
import os
import json
import uproot
import numpy as np
import pandas as pd

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

    y = 0
    for sample in payload['samples']:
        y += 1
        if sample['name'] == args.name:
            samples = sample['samples']

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

#mask = pd.Series()  
def get_branches_as_df(branches, mode):
    dfs = [tree.pandas.df(branches=branches) for tree in trees]
    df = pd.concat(dfs)
    global mask
    if mode=='features': #the first call to this function must have mode=features for everything to work
        #mask = eval(cut).bool()
        df = df[eval(cut)]
    print "in get_branches; mode: "+mode+"\n", df.head()
    return df.reset_index(drop=True)
       
in_cut_vars_only = list(set(cut_vars) - set(features))
#print "features: ", features, '\n'
#print "in_cut_vars_only: ", in_cut_vars_only, '\n'
print "\n branches used for x: ", features + in_cut_vars_only, '\n'
X = get_branches_as_df(features + in_cut_vars_only, 'features')
print list(X.columns)
X = X.drop(in_cut_vars_only, axis=1)
print '\n\n\n', list(X.columns)
W = get_branches_as_df([weight], 'weight')
Y = pd.DataFrame(data=y*np.ones(shape=W.shape))
substructure = get_branches_as_df(substructure_vars, 'substructure')

# writing to .pkl files

def save(df, label):
    fout = args.out+'/'+args.name+'_'+label+'.pkl'
    df.to_pickle(fout)
    #print '\n'+label+'\n', df.shape, '\n'
    #print df.head()

save(X, 'x')
save(Y, 'y')
save(W, 'w')
save(substructure, 'ss_vars')

