import argparse
import os
import json
import uproot
import numpy as np
from collections import defaultdict

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
#print '\n Keys: \n', keys, type(keys)

#if no features are provided, want to grab features w same number of entries as the default
if features == []:
    try:
        default
    except NameError:
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

#print features

#takes a list of dictionaries and returns a single dictionary
def merge_dicts(dicts):
    if len(dicts) == 1:
        return dicts[0]
    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.iteritems():
            merged[k].append(v)
    for k, v in merged.iteritems():
        merged[k] = np.concatenate(v)
    return dict(merged)

def get_branches_as_np(branches, mode):
    dict_ = merge_dicts([tree.arrays(branches) for tree in trees])
    #print '\n', dict_.values(), type(dict_.values())
    values = np.array(dict_.values())
    if mode == 'features' and values.ndim > 2:
        values = np.swapaxes(values,1,2)
    elif mode == 'features':
        pass
    elif mode == 'weight':
        values = values.flatten()
    elif mode != 'substructure':
        raise ValueError('Unrecgognized mode passed to get_branches_as_np')
    #print '\n mode=', mode, '\n', values, '\n', values.shape
    return values
        

X = get_branches_as_np(features, 'features')
W = get_branches_as_np([weight], 'weight')
Y = y * np.ones(shape=W.shape)
substructure = get_branches_as_np(substructure_vars, 'substructure')

# writing to .npy files

def save(arr, label):
    fout = args.out+'/'+args.name+'_'+label+'.npy'
    np.save(fout, arr)

save(X, 'x')
save(Y, 'y')
save(W, 'w')
save(substructure, 'ss_vars')
save(keys, 'all_keys')
save(features, 'used_keys')
