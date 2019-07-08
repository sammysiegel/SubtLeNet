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

    y = 0
    for sample in payload['samples']:
        y += 1
        if sample['name'] == args.name:
            samples = sample['samples']

# reading from the root files
filenames = [basedir+'/'+sample+'.root' for sample in samples]
files = [uproot.open(f) for f in filenames]
trees = [f['Events'] for f in files]
#print trees[0].keys()

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
    multiple_files = len(files) > 1
    dict_ = merge_dicts([tree.arrays(branches) for tree in trees])
    #print '\n', dict_
    values = np.array(dict_.values())
    if mode == 'features':
        values = np.swapaxes(values,1,2)
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
