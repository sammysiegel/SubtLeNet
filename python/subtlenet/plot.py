import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import argparse
import json
import os

#Note: right now, the out arg can only be used to name the pdf; the pdf will be created in the same dir as the script

#grabbing command line args
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='path/filename where output pdf will be created (including the .pdf extension is optional)')
parser.add_argument('--json', type=str, help='json file controlling the input files used')
parser.add_argument('--maxz', help='data with zscores above this value will not be plotted', action='store_true')
args = parser.parse_args()

#separating the 'out' argument into the file path and the file name
split = args.out.split('/')
out_dir = '/'+'/'.join(split[:-1])
out_name = split[-1]

#adding the .pdf in case it's not there
if out_name.split('.')[-1] != 'pdf':
    out_name = out_name + '.pdf'

#making the output directory in case it doesn't already exist
parent_dir = os.getcwd()
try:
    os.makedirs(parent_dir+out_dir)
except:
    pass

out = PdfPages(out_name)  #(out_dir + '/' + out_name)

#parsing the json file
with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    base_dir = payload['base_dir']
    filenames = payload['filenames']

dfs = {}

for k, v in filenames.iteritems():
    dfs[k] = pd.read_pickle(base_dir+v+".pkl")

var_names = list(dfs[list(dfs)[0]])

#print var_names
#for k, v in dfs.iteritems():
#    print v.shape

def make_hist(var):
    #Plots a histogram comparing var across all dataframes
    plt.figure(figsize=(4, 4), dpi=80)
    plt.xlabel(var)
    plt.title(var)
    
    min_ = min([v[var].min() for k, v in dfs.iteritems()])
    max_ = max([v[var].max() for k, v in dfs.iteritems()])

    if min_ == max_:
        if np.abs(min_) < 0.0001:
            delta = 1
        else:
            delta = 0.25 * min_
        min_ -= delta
        max_ += delta

    plt.xlim(min_, max_)
    bins = np.linspace(min_, max_, 100)
    #print min_, max_, '\n', bins

    for k, v in dfs.iteritems():
        if args.maxz:
            trimmed_data = v[var][(np.abs(stats.zscore(v[var])) < max_zscore)]
            trimmed_data.plot.hist(bins, label=k, histtype='step', density=True)
        else:
            v[var].plot.hist(bins, label=k, histtype='step', density=True)
    
    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=100)
    return

problems = {}

for var in var_names:
    try:
        make_hist(var)
    except Exception as e:
        problems[var] = str(e)
 
#make_hist("dPhi_metjet")
#make_hist('jetplusmet_mass')
#make_hist('jetplusmet_pt')
   
out.close()
