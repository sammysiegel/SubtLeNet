import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import argparse
import json
import os

#grabbing command line args
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='filename of output pdf (the .pdf extension is optional)')
parser.add_argument('--json', type=str, help='json file controlling the input files used')
parser.add_argument('--maxz', help='data with zscores above this value will not be plotted', action='store_true')
parser.add_argument('--dpi', type=int, nargs='?', action='store', help='dpi for output pdf')
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

dpi = 100
if args.dpi:
    dpi = args.dpi

#parsing the json file
with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    base_dir = payload['base_dir']
    filenames = payload['filenames']
    max_zscore = int(payload['maxz'])
    per_part = payload['per_part']
    if per_part:
        cut_vars = payload['cut_vars']
        cuts = payload['particle_cuts']
    else:
        cut = payload['jet_cut']

#making the dataframes and applying selections
dfs = {}

for k, v in filenames.iteritems():
    dfs[k] = pd.read_pickle(base_dir+v+".pkl")

var_names = list(dfs[list(dfs)[0]])

if per_part:
    if cuts and cut_vars:
        if len(cuts) != len(cut_vars): raise ValueError("Different number of cuts and cut_vars")
        #going from 'fj_cpf_pfType' to ['fj_cpf_pfType[0]', ...]
        all_cut_vars = [[spec_var for spec_var in var_names if gen_var in spec_var] for gen_var in cut_vars]
        for k, df in dfs.iteritems():
            print "df.shape before cuts: ", df.shape
            for i in range(len(cuts)):
                for var in all_cut_vars[i]:
                    #print cuts[i].format(var)
                    df = df[eval(cuts[i].format(var))]
            print "df.shape after cuts: ", df.shape
            dfs[k] = df
else:
    if cut:
        for df in dfs.itervalues():
            df = df[eval(cut)]

#print var_names
for k, v in dfs.iteritems():
    print v.shape

def make_hist(var):
    #Plots a histogram comparing var across all dataframes
    plt.figure(figsize=(4, 4), dpi=dpi)
    plt.xlabel(var)
    plt.title(var)
    
    min_ = min([v[var].min() for v in dfs.itervalues()])
    max_ = max([v[var].max() for v in dfs.itervalues()])

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

    for v in dfs.itervalues():
        if args.maxz:
            trimmed_data = v[var][(np.abs(stats.zscore(v[var])) < max_zscore)]
            trimmed_data.plot.hist(bins, label=k, histtype='step', density=True)
        else:
            v[var].plot.hist(bins, label=k, histtype='step', density=True)
    
    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=dpi)
    return

def make_plot(xvar, yvar):
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.title(xvar+" vs. "+yvar)

    for df in dfs.itervalues():
        if args.maxz:
            df = df[np.abs(stats.zscore(df[xvar])) < max_zscore]

        x = df[xvar]
        y = df[yvar]
       
        kwargs = {'ls': 'None'}
        
        plt.plot(x, y, 'bo', **kwargs)

    #plt.legend(loc='upper right')

    PdfPages.savefig(out, dpi=dpi)
    return



def make_hists(vars_to_plot):
    problems = {}
    for var in vars_to_plot:
        try:
            make_hist(var)
        except Exception as e:
            problems[var] = str(e)
    return problems

#make_hists(var_names) 
#make_hist("dPhi_metjet")
#make_hist('jetplusmet_mass')
#make_hist('jetplusmet_pt')

#make_plot('fj_cpf_pfType[0]', 'fj_cpf_dz[0]')

kinematics = ['fj_cpf_pt[0]', 'fj_cpf_eta[0]', 'fj_cpf_phi[0]', 'fj_cpf_dz[0]', 'fj_cpf_pup[0]', 'fj_cpf_q[0]']
make_hists(kinematics)


out.close()
