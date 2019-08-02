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
parser.add_argument('--maxz', type=int, nargs='?', action='store', help='filter data with zscores above this value')
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

if args.maxz:
    max_zscore = args.maxz

#parsing the json file
with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    base_dir = payload['base_dir']
    filenames = payload['filenames']
    per_part = payload['per_part']
    combine_particles = payload['combine_particles']
    if per_part:
        cut_vars = payload['cut_vars']
        cuts = payload['particle_cuts']
    else:
        cut = payload['jet_cut']

#reading data from the .pkl files and applying selections
dfs = {}

for k, v in filenames.iteritems():
    dfs[k] = pd.read_pickle(base_dir+v+".pkl")

var_names = list(dfs[list(dfs)[0]])

def apply_particle_cuts():
    if cuts and cut_vars:
        if len(cuts) != len(cut_vars): raise ValueError("Different number of cuts and cut_vars")
        #going from 'fj_cpf_pfType' to ['fj_cpf_pfType[0]', ...]
        all_cut_vars = [[spec_var for spec_var in var_names if gen_var in spec_var] for gen_var in cut_vars]
        #all_cut_vars = [var+"[0]" for var in cut_vars]
        #print "all_cut_vars: ", all_cut_vars
        for k, df in dfs.iteritems():
            print "df.shape before cuts: ", df.shape
            for i in range(len(cuts)):
                for var in all_cut_vars[i]:
                    #print cuts[i].format(var)
                    df = df[eval(cuts[i].format(var))]
            print "df.shape after cuts: ", df.shape
            dfs[k] = df

def combine_particle_columns():
    nparticles = len([var for var in var_names if var_names[0][:-3] in var])
    gen_var_names = [var_names[i][:-3] for i in range(0, len(var_names), nparticles)]
    #print gen_var_names
    for k, df in dfs.iteritems():
        #print "\ncombine_particle_columns: df.head before changes\n", df.head(), df.shape, '\n'
        condensed_data = []
        for var in gen_var_names:
            columns = [v for v in var_names if var in v]
            #print "combine_particle_columns: current var, columns", var, columns
            combined = pd.concat([df[col] for col in columns], axis=0)
            condensed_data.append(combined)
        df = pd.concat(condensed_data, axis=1, keys=gen_var_names)
        #print "combine_particle_columns: df.head after changes\n", df.head(), df.shape, '\n'
        dfs[k] = df

if per_part:
    apply_particle_cuts()
    if combine_particles:
        combine_particle_columns()
else:
    if cut:
        for df in dfs.itervalues():
            df = df[eval(cut)]

var_names = list(dfs[list(dfs)[0]])

# functions to make individual plots
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

    for k, v in dfs.iteritems():
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

    plt.legend(loc='upper right')

    PdfPages.savefig(out, dpi=dpi)
    return

# functions to make plots for a given list of vars
def make_hists(vars_to_plot):
    problems = {}
    for var in vars_to_plot:
        try:
            make_hist(var)
        except Exception as e:
            problems[var] = str(e)
    return problems

#make_hists(var_names)
#make_hist('fj_cpf_pfType')

#make_plot('fj_cpf_pfType[0]', 'fj_cpf_dz[0]')

kinematics = ['fj_cpf_pt', 'fj_cpf_eta', 'fj_cpf_phi', 'fj_cpf_dz', 'fj_cpf_pup', 'fj_cpf_q']
old_kinematics = ['fj_cpf_pt[0]', 'fj_cpf_eta[0]', 'fj_cpf_phi[0]', 'fj_cpf_dz[0]', 'fj_cpf_pup[0]', 'fj_cpf_q[0]']
make_hists(kinematics)

out.close()
