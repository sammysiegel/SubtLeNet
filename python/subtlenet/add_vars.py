import numpy as np
import pandas as pd
import argparse
import json
import ROOT

parser = argparse.ArgumentParser()
#might want to add an --out arg in case we want to leave the original .pkl
parser.add_argument('--json', type=str, help='json file pointing to the .pkl files to be edited')
args = parser.parse_args()

with open(args.json) as jsonfile:
    #this script should be able to use the same .json files as plot.py
    payload = json.load(jsonfile)
    base_dir = payload['base_dir']
    filenames = payload['filenames']

dfs = {}

for k, v in filenames.iteritems():
    dfs[k] = pd.read_pickle(base_dir+v+'.pkl')

#dPhi_metjet
def make_dPhi_metjet():
    for k, v in dfs.iteritems():
        v['dPhi_metjet'] = v['pfmetphi'] - v['fj_phi']
    return

#jetplusmet_mass
def calc_jpm_mass(j_pt, j_eta, j_phi, j_mass, met, met_phi):
    jet = ROOT.TLorentzVector(j_pt, j_eta, j_phi, j_mass)
    met = ROOT.TLorentzVector(met, j_eta, met_phi, 0)
    jpm = jet+met
    return jpm.M()

def make_jetplusmet_mass(j_pt='fj_pt', j_eta='fj_eta', j_phi='fj_phi', j_mass='fj_mass', met='pfmet', met_phi='pfmetphi'):
    for k, v in dfs.iteritems():
        v['jetplusmet_mass'] = np.vectorize(calc_jpm_mass)(v[j_pt], v[j_eta], v[j_phi], v[j_mass], v[met], v[met_phi])
        print 'fj_mass:\n', v[j_mass][:5]
        print 'jetplusmet_mass:\n', v['jetplusmet_mass'][:5]

#jetplusmet_pt
def calc_jpm_pt(j_pt, j_eta, j_phi, j_mass, met, met_phi):
    jet = ROOT.TLorentzVector(j_pt, j_eta, j_phi, j_mass)
    met = ROOT.TLorentzVector(met, j_eta, met_phi, 0)
    jpm = jet+met
    return jpm.Pt()

def make_jetplusmet_pt(j_pt='fj_pt', j_eta='fj_eta', j_phi='fj_phi', j_mass='fj_mass', met='pfmet', met_phi='pfmetphi'):
    for k, v in dfs.iteritems():
        v['jetplusmet_pt'] = np.vectorize(calc_jpm_pt)(v[j_pt], v[j_eta], v[j_phi], v[j_mass], v[met], v[met_phi])

def make_jpm_vars():
    make_jetplusmet_mass()
    make_jetplusmet_pt()

def save():
    for k, v in dfs.iteritems():
        v.to_pickle(base_dir+filenames[k]+'.pkl')
    return

make_dPhi_metjet()
#make_jpm_vars()
save()





