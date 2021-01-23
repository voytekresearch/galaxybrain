import numpy as np
from scipy import io, signal, stats
import sys, os
sys.path.append('../')

import pandas as pd
from sklearn.decomposition import PCA
import fooof
from fooof import FOOOFGroup
from seaborn import despine

from galaxybrain.data_utils import load_mouse_data, return_pops
from galaxybrain import ramsey
from neurodsp.spectral import compute_spectrum

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# load mice
datafolder = '../../galaxybrain_data/'

#load mouse data

all_mice = []
for i_m in range(3):
    print('Mouse ' + (str(i_m+1)))
    all_mice.append([])
    df_spk, df_info = load_mouse_data(datafolder, i_m, return_type='binned', bin_width=1)
    region_indices = {}
    for region in df_info.region.unique():
        region_indices.update({region:np.where(df_info['region'] == str(region))[0]})

    spk_list, region_labels = return_pops(df_spk, df_info)
    print(list(zip(region_labels, [s.shape[1] for s in spk_list])), 'Total:',sum([s.shape[1] for s in spk_list]))

    su_start_ind = len(region_labels)+1

    all_mice[i_m].append(df_spk[df_spk.columns[su_start_ind:]])
    all_mice[i_m].append(region_indices)

krebs = all_mice[0]; robbins = all_mice[1]; waksman = all_mice[2]

mice_regions = {'krebs': [krebs, [('all', 1462), ('CP', 176), ('HPF', 265), ('LS', 122), ('MB', 127), ('TH', 227), ('V1', 334)]],
               'robbins': [robbins, [('all', 2688), ('FrMoCtx', 647), ('HPF', 333), ('LS', 133), ('RSP', 112), ('SomMoCtx', 220), ('TH', 638), ('V1', 251), ('V2', 124)]],
               'waksman': [waksman, [('all', 2296), ('CP', 134), ('HPF', 155), ('TH', 1878)]] }

burn_in = 20 #skipping beginning and end of recordings (see DataDiagnostic.ipynb)
#analysis + saving data
shuffle = 'time'  
for mouse_key in ['krebs', 'robbins', 'waksman']:
    mouse = mice_regions[mouse_key][0] #this is the data
    for region in mice_regions[mouse_key][1]:

        region_name = region[0]; region_count = region[1]
        os.makedirs(f'../data/experiments/exp7/{mouse_key}/{region_name}')

        if region_name == 'all':
            mouse_raster = mouse[0].iloc[burn_in:-burn_in]
        else:
            mouse_raster = mouse[0][mouse[1][region_name]].iloc[burn_in:-burn_in] #mouse[1] are indices

        subsetsizes = np.linspace(30,region_count,16, dtype=int)

        for i in range(4):
            eigs, pows, pca_m, s_r2, ft_m, t_r2, psn_r, spn_r, psn_p, spn_p = ramsey.ramsey(mouse_raster, subsetsizes, n_iters = 95, n_pc = 0.8, f_range = [0,0.4], shuffle = shuffle, verbose = False)
            np.savez(f'../data/experiments/exp7/{mouse_key}/{region_name}/ramsey_{i+1}', eigs=eigs, pows=pows, pca_m=pca_m, space_r2=s_r2, ft_m=ft_m, time_r2=t_r2, pearson_r=psn_r, spearman_rho=spn_r, pearson_p=psn_p, spearman_p=spn_p)
