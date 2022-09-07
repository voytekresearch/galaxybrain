import numpy as np
import pandas as pd
import sys, os
import json
import h5py
import argparse
from pathlib import Path

here = Path(__file__)
sys.path.append(str(here.parent.absolute()))
sys.path.append(str(here.parent.absolute().parent.absolute()/'galaxybrain'))

import ramsey
from ising import metro_ising, T_CRIT
from analysis_pipe import shuffle_data

import multiprocessing as mp

import warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#Parallel stuff
cores = mp.cpu_count()
pool = mp.Pool(8)


def n_from_x(x,n,stp):
    """like np.arange except starting from the middle
    x: start, n: num in either direction, stp: step size"""
    r = [x]
    for i in range(n):
        nxt_p, nxt_n = x + (i+1) * stp, x - (i+1) * stp
        r.append(nxt_p)
        r.insert(0,nxt_n)
    return r


def sim_and_save(path, temps, **ising_kwargs):
    with h5py.File(f'{path}/ising.hdf5', 'a') as f:
        for temp in temps:
            data = metro_ising(T=temp, **ising_kwargs)
            f.create_dataset(f'{temp:.2f}', data=data, dtype='i')


def run_analysis(output_dir, num_trials, ramsey_params, temps, burn_in=20, shuffle = False, parallel=True, **ising_kwargs):
    """
    output_dir: assumes you've made expNUM folder
    ramsey_params: params besides data and subsetsizes
    burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb
    shuffle = (axis, num_shuffles)
    
    **This function refers to some variables declared below in __main__**
    """
    #var housekeeping
    n_iters = ramsey_params['n_iters']
    n_pc = ramsey_params['n_pc']
    f_range = ramsey_params['f_range']
    #analysis + saving data  
    parallel_args = [] #just need to keep track of indices
    parallel_labels = [] # for going through results and saving data later
    for t in temps:
        os.makedirs(f'{output_dir}/{t:.2f}')
        sim_mouse = h5py.File('test.hdf5', 'r')
        sim_slice = pd.DataFrame(sim_mouse[200:][:,325]) #this is the data analyzed
        subsetsizes = np.linspace(30, ising_kwargs['N']-10, 16, dtype=int) # using N-10 to accomodate PCA error
        if shuffle:
            for s in range(shuffle[1]):
                curr_raster = shuffle_data(sim_slice, shuffle[0]) 
                #curr_output = {'eigs':[],'pows':[],'pca_m':[],'s_er':[],'ft_m':[],'t_er':[],'psn_r':[], 'spn_r':[], 'psn_p':[], 'spn_p':[]}

                if parallel:
                    [parallel_args.append((curr_raster,subsetsizes)) for n in range(num_trials)]
                    [parallel_labels.append((t, s)) for n in range(num_trials)]
                    
                else:
                    for n in range(num_trials):
                        eigs, pows, pca_m, s_er, ft_m, t_er, psn_r, spn_r, psn_p, spn_p = ramsey.ramsey(curr_raster, subsetsizes, **ramsey_params)
                        curr_output.append([eigs, pows, pca_m, s_er, ft_m, t_er, psn_r, spn_r, psn_p, spn_p])
                    
                    # AVG ACROSS TRIALS HERE
                    curr_output = np.array(curr_output)
                    np.savez(f'{output_dir}/{t:.2f}/ramsey_{s+1}',eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
                                                                                    pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
                                                                                    pca_m=curr_output[:,2].mean(0), space_er=curr_output[:,3].mean(0), 
                                                                                    ft_m=curr_output[:,4].mean(0), time_er=curr_output[:,5].mean(0), 
                                                                                    pearson_r=curr_output[:,6].mean(0), spearman_rho=curr_output[:,7].mean(0), 
                                                                                    pearson_p=curr_output[:,8].mean(0), spearman_p=curr_output[:,9].mean(0))
    
        else:
            if parallel:
                [parallel_args.append((sim_slice,subsetsizes)) for n in range(num_trials)]
                [parallel_labels.append((t, n)) for n in range(num_trials)]
            else:
                for i in range(num_trials):
                    eigs, pows, pca_m, s_er, ft_m, t_er, psn_r, spn_r, psn_p, spn_p = ramsey.ramsey(mouse_raster, subsetsizes, **ramsey_params)
                    np.savez(f'{output_dir}/{t:.2f}/ramsey_{i+1}', eigs=eigs, pows=pows, pca_m=pca_m, space_er=s_er, ft_m=ft_m, time_er=t_er, pearson_r=psn_r, spearman_rho=spn_r, pearson_p=psn_p, spearman_p=spn_p)
        
    if parallel:
        results = [pool.apply(ramsey.ramsey, args=(_curr_raster, _subsetsizes, n_iters, n_pc, f_range)) for (_curr_raster,_subsetsizes) in parallel_args]
        pool.close()
        if shuffle:
            for i in np.arange(0,len(results), num_trials):
                temp, s = parallel_labels[i][0], parallel_labels[i][1]
                curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
                np.savez(f'{output_dir}/{temp:.2f}/ramsey_{s+1}',eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
                                                                                pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
                                                                                pca_m=curr_output[:,2].mean(0), space_er=curr_output[:,3].mean(0), 
                                                                                ft_m=curr_output[:,4].mean(0), time_er=curr_output[:,5].mean(0), 
                                                                                pearson_r=curr_output[:,6].mean(0), spearman_rho=curr_output[:,7].mean(0), 
                                                                                pearson_p=curr_output[:,8].mean(0), spearman_p=curr_output[:,9].mean(0))
        else:
            for i in range(len(results)):
                temp, tn = parallel_labels[i][0], parallel_labels[i][1]
                curr_output = np.array(results[i])
                np.savez(f'{output_dir}/{temp:.2f}/ramsey_{tn+1}', eigs=curr_output[0], pows=curr_output[1], 
                                                                                    pca_m=curr_output[2], space_er=curr_output[3], 
                                                                                    ft_m=curr_output[4], time_er=curr_output[5], 
                                                                                    pearson_r=curr_output[6], spearman_rho=curr_output[7], pearson_p=curr_output[8], spearman_p=curr_output[9])

### SCRIPT ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='simulate', action='store_true')
    parser.add_argument('-a', dest='analyze', action='store_true')
    args = parser.parse_args()
    # temps = n_from_x(T_CRIT, 3, 0.1)
    temps = np.sort(np.append(np.linspace(0.01, 5, 20), T_CRIT))

    OUT_PATH = '/home/brirry/galaxybrain/data/experiments/ising_better_fit'  
    ising_args = {'runtime':10000,
                  'N' : 64}
    analysis_args={'output_dir' : OUT_PATH,
                    'temps' : temps,
                    'ramsey_params' : {'n_iters' : 95, 'n_pc' : 0.8, 'pc_range': [0,0.1], 'f_range' : [0,0.4]},
                    'num_trials' : 5,
                    'ft_kwargs': {
                        'fs': 1,
                        'nperseg': 2000,
                        'noverlap': int(.8*2000)
                    }}
    if args.simulate:
        sim_and_save(OUT_PATH, temps, **ising_args)
    if args.analyze:
        run_analysis(**analysis_args, **ising_args)

    with open(f"{OUT_PATH}/ising_analysis_args.json",'w') as f:
        json.dump({**ising_args, **analysis_args}, f, indent=1)


