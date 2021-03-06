from re import T
import numpy as np
import pandas as pd
import sys, os
import json
sys.path.append('../')
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import ramsey
from ising import metro_ising
from analysis_pipe import shuffle_data

import multiprocessing as mp

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
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

def run_analysis(output_dir, num_trials, ramsey_params, N, ising_time, temps, burn_in = 20, shuffle = False, parallel=True):
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
        sim_mouse = metro_ising(N=N,T=t, plot=False, runtime=ising_time)
        sim_slice = pd.DataFrame(sim_mouse[200:][:,325]) #this is the data analyzed
        subsetsizes = np.linspace(30,N-10,16, dtype=int) # using N-10 to accomodate PCA error

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
        results = [pool.apply(ramsey.ramsey, args = (_curr_raster, _subsetsizes, n_iters, n_pc, f_range)) for (_curr_raster,_subsetsizes) in parallel_args]
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

    analysis_args={'output_dir' : '../../../../projects/ps-voyteklab/brirry/data/experiments/expTESTSIM',
                    'temps' : n_from_x(2.26918531421, 3, 0.1),
                    'ising_time': 1000,
                    'N' : 650,
                    'ramsey_params' : {'n_iters' : 95, 'n_pc' : 0.8, 'f_range' : [0,0.4]},
                    'num_trials' : 5}
    
    run_analysis(**analysis_args)

    with open(f"{analysis_args['output_dir']}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)


