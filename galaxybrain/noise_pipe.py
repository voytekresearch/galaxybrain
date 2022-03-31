import numpy as np
import pandas as pd
import sys, os
import json
sys.path.append('../')
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import ramsey

import multiprocessing as mp

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#Parallel stuff
cores = mp.cpu_count()
pool = mp.Pool(8)

def run_analysis(output_dir, num_trials, ramsey_params, time, burn_in = 20, shuffle = False, parallel=True):
    """
    Gaussian noise simulation
    
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

    
    sim_mouse = np.array([np.random.normal(size=time) for i in range(1462)]).T # Time x Units
    sim_slice = pd.DataFrame(sim_mouse[burn_in:-burn_in]) #this is the data analyzed
    subsetsizes = np.linspace(30,1462,16, dtype=int) # using N-10 to accomodate PCA error

    [parallel_args.append((sim_slice,subsetsizes)) for n in range(num_trials)]
    [parallel_labels.append(n) for n in range(num_trials)]    

    results = [pool.apply(ramsey.ramsey, args = (_curr_raster, _subsetsizes, n_iters, n_pc, f_range)) for (_curr_raster,_subsetsizes) in parallel_args]
    pool.close()
    if shuffle:
        for i in np.arange(0,len(results), num_trials):
            s = parallel_labels[i]
            curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
            np.savez(f'{output_dir}/ramsey_{s+1}', eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
                                                    pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
                                                    pca_m=curr_output[:,2].mean(0), pca_b=curr_output[:,10].mean(0), space_er=curr_output[:,3].mean(0), 
                                                    ft_m=curr_output[:,4].mean(0), ft_b=curr_output[:,11].mean(0), time_er=curr_output[:,5].mean(0), 
                                                    pearson_r=curr_output[:,6].mean(0), spearman_rho=curr_output[:,7].mean(0), 
                                                    pearson_p=curr_output[:,8].mean(0), spearman_p=curr_output[:,9].mean(0),
                                                    curr_pc_range=curr_output[:,12][0]) # saving the first because they are all identical

    else:
        for i in range(len(results)):
            tn = parallel_labels[i]
            curr_output = np.array(results[i])
            np.savez(f'{output_dir}/ramsey_{tn+1}', eigs=curr_output[0], pows=curr_output[1], 
                                                    pca_m=curr_output[2], pca_b=curr_output[10], space_er=curr_output[3], 
                                                    ft_m=curr_output[4],ft_b=curr_output[11], time_er=curr_output[5], 
                                                    pearson_r=curr_output[6], spearman_rho=curr_output[7], 
                                                    pearson_p=curr_output[8], spearman_p=curr_output[9],
                                                    curr_pc_range=curr_output[12])
### SCRIPT ###
if __name__ == '__main__':

    analysis_args={'output_dir' : '../../../../projects/ps-voyteklab/brirry/data/experiments/NOISESIM',
                    'time': 1301,
                    'ramsey_params' : {'n_iters' : 95, 'n_pc' : 0.8, 'f_range' : [0,0.4]},
                    'num_trials' : 5}
    
    run_analysis(**analysis_args)

    with open(f"{analysis_args['output_dir']}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)


