import numpy as np
import sys, os
import json
from pathlib import Path

here = Path(__file__)
sys.path.append(str(here.parent.absolute()))
sys.path.append(str(here.parent.absolute().parent.absolute()/'galaxybrain'))

from data_utils import MouseData
import ramsey

import multiprocessing as mp

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def shuffle_data(data, axis): 
    """Helper function to shuffle data"""
    if axis == 'time':
        t_ind = np.arange(data.shape[0]) 
        np.random.shuffle(t_ind)
        raster_curr = data.iloc[t_ind]
    elif axis == 'space':
        raster_curr = data.apply(np.random.permutation, axis=1, result_type='broadcast') #need broadcast to maintain shape
    return raster_curr


def run_analysis(output_dir, mice_regions, num_trials, ramsey_kwargs, mouse_kwargs, shuffle=False, parallel=True):
    """
    output_dir: assumes you've made expNUM folder
    ramsey_kwargs: dict
    burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb
    shuffle = (axis, num_shuffles)
    
    **This function refers to some variables declared below in __main__**
    """
    #var housekeeping
    n_iters = ramsey_kwargs['n_iters']
    n_pc = ramsey_kwargs['n_pc']
    f_range = ramsey_kwargs['f_range']
    
    #analysis + saving data 

    mice_data = MouseData(**mouse_kwargs)
    for mouse_name in mice_data.mouse_in:
        ### TODO : maybe move this outside and append mouse names to labels ###
        parallel_args = [] # keep track of indices
        parallel_labels = [] # for going through results and saving data later
        for mouse_raster, region_name, region_count in mice_data.mouse_iter(mouse_name):
            
            os.makedirs(f'{output_dir}/{mouse_name}/{region_name}')

            subsetsizes = np.linspace(30,region_count,16, dtype=int)
            
            if shuffle:
                for s in range(shuffle[1]):
                    curr_raster = shuffle_data(mouse_raster, shuffle[0]) 
                    #curr_output = {'eigs':[],'pows':[],'pca_m':[],'s_er':[],'ft_m':[],'t_er':[],'psn_r':[], 'spn_r':[], 'psn_p':[], 'spn_p':[]}

                    [parallel_args.append((curr_raster,subsetsizes)) for n in range(num_trials)]
                    [parallel_labels.append((region_name, s)) for n in range(num_trials)]
                        
                    
            else:
                [parallel_args.append((mouse_raster,subsetsizes)) for n in range(num_trials)]
                [parallel_labels.append((region_name, n)) for n in range(num_trials)]
                
            
            
        if parallel:
            results = [POOL.apply(ramsey.ramsey, args=(_curr_raster, _subsetsizes, n_iters, n_pc, f_range)) for (_curr_raster,_subsetsizes) in parallel_args]
            POOL.close()
            if shuffle:
                for i in np.arange(0,len(results), num_trials):
                    region_name, s = parallel_labels[i][0], parallel_labels[i][1]
                    curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
                    np.savez(f'{output_dir}/{mouse_name}/{region_name}/ramsey_{s+1}',eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
                                                                                    pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
                                                                                    pca_m=curr_output[:,2].mean(0), pca_er=curr_output[:,3].mean(0), pca_b=curr_output[:,4].mean(0), 
                                                                                    ft_m1=curr_output[:,5].mean(0), ft_er1=curr_output[:,6].mean(0), ft_b1=curr_output[:,7].mean(0), 
                                                                                    ft_m2=curr_output[:,8].mean(0), ft_er2=curr_output[:,9].mean(0), ft_b2=curr_output[:,10].mean(0), 
                                                                                    pearson_r1=curr_output[:,11].mean(0), spearman_rho1=curr_output[:,13].mean(0), 
                                                                                    pearson_p1=curr_output[:,12].mean(0), spearman_p1=curr_output[:,14].mean(0),
                                                                                    pearson_r2=curr_output[:,15].mean(0), spearman_rho2=curr_output[:,17].mean(0), 
                                                                                    pearson_p2=curr_output[:,16].mean(0), spearman_p2=curr_output[:,18].mean(0))
                                                                                     #curr_pc_range=curr_output[:,12][0] # saving the first because they are all identical
                
            else:
                for i in range(len(results)):
                    region_name, tn = parallel_labels[i][0], parallel_labels[i][1]
                    curr_output = np.array(results[i])
                    np.savez(f'{output_dir}/{mouse_name}/{region_name}/ramsey_{tn+1}', eigs=curr_output[0], pows=curr_output[1], 
                                                                                      pca_m=curr_output[2], pca_er=curr_output[3], pca_b=curr_output[4], 
                                                                                      ft_m1=curr_output[5], ft_er1=curr_output[6],ft_b1=curr_output[7], 
                                                                                      ft_m2=curr_output[8], ft_er2=curr_output[9],ft_b2=curr_output[10], 
                                                                                      pearson_r1=curr_output[11], spearman_rho1=curr_output[13], 
                                                                                      pearson_p1=curr_output[12], spearman_p1=curr_output[14],
                                                                                      pearson_r2=curr_output[15], spearman_rho2=curr_output[17], 
                                                                                      pearson_p2=curr_output[16], spearman_p2=curr_output[18])

            
### SCRIPT ###
if __name__ == '__main__':
    #Parallel stuff
    # CORES = mp.cpu_count()
    POOL = mp.Pool(7)
    analysis_args={'output_dir' : '../../../../projects/ps-voyteklab/brirry/data/experiments/04242022',
                    'ramsey_kwargs' : {'n_iters' : 95, 'n_pc' : 0.8, 'f_range' : [0,0.4]},
                    'num_trials' : 4,
                    'mouse_kwargs' : {'mouse_in': ['krebs']},
                    }
    
    run_analysis(**analysis_args)

    analysis_args['mice_regions'] = 'mice_regions_var'

    with open(f"{analysis_args['output_dir']}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)


