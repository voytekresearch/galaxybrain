from unittest.util import strclass
import numpy as np
import sys, os
import json
import h5py
from pathlib import Path
import argparse

here_dir = Path(__file__).parent.absolute()
sys.path.append(str(here_dir))
sys.path.append(str(here_dir.parent.absolute()/'galaxybrain'))

from data_utils import MouseData, shuffle_data
import ramsey

import multiprocessing as mp

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def run_analysis(output_dir, num_trials, ramsey_kwargs, mouse_kwargs={}, data_type='mouse', shuffle=False):
    """
    output_dir: assumes you've made expNUM folder
    ramsey_kwargs: dict
    burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb
    shuffle = (axis, num_shuffles)
    data_type: mouse, or ising
    **This function refers to some variables declared below in __main__**
    """
    
    def distributed_compute(save_dir):
        """
        save_dir: dir, not including last subdir (e.g., region or temp)
        other vars are side effects
        """
        results = [POOL.apply(ramsey.ramsey, 
                              kwds={'data' :_curr_raster, 
                                     **ramsey_kwargs}) 
                        for _curr_raster in parallel_args]
        POOL.close()
        if shuffle:
            for i in np.arange(0,len(results), num_trials):
                region_or_temp, s = parallel_labels[i][0], parallel_labels[i][1]
                curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
                np.savez(f'{save_dir}/{region_or_temp}/ramsey_{s+1}', eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
                                                                    pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
                                                                    pca_m=curr_output[:,2].mean(0), pca_er=curr_output[:,3].mean(0), pca_b=curr_output[:,4].mean(0), 
                                                                    ft_m1=curr_output[:,5].mean(0), ft_er1=curr_output[:,6].mean(0), ft_b1=curr_output[:,7].mean(0), 
                                                                    ft_m2=curr_output[:,8].mean(0), ft_er2=curr_output[:,9].mean(0), ft_b2=curr_output[:,10].mean(0), 
                                                                    pearson_r1=curr_output[:,11].mean(0), spearman_rho1=curr_output[:,13].mean(0), 
                                                                    pearson_p1=curr_output[:,12].mean(0), spearman_p1=curr_output[:,14].mean(0),
                                                                    pearson_r2=curr_output[:,15].mean(0), spearman_rho2=curr_output[:,17].mean(0), 
                                                                    pearson_p2=curr_output[:,16].mean(0), spearman_p2=curr_output[:,18].mean(0))
            
        else:
            for i in range(len(results)):
                region_or_temp, tn = parallel_labels[i][0], parallel_labels[i][1]
                curr_output = results[i]
                np.savez(f'{save_dir}/{region_or_temp}/ramsey_{tn+1}', **curr_output)

    if data_type == 'mouse':
        mice_data = MouseData(**mouse_kwargs)
        for mouse_name in mice_data.mouse_in:
            ### TODO : maybe move this outside and append mouse names to labels ###
            parallel_args = [] # keep track of indices
            parallel_labels = [] # for going through results and saving data later
            for mouse_raster, region_name in mice_data.mouse_iter(mouse_name):
                if not os.path.exists(f'{output_dir}/{mouse_name}/{region_name}'):
                    os.makedirs(f'{output_dir}/{mouse_name}/{region_name}')

                if shuffle:
                    for s in range(shuffle[1]):
                        curr_raster = shuffle_data(mouse_raster, shuffle[0]) 
                        [parallel_args.append(curr_raster) for n in range(num_trials)]
                        [parallel_labels.append((region_name, s)) for n in range(num_trials)]
                else:
                    [parallel_args.append(mouse_raster) for n in range(num_trials)]
                    [parallel_labels.append((region_name, n)) for n in range(num_trials)]
                    
            distributed_compute(save_dir=f'{output_dir}/{mouse_name}')
    
    elif data_type == 'ising':
        ising_h5 = h5py.File(str(here_dir/'../data/spikes/ising.hdf5'), 'r')
        parallel_args = [] # keep track of indices
        parallel_labels = [] # for going through results and saving data later
        for temp in ising_h5: #keys are str!
            os.makedirs(f'{output_dir}/{temp}')
            tensor = np.array(ising_h5[temp]).reshape(tensor.shape[0], -1)
            raster = tensor.reshape(tensor.shape[0], -1) # shape := (Time x N^2)
            os.makedirs(f'{output_dir}/{temp}')

            if shuffle:
                for s in range(shuffle[1]):
                    curr_raster = shuffle_data(raster, shuffle[0]) 
                    [parallel_args.append(curr_raster) for n in range(num_trials)]
                    [parallel_labels.append((region_name, s)) for n in range(num_trials)]
            else:
                [parallel_args.append(raster) for n in range(num_trials)]
                [parallel_labels.append((region_name, n)) for n in range(num_trials)]
                    
            distributed_compute(save_dir=output_dir)

            
### SCRIPT ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mouse', action='store_true')
    parser.add_argument('-t', dest='test', action='store_true')
    parser.add_argument('-i', dest='ising', action='store_true')
    cl_args = parser.parse_args()
    #Parallel stuff
    # There are 28 cores
    POOL = mp.Pool(8)
    if cl_args.test:
        analysis_args = {'output_dir' :  '/Users/brianbarry/Desktop/computing/personal/galaxybrain/data/experiments/TEST',#'../../../../projects/ps-voyteklab/brirry/data/experiments/04242022',
                        'mouse_kwargs': {'phantom':True},
                        'ramsey_kwargs' : {
                                            'n_iter': 2, 
                                            'n_pc': 0.8, 
                                            'pc_range': [0, None],
                                            'f_range': [0,0.4],
                                            'ft_kwargs': {
                                                            'fs'      : 1,
                                                            'nperseg' : 120,
                                                            'noverlap': 120/2
                                                        }
                                        },
                        'num_trials' : 4,
                        }
    elif cl_args.mouse:
        analysis_args = {'output_dir' :  '/Users/brianbarry/Desktop/computing/personal/galaxybrain/data/experiments/TEST',#'../../../../projects/ps-voyteklab/brirry/data/experiments/04242022',
                            'ramsey_kwargs' : {
                                                'n_iter': 95, 
                                                'n_pc': 0.8, 
                                                'pc_range': [0, None],
                                                'f_range': [0,0.4],
                                                'ft_kwargs': {
                                                                'fs'      : 1,
                                                                'nperseg' : 120,
                                                                'noverlap': 120/2
                                                            }
                                            },
                            'num_trials' : 4,
                            }
    elif cl_args.ising:
        analysis_args={'output_dir' : '/home/brirry/galaxybrain/data/experiments/ising_better_fit',
                        'data_type': 'ising',
                       'ramsey_kwargs' : {'n_iters' : 95,
                                          'n_pc' : 0.8,
                                          'pc_range': [0,0.1],
                                          'f_range' : [0,0.01],
                                          'ft_kwargs': {
                                                        'fs': 1,
                                                        'nperseg': 2000,
                                                        'noverlap': int(.8*2000)
                                                    },
                                          'fooof_kwargs': {
                                                            'es': {'return_params': [['aperiodic_params', 'exponent'],
                                                                                    ['aperiodic_params', 'knee'],
                                                                                    ['error'], # MAE
                                                                                    ['aperiodic_params', 'offset']],
                                                            'fit_kwargs': {'aperiodic_mode': 'knee'}},
                                                        }
                                         },
                        'num_trials' : 5,
                        }
    
    run_analysis(**analysis_args)

    with open(f"{analysis_args['output_dir']}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)


