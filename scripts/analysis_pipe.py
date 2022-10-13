import numpy as np
import pandas as pd
import sys, os
import json
import h5py
from pathlib import Path
import argparse
#DEBUG
import shutil
import sys
here_dir = Path(__file__).parent.absolute()
sys.path.append(str(here_dir))
sys.path.append(str(here_dir.parent.absolute()/'galaxybrain'))
sys.path.append(str(here_dir.parent.absolute()/'log_utils'))

from data_utils import MouseData, shuffle_data
import ramsey
from logs import init_log
import logging
import multiprocessing as mp

import warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def run_analysis(output_dir, num_trials, ramsey_kwargs, mouse_kwargs={}, shuffle=False):
    """
    output_dir: assumes you've made expNUM folder
    ramsey_kwargs: dict
    burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb
    shuffle = (axis, num_shuffles)
    data_type: mouse, or ising
    **This function refers to some variables declared below in __main__**
    """
    data_type = ramsey_kwargs.get('data_type', 'mouse')
    def distributed_compute(save_dir):
        """
        save_dir: dir, not including last subdir (e.g., region or temp)
        other vars are side effects
        """
        results = [POOL.apply(ramsey.ramsey, 
                              kwds={'data' :_curr_raster, 
                                     **ramsey_kwargs}) 
                        for _curr_raster in parallel_args]
        if shuffle:
            for i in np.arange(0,len(results), num_trials):
                region_or_temp, s = parallel_labels[i][0], parallel_labels[i][1]
                curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
                np.savez(f'{save_dir}/{region_or_temp}/{s+1}', eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
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
                np.savez(f'{save_dir}/{region_or_temp}/{tn+1}', **curr_output)
        return


    if data_type == 'mouse':
        mice_data = MouseData(**mouse_kwargs)
        for mouse_name in mice_data.mouse_in:
            ### TODO : maybe move this outside and append mouse names to labels ###
            parallel_args = [] # keep track of indices
            parallel_labels = [] # for going through results and saving data later
            for mouse_raster, region_name in mice_data.mouse_iter(mouse_name):
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
    #DEBUG
    elif data_type == 'ising':
        ising_h5 = h5py.File(str(here_dir/'../data/spikes/ising.hdf5'), 'r')
        parallel_args = [] # keep track of indices
        parallel_labels = [] # for going through results and saving data later
        logging.info(list(ising_h5.keys()))
        for temp in list(ising_h5.keys()): #[6:]: #keys are str # NOTE: power chans broken for indices 0...5
            # DEBUG (rmtree not safe)
            try:
                os.makedirs(f'{output_dir}/{temp}')
            except FileExistsError:  
                shutil.rmtree(f'{output_dir}/{temp}')
                
            tensor = np.array(ising_h5[temp])
            raster = pd.DataFrame(tensor.reshape(tensor.shape[0], -1)) # shape := (Time x N^2)
            if shuffle:
                for s in range(shuffle[1]):
                    curr_raster = shuffle_data(raster, shuffle[0]) 
                    [parallel_args.append(curr_raster) for n in range(num_trials)]
                    [parallel_labels.append((temp, s)) for n in range(num_trials)]
            else:
                [parallel_args.append(raster) for n in range(num_trials)]
                [parallel_labels.append((temp, n)) for n in range(num_trials)]
            
        distributed_compute(save_dir=output_dir)

            
if __name__ == '__main__':
    DEBUG = True

    init_log()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mouse', action='store_true')
    parser.add_argument('-t', dest='test',  action='store_true') # test mouse
    parser.add_argument('-i', dest='ising', action='store_true')
    cl_args = parser.parse_args()

    if DEBUG:
        cl_args.ising = True
        NUM_CORES = 1    # DEBUG
    else:
        NUM_CORES = 20
    #Parallel stuff
    # There are 28 cores
    POOL = mp.Pool(NUM_CORES)
    if cl_args.test:
        analysis_args = {'output_dir' :  str(here_dir/'../data/experiments/TEST'),
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
                        'num_trials' : 1,
                        }
    elif cl_args.mouse:
        analysis_args = {'output_dir' :  str(here_dir/'../data/experiments/mouse'),
                        'mouse_kwargs': {'mouse_in'  : ['robbins', 'waksman']},
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
        analysis_args={'output_dir' : str(here_dir/'../data/experiments/ising_better_fit'),
                       'ramsey_kwargs' : {'data_type': 'ising',
                                          'n_iter' : 95,
                                          'n_pc' : 0.8,
                                          'pc_range': [0,0.01],
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
                                                                    'fit_kwargs': {'aperiodic_mode': 'knee'}
                                                            },
                                                        }
                                         },
                        'num_trials' : 4,
                        }
    
    with open(f"{analysis_args['output_dir']}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)

    run_analysis(**analysis_args)
    POOL.close()
