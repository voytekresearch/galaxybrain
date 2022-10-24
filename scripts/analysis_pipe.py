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
sys.path.append(str(here_dir.parent.absolute()/'ising'))
from ising import tensor_to_raster
from data_utils import MouseData, shuffle_data
import ramsey
from logs import init_log
import logging
import concurrent.futures
from memory_profiler import profile #DEBUG
import warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#@profile #DEBUG
def run_analysis(output_dir, num_trials, ramsey_kwargs, mouse_kwargs={}, shuffle=False):
    """
    ramsey_kwargs: dict
    burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb)
    shuffle = (axis, num_shuffles)
    data_type: mouse, or ising
    """

    data_type = ramsey_kwargs.get('data_type', 'mouse')
    parallel_args = [] # keep track of indices
    parallel_labels = [] # for going through results and saving data later
    if data_type == 'mouse':
        mice_data = MouseData(**mouse_kwargs)
        labels = mice_data.get_labels() # labels of form (mouse_name, region_name)
        get_function = lambda label: mice_data.get_spikes(label=label)
    elif data_type == 'ising':
        ising_h5 = h5py.File(str(here_dir/'../data/spikes/ising.hdf5'), 'r')
        # labels = list(ising_h5.keys()) # these are str temperatures
        labels = ['2.27']
        get_function = lambda label: tensor_to_raster(ising_h5[label])

    for label in labels:
        os.makedirs(f'{output_dir}/{label}')
        curr_raster = get_function(label)
        if shuffle:
            for s in range(shuffle[1]):
                curr_raster = shuffle_data(curr_raster, shuffle[0]) 
                [parallel_args.append(curr_raster) for n in range(num_trials)]
                [parallel_labels.append((label, s)) for n in range(num_trials)]
        else:
            [parallel_args.append(curr_raster) for n in range(num_trials)]
            [parallel_labels.append((label, n)) for n in range(num_trials)]
            
    results = [] #actually futures if using submit
    for label, _curr_raster in zip(parallel_labels, parallel_args):
        logging.info(label)
        results.append(EXECUTOR.submit(ramsey.ramsey, **{'data' :_curr_raster, **ramsey_kwargs} ))
        # results.append(ramsey.ramsey(**{'data' :_curr_raster, **ramsey_kwargs}))
    results = [f.result() for f in concurrent.futures.as_completed(results)]
    if shuffle:
        for i in np.arange(0,len(results), num_trials):
            label, s = parallel_labels[i][0], parallel_labels[i][1]
            curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
            np.savez(f'{output_dir}/{label}/{s+1}', eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
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
            label, tn = parallel_labels[i][0], parallel_labels[i][1]
            curr_output = results[i]
            np.savez(f'{output_dir}/{label}/{tn+1}', **curr_output)

            
if __name__ == '__main__':
    DEBUG = 1

    init_log()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mouse', action='store_true')
    parser.add_argument('-t', dest='test',  action='store_true') # test mouse
    parser.add_argument('-i', dest='ising', action='store_true')
    cl_args = parser.parse_args()

    if DEBUG:
        cl_args.ising = True
    #Parallel stuff
    # There are 28 cores
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
                        'num_trials' : 100,
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
    #DEBUG args
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
    output_dir = analysis_args['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(f"{output_dir}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)
    
    import time
    start= time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as EXECUTOR:
        run_analysis(**analysis_args)

    print(time.perf_counter()-start)
