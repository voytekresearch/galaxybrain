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
def run_analysis(output_dir, num_trials, ramsey_kwargs, data_type, mouse_kwargs={}, shuffle=False):
    """
    ramsey_kwargs: dict
    burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb)
    shuffle = (axis, num_shuffles)
    data_type: mouse, or ising
    """

    if data_type == 'mouse':
        mice_data = MouseData(**mouse_kwargs)
        labels = mice_data.get_labels() # labels of form (mouse_name, region_name)
        get_function = lambda label: mice_data.get_spikes(label=label)
    elif data_type == 'ising':
        ising_h5 = h5py.File(str(here_dir/'../data/spikes/ising.hdf5'), 'r')
        # DEBUG
        labels = list(ising_h5.keys())[4:-4] # these are str temperatures
        get_function = lambda label: tensor_to_raster(ising_h5[label], keep=1024)


    def trial_task(t):
        logging.info(f'trial {t}')
        curr_raster = get_function(label)
        results = ramsey.Ramsey(data=curr_raster, **ramsey_kwargs).subset_iter()
        np.savez(f'{output_dir}/{label}/{t+1}', **results)
        # if shuffle:
        #     results = []
        #     for s in range(shuffle[1]):
        #         curr_raster = shuffle_data(curr_raster, shuffle[0])
        #         results.append(ramsey.ramsey(data=curr_raster, **ramsey_kwargs))
        #     #TODO save data


    for label in labels:
        logging.info(label)
        os.makedirs(f'{output_dir}/{label}')
        if cl_args.mpi:
            if MY_RANK != 0: # maps to trial number
                COMM.send(trial_task(t=MY_RANK))
            else:
                trial_task(t=MY_RANK)
                for t in range(1, NUM_TRIAL):
                    COMM.recv(source=t)
        else:
            for t in range(num_trials):
                trial_task(t)
            
    # TODO this is outdated
    # if shuffle:
    #     for i in np.arange(0,len(results), num_trials):
    #         label, s = parallel_labels[i][0], parallel_labels[i][1]
    #         curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
    #         np.savez(f'{output_dir}/{label}/{s+1}', eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
    #                                                             pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
    #                                                             pca_m=curr_output[:,2].mean(0), pca_er=curr_output[:,3].mean(0), pca_b=curr_output[:,4].mean(0), 
    #                                                             ft_m1=curr_output[:,5].mean(0), ft_er1=curr_output[:,6].mean(0), ft_b1=curr_output[:,7].mean(0), 
    #                                                             ft_m2=curr_output[:,8].mean(0), ft_er2=curr_output[:,9].mean(0), ft_b2=curr_output[:,10].mean(0), 
    #                                                             pearson_r1=curr_output[:,11].mean(0), spearman_rho1=curr_output[:,13].mean(0), 
    #                                                             pearson_p1=curr_output[:,12].mean(0), spearman_p1=curr_output[:,14].mean(0),
    #                                                             pearson_r2=curr_output[:,15].mean(0), spearman_rho2=curr_output[:,17].mean(0), 
    #                                                             pearson_p2=curr_output[:,16].mean(0), spearman_p2=curr_output[:,18].mean(0))

            
if __name__ == '__main__':
    DEBUG = 1

    init_log()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mouse', action='store_true')
    parser.add_argument('-t', dest='test',  action='store_true') # test mouse
    parser.add_argument('-i', dest='ising', action='store_true')
    parser.add_argument('-p', dest='mpi', action='store_true')

    cl_args = parser.parse_args()

    if cl_args.mpi:
        from mpi4py import MPI

        COMM = MPI.COMM_WORLD
        MY_RANK = COMM.Get_rank()
        NUM_TRIAL = COMM.Get_size()
    if DEBUG:
        cl_args.ising = True
    #Parallel stuff
    # There are 28 cores
    if cl_args.test:
        analysis_args = {'output_dir' :  str(here_dir/'../data/experiments/TEST'),
                        'mouse_kwargs': {'phantom':True},
                        'ramsey_kwargs' : {
                                            'n_iter': 5, 
                                            'n_pc': 0.8, 
                                            'pc_range': [0, None],
                                            'f_range': [0,0.4],
                                            'ft_kwargs': {
                                                            'fs'      : 1,
                                                            'nperseg' : 120,
                                                            'noverlap': 120/2
                                                        }
                                        },
                        'num_trials' : 5,
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
                       'ramsey_kwargs' : {'n_iter' : 10,
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
                        'data_type': 'ising',
                        }
    output_dir = analysis_args['output_dir']
    if not cl_args.mpi:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    with open(f"{output_dir}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)
    
    import time
    start= time.perf_counter()
    run_analysis(**analysis_args)

    print(time.perf_counter()-start)
