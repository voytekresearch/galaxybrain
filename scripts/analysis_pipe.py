import numpy as np
import pandas as pd
import sys, os
import json
import h5py
import argparse
from pathlib import Path
import sys
from galaxybrain.ising import tensor_to_raster
from galaxybrain.data_utils import MouseData, shuffle_data
from galaxybrain import ramsey
from log_utils.logs import init_log
import logging
import warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def run_analysis(output_dir, num_trials, ramsey_kwargs, data_type, mouse_kwargs={}, shuffle=False, mpi_args={}):
    """
    ramsey_kwargs: dict
    shuffle = (axis, num_shuffles)
    data_type: mouse, or ising
    """

    if data_type == 'mouse':
        mice_data = MouseData(**mouse_kwargs)
        labels = mice_data.get_labels() # labels of form (mouse-name_region-name)
        get_function = lambda label: mice_data.get_spikes(label=label)
    elif data_type == 'ising':
        ising_h5 = h5py.File(str(here_dir/'../data/spikes/ising.hdf5'), 'r')
        labels = list(ising_h5.keys())[4:-4] # these are str temperatures DEBUG for limited temps
        get_function = lambda label: tensor_to_raster(ising_h5[label], keep=1024)


    def trial_task(t):
        logging.info(f'trial {t}')
        curr_raster = get_function(label)
        results = ramsey.Ramsey(data=curr_raster, **ramsey_kwargs, data_type=data_type).subset_iter()
        np.savez(f'{output_dir}/{label}/{t+1}', **results)
        # if shuffle:
        #     results = []
        #     for s in range(shuffle[1]):
        #         curr_raster = shuffle_data(curr_raster, shuffle[0])
        #         results.append(ramsey.ramsey(data=curr_raster, **ramsey_kwargs))
        #     #TODO save data


    for label in labels:
        logging.info(label)
        try:
            os.makedirs(f'{output_dir}/{label}')
        except FileExistsError:
            pass
        if mpi_args:
            if mpi_args['MY_RANK'] != 0: # maps to trial number
                mpi_args['COMM'].send(trial_task(t=mpi_args['MY_RANK']), dest=0)
            else:
                trial_task(t=mpi_args['MY_RANK'])
                for t in range(1, mpi_args['NUM_TRIAL']):
                    mpi_args['COMM'].recv(source=t)
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

            
def main():
    here_dir = Path(__file__).parent.absolute()

    DEBUG = False

    init_log()
    logging.info('begin! \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mouse', action='store_true')
    parser.add_argument('-t', dest='test',  action='store_true') # test mouse
    parser.add_argument('-i', dest='ising', action='store_true')
    parser.add_argument('-p', dest='mpi', action='store_true')
    cl_args = parser.parse_args()
    mpi_args = {}
    if cl_args.mpi:
        from mpi4py import MPI
        COMM = MPI.COMM_WORLD
        mpi_args = {'COMM'      : COMM,
                    'MY_RANK'   : COMM.Get_rank(),
                    'NUM_TRIAL' : COMM.Get_size()}
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
                        'data_type': 'mouse',
                        }
    elif cl_args.mouse:
        analysis_args = {'output_dir' :  str(here_dir/'../data/experiments/mouse'),
                        'mouse_kwargs': {'mouse_in'  : ['waksman']},
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
                        'data_type': 'mouse',
                        }
    #DEBUG args
    elif cl_args.ising:
        analysis_args={'output_dir' : str(here_dir/'../data/experiments/ising'),
                       'ramsey_kwargs' : {'n_iter' : 10,
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
                                                                    'fit_kwargs': {'aperiodic_mode': 'knee'}
                                                            },
                                                        }
                                         },
                        'num_trials' : 4,
                        'data_type': 'ising',
                        }
    output_dir = analysis_args['output_dir']

    with open(f"{output_dir}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)
    
    import time
    start= time.perf_counter()
    run_analysis(**analysis_args, mpi_args=mpi_args)

    print(f'time elapsed {time.perf_counter()-start:.2f}')
