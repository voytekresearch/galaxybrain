import numpy as np
import pandas as pd
import sys, os
import json
import yaml
import h5py
import argparse
from pathlib import Path
import sys
from galaxybrain.ising import tensor_to_raster
from galaxybrain.data_utils import MouseData, shuffle_data
from galaxybrain import ramsey
from log_utils.logs import init_log
import gc
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

HERE_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR_ROOT = str(HERE_DIR/'../data/experiments')


def run_analysis(output_dir, logger, num_trials, data_type, shuffle, ramsey_kwargs, mouse_kwargs={}, mpi_args={}):
    """
    ramsey_kwargs: dict
    shuffle = (axis, num_shuffles)
    data_type: mouse, or ising
    """

    if data_type == 'mouse':
        mice_data = MouseData(**mouse_kwargs)
        labels = mice_data.get_labels() # labels of form (mouse-name_region-name)
        def get_function(label): 
            return mice_data.get_spikes(label=label)
    elif data_type == 'ising':
        with h5py.File(str(HERE_DIR/'../data/spikes/ising.hdf5'), 'r') as f:
            ising_h5 = {k: np.array(f[k]) for k in f.keys()}
        labels = list(ising_h5.keys())[4:8] # these are str temperatures DEBUG for limited temps
        del ising_h5
        gc.collect()
        def get_function(label):
            with h5py.File(str(HERE_DIR/'../data/spikes/ising.hdf5'), 'r') as f:
                ising_h5 = {k: np.array(f[k]) for k in f.keys()}
            return tensor_to_raster(ising_h5[label], keep=700)


    def trial_task(t, label):
        logger.info(f'trial {t+1}')
        if not shuffle:
            curr_raster = get_function(label)
            results = ramsey.Ramsey(data=(get_function, label), **ramsey_kwargs, data_type=data_type, logger=logger).subset_iter()
            np.savez(f'{output_dir}/{label}/{t+1}', **results)
        if shuffle:
            results = []
            for s in range(shuffle[1]):
                curr_raster = get_function(label)
                curr_raster = shuffle_data(curr_raster, shuffle[0])
                results.append(ramsey.Ramsey(data=curr_raster, **ramsey_kwargs, data_type=data_type, logger=logger).subset_iter())
            #TODO save data


    for label in labels:
        logger.info(f'label: {label}')
        try:
            os.makedirs(f'{output_dir}/{label}')
        except FileExistsError:
            pass
        if mpi_args:
            trial_task(mpi_args['MY_RANK'], label)
        else:
            for t in range(num_trials):
                trial_task(t, label)
            
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

    DEBUG = True

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
                    'NUM_TRIAL' : COMM.Get_size()} # specified in job script
        logger = init_log(COMM.Get_rank())
    else:
        logger = init_log(None)
    if DEBUG:
        cl_args.test = True
    logger.info('Begin')
    # TODO read yaml file, overwrite values if not on cluster (e.g, num_trials)
    with open('pipeline_config.yaml', 'r') as f:
        pipeline_args = yaml.safe_load(f)
        analysis_args_main = pipeline_args['analysis_args']

    #Parallel stuff
    # There are 28 cores
    if cl_args.test:
        analysis_args = analysis_args_main['test']
        analysis_args['output_dir'] = os.path.join(OUTPUT_DIR_ROOT, 'TEST')
    elif cl_args.mouse:
        analysis_args['output_dir'] = os.path.join(OUTPUT_DIR_ROOT, 'mouse')
    elif cl_args.ising:
        analysis_args['output_dir'] = os.path.join(OUTPUT_DIR_ROOT, 'ising')

    analysis_args['num_trials'] = pipeline_args['job_args']['NUM_NODES']
    analysis_args['ramsey_kwargs']['num_proc'] = pipeline_args['job_args']['NUM_PROC']

    os.environ['NUMEXPR_MAX_THREADS'] = str(analysis_args['ramsey_kwargs']['n_iter']) # otherwise numexpr knocks it down
    output_dir = analysis_args['output_dir']

    with open(f"{output_dir}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)
    
    import time
    start = time.perf_counter()
    run_analysis(**analysis_args, mpi_args=mpi_args, logger=logger)

    logger.info(f'time elapsed: {time.perf_counter()-start:.2f}')
