import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from fooof import FOOOFGroup, FOOOF
from scipy import stats
from neurodsp.spectral import compute_spectrum
import time # debug
# from numba import jit
from pathlib import Path
import sys
here_dir = Path(__file__).parent.absolute()
sys.path.append(str(here_dir.parent.absolute()/'log_utils'))
from logs import init_log
import logging
init_log()
import warnings
warnings.filterwarnings("ignore")

from memory_profiler import profile #DEBUG

def fooofy(components, spectra, fit_range,
           group=True, 
           fit_kwargs={},
           return_params='default', 
           ):
    """
    fit FOOOF model on given spectrum and return params
        components: frequencies or PC dimensions
        spectra: PSDs or variance explained
        fit_range: range for x axis of spectrum to fit
        group: whether to use FOOOFGroup or not
    """
    if return_params == 'default':
        return_params = [['aperiodic_params', 'exponent'],
                          ['error'], # MAE
                          ['aperiodic_params', 'offset']]
    if group:
        fg = FOOOFGroup(max_n_peaks=0, verbose=False, **fit_kwargs)
    else:
        fg = FOOOF(max_n_peaks=0, verbose=False, **fit_kwargs)

    fg.fit(components, spectra, fit_range)
    return {p[-1]: fg.get_params(*p) 
                    for p in return_params}

#@profile
def pca(data, n_pc=None):
    """
    Decomposition in space
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    pop_pca = PCA(n_pc).fit(data)
    evals = pop_pca.explained_variance_ratio_

    return evals

#@profile
def ft(subset, **ft_kwargs):
    """
    Welch method over both summed and non summed neurons
    returns: freqs, powers_summed, powers_chans (2d array n_chans x n_freqs)

    fourier kwargs for mouse:
        fs       = 1
        nperseg  = 120
        noverlap = NPERSEG/2
    for ising model 10000 long
        fs       = 1
        nperseg  = 2000
        noverlap = int(.8*NPERSEG)
    """
    if not isinstance(subset, np.ndarray):
        subset = np.array(subset)
    summed_neurons = subset.sum(axis= 1) # summing data for ft decomp.
    freqs, powers_summed = compute_spectrum(summed_neurons, **ft_kwargs, method='welch') #powers_sum is an array
    freqs, powers_chans  = compute_spectrum(subset.T,       **ft_kwargs, method='welch') #returns a matrix!

    return freqs, powers_summed, powers_chans


#@jit(nopython=True) # jit not working because I think the data passed in has to be array
#@profile
def random_subset_decomp(raster_curr, subset_size, n_iter, n_pc, ft_kwargs, pc_range, f_range, fooof_kwargs={}):
    """
    returned data include 1 pca exponent and 2 PSD exponents
    fooof_kwargs = { 'es' : {
                        'return_params': [...],
                        'fit_kwargs': {...}
                        }
                     'psd': SAME^
                     }
    """
    freqs = np.fft.rfftfreq(ft_kwargs['nperseg'])
    pcs   = np.arange(1,n_pc+1)

    evals_mat       = np.empty((n_iter, n_pc)) # n_iter * |evals|
    sum_powers_mat  = np.empty((n_iter, len(freqs)))
    chan_powers_mat = np.empty((n_iter, subset_size, len(freqs)))

    for i in np.arange(n_iter):

        loc_array = np.sort(np.random.choice(raster_curr.shape[1], subset_size, replace=False))
        subset = np.array(raster_curr.iloc[:,loc_array]) #currently converted to array for testing jit

        # decomposition in space
        evals = pca(subset, n_pc)
        # PCA_TIMES.append(tdiff(pca_start))
        evals_mat[i] = evals

        # decomposition in time
        freqs, powers_sum, powers_chans = ft(subset, **ft_kwargs)

        sum_powers_mat[i]  = powers_sum
        chan_powers_mat[i] = powers_chans
        
    es_fooof_kwargs, psd_fooof_kwargs = fooof_kwargs.get('es', {}), fooof_kwargs.get('psd', {})
    es_fit   = fooofy(pcs, evals_mat,      pc_range, **es_fooof_kwargs)
    psd_fit1 = fooofy(freqs,  sum_powers_mat, f_range,  **psd_fooof_kwargs)
    include_psd_fit2 = True #DEBUG
    try:
        psd_fit2_list= [fooofy(freqs, chan_powers_mat[:,chan], f_range, **psd_fooof_kwargs) 
                                                        for chan in range(subset_size)] # list comp here iterates over each neuron
        psd_fit2 = defaultdict(lambda: [])
        for d in psd_fit2_list:
            for key, params in d.items():
                psd_fit2[key].append(params)
        psd_fit2 = {k:np.mean(v, axis=0) for k,v in psd_fit2.items()}
    except Exception as e:
        include_psd_fit2 = False

    spectra = {'evals':evals_mat,'psd':sum_powers_mat, 'psd_chan':chan_powers_mat}
    fit_dict = {**{f'es_{k}'   :v for k,v in es_fit.items()}, # renaming keys for each measure
                **{f'psd_{k}1':v for k,v in psd_fit1.items()}}
    if include_psd_fit2:
        fit_dict = {**fit_dict, 
                    **{f'psd_{k}2':v for k,v in psd_fit2.items()}}

    return spectra, fit_dict

#@profile
def ramsey(data, n_iter, n_pc, ft_kwargs, pc_range, f_range, fooof_kwargs={}, data_type='mouse'):
    """Do random_subset_decomp over incrementing subset sizes
    slope dims: n_iter * amount of subset sizes
    b: offsets
    returns: eigs, pows (2D)
            fit results and stats"""
    if data_type == 'mouse':
        subset_sizes = np.linspace(30, data.shape[1], 16, dtype=int)
    elif data_type == 'ising':
        subset_sizes = np.linspace(30, data.shape[1], 16, dtype=int) #  - 10
        
    n           = len(subset_sizes)
    eigs        = []
    powers_sum  = []
    fit_results = defaultdict(lambda: np.empty((n_iter, n)))
    stats_      = defaultdict(lambda: np.empty(n))

    for i, num in enumerate(subset_sizes):
        #if at some subset size not enough pc's, default to biggest
        #default is using a proportion of that

        if n_pc == None: #does this still need to be None?  Will it ever be manually changed?
            n_pc_curr = min(subset_sizes)
        elif isinstance(n_pc, int) and n_pc < num:
            n_pc_curr = n_pc
        elif isinstance(n_pc, float):
            n_pc_curr = int(n_pc*num)

        #write conditions for pc_range,  use a function, or outside of this
        # [0,None] for whole range, otherwise check if float for fraction
        if pc_range == [0,None]:
            curr_pc_range = [0, int(min(.5*n_pc_curr, .25*max(subset_sizes*n_pc)))]
        elif isinstance(pc_range[1], float): #if second element of pc_range is float, it is a percentage of pc's
            pc_frac = pc_range[1]
            curr_pc_range = [pc_range[0],int(n_pc_curr*pc_frac)]
        # DEBUG can't fit if range is too small
        if curr_pc_range[1] < 3:
            logging.info(f'    skipping subset {num}')
            continue

        #f_range conditions
        if isinstance(f_range[1], float):
            curr_f_range = f_range
        elif f_range[1] == None:
            curr_f_range = None

        spectra_i, results_i = random_subset_decomp(data, num, n_iter, n_pc_curr, ft_kwargs, curr_pc_range, curr_f_range, fooof_kwargs)

        #append average across iterations
        eigs.append(spectra_i['evals'].mean(0)) 
        powers_sum.append(spectra_i['psd'].mean(0))

        for measure, dat in results_i.items():
            fit_results[measure][:,i] =  dat

        for it in [1,2]: #summed and non summed
            try:
                stats_[f'pearson_corr{it}'][i],  stats_[f'pearson_p{it}'][i]  = stats.pearsonr( results_i['es_exponent'], results_i[f'psd_exponent{it}'])
                stats_[f'spearman_corr{it}'][i], stats_[f'spearman_p{it}'][i] = stats.spearmanr(results_i['es_exponent'], results_i[f'psd_exponent{it}'])
            except ValueError: # can't compute correlation
                logging.info(f"NaNs at subset iter: {i}")
            except KeyError: # skip psd_exponent nosum because of 0s spec
                logging.info(f'0s spec at subset iter: {i}')

    # NOTE: need to unpack dict in shuffle case (key order conserved python 3.6)
    print('returning stiuff!')
    return {'eigs': eigs, 
            'pows': powers_sum, 
            **fit_results, 
            **stats_}