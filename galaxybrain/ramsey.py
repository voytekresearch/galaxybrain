import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from fooof import FOOOFGroup, FOOOF
from scipy import stats
from neurodsp.spectral import compute_spectrum

# from numba import jit

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def fooofy(components, spectra, x_range, group=True):
    """
    fit FOOOF model on given spectrum and return params
        components: frequencies or PC dimensions
        spectra: PSDs or variance explained
        x_range: range for x axis of spectrum to fit
        group: whether to use FOOOFGroup or not
    """
    if group:
        fg = FOOOFGroup(max_n_peaks=0, aperiodic_mode='fixed', verbose = False) #initialize FOOOF object
    else:
        fg = FOOOF(max_n_peaks=0, aperiodic_mode='fixed', verbose = False) #initialize FOOOF object
    #print(spectra.shape, components.shape) #Use this line if things go weird

    fg.fit(components, spectra, x_range)
    exponents = fg.get_params('aperiodic_params', 'exponent')
    errors    = fg.get_params('error') # MAE
    offsets   = fg.get_params('aperiodic_params', 'offset')
    return exponents, errors, offsets


def pca(data, n_pc):
    """
    Decomposition in space
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    pop_pca = PCA(n_pc).fit(data)
    evals = pop_pca.explained_variance_ratio_

    return evals


def ft(subset, **ft_kwargs):
    """
    Decomposition in time over both summed and non summed neurons
    returns: freqs, powers_summed, powers_chans (2d array n_chans x n_freqs)
    """
    if not isinstance(subset, np.ndarray):
        subset = np.array(subset)
    summed_neurons = subset.sum(axis= 1) # summing data for ft decomp.
    freqs, powers_summed = compute_spectrum(summed_neurons, **ft_kwargs)   #powers_sum is an array
    freqs, powers_chans  = compute_spectrum(subset.T, **ft_kwargs)   #returns a matrix! #TODO make sure transpose

    return freqs, powers_summed, powers_chans


#@jit(nopython=True) # jit not working because I think the data passed in has to be array
def random_subset_decomp(raster_curr, subset_size, n_pc, pc_range, f_range, n_iter=150):
    """shuffle: either 'space' or 'time' to destroy correlations differently
    returned data include 1 pca exponent and 2 PSD exponents
    """
    FS       = 1
    NPERSEG  = 120
    NOVERLAP = NPERSEG/2

    freqs = np.fft.rfftfreq(NPERSEG)

    evals_mat       = np.empty((n_iter, n_pc)) # n_iter * |evals|
    sum_powers_mat  = np.empty((n_iter, len(freqs)))
    chan_powers_mat = np.empty((n_iter, subset_size, len(freqs)))

    for i in np.arange(n_iter):

        loc_array = np.sort(np.random.choice(raster_curr.shape[1], subset_size, replace=False))
        subset = np.array(raster_curr.iloc[:,loc_array]) #currently converted to array for testing jit

        # decomposition in space
        evals = pca(subset, n_pc)
        evals_mat[i] = evals

        # decomposition in time
        freqs, powers_sum, powers_chans = ft(subset, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)

        sum_powers_mat[i]  = powers_sum
        chan_powers_mat[i] = powers_chans

    e_axis = np.arange(1,n_pc+1)
    pca_m_array, pca_er_array, pc_offsets  = fooofy(e_axis, evals_mat, pc_range) #space decomposition exponents, and er
    ft_m_array1, ft_er_array1, ft_offsets1 = fooofy(freqs, sum_powers_mat, f_range) #time decomposition exponents, and er
    ft_m_array2, ft_er_array2, ft_offsets2 = np.mean([fooofy(freqs, chan_powers_mat[:,it], f_range) for it in range(subset_size)], axis=0) # list comp here iterates over each neuron
    
    spectra = {'evals':evals_mat,'psd':sum_powers_mat, 'psd_chan':chan_powers_mat}
    fit_dict = {'pca_m':pca_m_array,
                'pca_er':pca_er_array,
                'pca_b':pc_offsets,
                'ft_m1':ft_m_array1,
                'ft_er1':ft_er_array1,
                'ft_b1':ft_offsets1,
                'ft_m2':ft_m_array2,
                'ft_er2':ft_er_array2,
                'ft_b2':ft_offsets2 }

    return spectra, fit_dict


def ramsey(data, subset_sizes, n_iters, n_pc=None, pc_range=[0,None], f_range=[0,None]):
    """Do random_subset_decomp over incrementing subset sizes
    slope dims: n_iters * amount of subset sizes
    b: offsets
    returns: eigs, pows (2D)
            fit results and stats"""
    n = len(subset_sizes)
    eigs        = []
    powers_sum  = []
    powers_chan = [] #UNUSED
    fit_results = defaultdict(lambda: np.empty((n_iters, n)))
    stats_ = defaultdict(lambda: np.empty(n))

    # pc_range_history = []
    for i, n_i in enumerate(subset_sizes):

        #if at some subset size not enough pc's, default to biggest
        #default is using a proportion of that

        if n_pc == None: #does this still need to be None?  Will it ever be manually changed?
            n_pc_curr = min(subset_sizes)
        elif isinstance(n_pc, int) and n_pc < n_i:
            n_pc_curr = n_pc
        elif isinstance(n_pc, float):
            n_pc_curr = int(n_pc*n_i)

        #write conditions for pc_range,  use a function, or outside of this
        # [0,None] for whole range, otherwise check if float for fraction
        if pc_range == [0,None]:
            curr_pc_range = [0, int(min(.5*n_pc_curr, .25*max(subset_sizes*n_pc)))]
        elif isinstance(pc_range[1], float): #if second element of pc_range is float, it is a percentage of pc's
            pc_frac = pc_range[1]
            curr_pc_range = [pc_range[0],int(n_pc_curr*pc_frac)]

        #f_range conditions
        if isinstance(f_range[1], float):
            curr_f_range = f_range
        elif f_range[1] == None:
            curr_f_range = None

        # pc_range_history.append(curr_pc_range)
        spectra_i, results_i = random_subset_decomp(data, n_i, n_pc_curr, curr_pc_range, curr_f_range, n_iters) #remember to add parameters later, check function doc for output

        #append average across iterations
        eigs.append(spectra_i['evals'].mean(0)) 
        powers_sum.append(spectra_i['psd'].mean(0))

        for measure, dat in results_i.items():
            # print(f'~~~~~~~~~~DEBUG {i} {n_i} \n {measure}, {dat.shape} \n', flush=True)
            fit_results[measure][:,i] =  dat

        for it in [1,2]: #summed and non summed
            stats_[f'pearson_r{it}'][i], stats_[f'pearson_p{it}'][i] = stats.pearsonr(results_i['pca_m'], results_i[f'ft_m{it}'])
            stats_[f'spearman_rho{it}'][i], stats_[f'spearman_p{it}'][i] = stats.pearsonr(results_i['pca_m'], results_i[f'ft_m{it}'])
        
    # Have to unpack dict for parallel processing. key order is conserved in python 3.6+
    return (eigs, powers_sum, *[v for v in fit_results.values()], *[v for v in stats_.values()]) #, pc_range_history