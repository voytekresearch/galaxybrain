import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import fooof
from fooof import FOOOFGroup, FOOOF
from scipy import io, signal, stats
from neurodsp.spectral import compute_spectrum

from numba import jit

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def fooofy(components, spectra, freq_range, group = True):
    """
    A FOOOF Function, gets exponent parameters
    """
    if group:
        fg = FOOOFGroup(max_n_peaks=0, aperiodic_mode='fixed', verbose = False) #initialize FOOOF object
    else:
        fg = FOOOF(max_n_peaks=0, aperiodic_mode='fixed', verbose = False) #initialize FOOOF object
    #print(spectra.shape, components.shape) #Use this line if things go weird

    fg.fit(components, spectra, freq_range) # THIS IS WHERE YOU SAY WHICH FREQ RANGE TO FIT
    m_array = fg.get_params('aperiodic_params', 'exponent')
    #r2_array = fg.get_params('r_squared') #correlation between components (freqs or PCs) and spectra (powers or eigvals)
    er_array = fg.get_params('error')
    #fg.r_squared_
    return m_array, er_array

def pca_on_data(subset, n_pc):
    """
    Decomposition in space
    """
    pop_pca = PCA(n_pc).fit(subset)
    evals = pop_pca.explained_variance_ratio_

    return evals

def ft_on_data(subset, fs, nperseg, noverlap):
    """
    Decomposition in time
    """
    if type(subset) != np.ndarray:
        subset = np.array(subset)
    summed_neurons = subset.sum(axis= 1) # summing data for ft decomp.
    freqs, powers = compute_spectrum(summed_neurons, fs = fs, nperseg = nperseg, noverlap = noverlap)   #making these parameters now

    return freqs, powers

#@jit(nopython=True) # jit not working because I think the data passed in has to be array
def random_subset_decomp(data, subset_size, n_iter, n_pc, pc_range, f_range, verbose = False):
    """shuffle: either 'space' or 'time' to destroy correlations differently"""
    #Make these parameters for main func later
    fs=1; nperseg=120; noverlap=60

    freqs = np.fft.rfftfreq(nperseg) #added this

    evals_mat = np.zeros((n_iter, n_pc)) # n_iter * |evals|
    powers_mat = np.zeros((n_iter, len(freqs)))

    for i in np.arange(n_iter):
        raster_curr = data

        loc_array = np.sort(np.random.choice(raster_curr.shape[1], subset_size, replace=False))
        subset = np.array(raster_curr.iloc[:,loc_array]) #currently converted to array for testing jit

        # decomposition in space
        evals = pca_on_data(subset, n_pc)
        evals_mat[i] = evals

        # decomposition in time
        freqs, powers = ft_on_data(subset, fs, nperseg, noverlap)
        powers_mat[i] = powers

    e_axis = np.arange(1,n_pc+1)
    pca_m_array, pca_er_array = fooofy(e_axis, evals_mat, pc_range) #space decomposition exponents, and er
    ft_m_array, ft_er_array = fooofy(freqs, powers_mat, f_range) #time decomposition exponents, and er

    if verbose == True:

        print('Avg Space Decomposition slope:', np.mean(pca_m_array))
        print('Avg Time Decomposition slope:', np.mean(ft_m_array))

        plt.figure()
        plt.hist(pca_m_array)
        plt.title('Space Decomp')
        #plt.title('n_subset = ', subset_size)
        plt.figure()
        plt.hist(ft_m_array)
        plt.title('Time Decomp')
        plt.plot()

    return evals_mat, pca_m_array, pca_er_array, powers_mat, ft_m_array, ft_er_array

def ramsey(data, subset_sizes, n_iters = 150, n_pc = None, pc_range = [0,None], f_range = [0,None]):
    """Do random_subset_decomp over incrementing subset sizes"""
    n = len(subset_sizes)

    #these could also be np.zeros((n_iters, n)) (but check dims)
    eigs = []
    powers = []

    pca_m = np.zeros((n_iters, n)) # dims: n_iters * amount of subset sizes
    ft_m = np.zeros((n_iters, n))

    space_er = np.zeros((n_iters, n))
    time_er = np.zeros((n_iters, n))

    pearson_r = np.zeros(n)
    pearson_p = np.zeros(n)
    spearman_rho = np.zeros(n)
    spearman_p = np.zeros(n)

    for i, n_i in enumerate(subset_sizes):

        #if at some subset size not enough pc's, default to biggest
        #default is using a proportion of that

        if n_pc == None: #does this still need to be None?  Will it ever be manually changed?
            n_pc_curr = min(subset_sizes)

        elif type(n_pc) == int and n_pc < n_i:
            n_pc_curr = n_pc

        elif type(n_pc) == float:
            n_pc_curr = int(n_pc*n_i)

        #write conditions for pc_range,  use a function, or outside of this
        # [0,None] for whole range, otherwise check if float for fraction
        if pc_range == [0,None]:
            curr_pc_range = [0, int(min(.5*n_pc_curr, .25*max(subset_sizes*n_pc)))]
            # if n_i < 200:
            #     curr_pc_range = [0,int(n_pc_curr*0.5)]
            # else:
            #     curr_pc_range = [0,200]

        elif type(pc_range[1]) == float: #if second element of pc_range is float, it is a percentage of pc's
            pc_frac = pc_range[1]
            curr_pc_range = [pc_range[0],int(n_pc_curr*pc_frac)]
        #
        # elif pc_range[1] == None:
        #     curr_pc_range = None

        #f_range conditions
        if type(f_range[1]) == float:
            curr_f_range = [f_range[0],f_range[1]]

        elif f_range[1] == None:
            curr_f_range = None

        evs, ev_m, ev_er, pows, pow_m, pow_er = random_subset_decomp(data, n_i, n_iters, n_pc_curr , pc_range = curr_pc_range, f_range = curr_f_range) #remember to add parameters later, check function doc for output

        #append average across iterations
        eigs.append(evs.mean(0)) 
        powers.append(pows.mean(0))

        pca_m[:,i] = ev_m
        ft_m[:,i] = pow_m

        space_er[:,i] = ev_er
        time_er[:,i] = pow_er

        #This is where you'd start resampling (iteratively)
        pearson_r[i], pearson_p[i] = stats.pearsonr(ev_m, pow_m)
        spearman_rho[i], spearman_p[i] = stats.spearmanr(ev_m, pow_m)
    #eigs, pows are 2d
    return eigs, powers, pca_m, space_er, ft_m, time_er, pearson_r, spearman_rho, pearson_p, spearman_p

def plot_all_measures(subsetsizes, space_er, time_er, n_iters, n_pc, f_range, pc_range, eigs, pows, space_slopes, time_slopes, pearson_corr, spearman_corr, pearson_p, spearman_p, figsize = (23,8)):

    """
    Plots everything from above
    """
    
    #stylistic details
             
    plt.rcParams['mathtext.default'] = 'regular'

    font = {'family' : 'Arial',
       'weight' : 'regular',
       'size'   : 14}
    
    plt.rc('font', **font)
    plt.rcParams['axes.spines.top']=False
    plt.rcParams['axes.spines.right']=False

    n = len(subsetsizes)
    plt.figure(figsize=figsize)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cool(np.linspace(0,1,n)))

    #plot spectra
    for i, n_i in enumerate(subsetsizes):
        mean_evs = eigs[i]
        mean_pows = pows[i]
        if n_pc == None: #does this still need to be None?  Will it ever be manually changed?
            n_pc_curr = min(subsetsizes)

        elif type(n_pc) == int and n_pc < n_i:
            n_pc_curr = n_pc

        elif type(n_pc) == float:
            n_pc_curr = int(n_pc*n_i)

        # plot eigenspectrum
        plt.subplot(2,5,1)
        #plt.loglog(np.arange(1,n_pc+1), evs.T, 'k', lw=1, alpha=0.2)
        #plt.loglog(np.arange(1,n_pc+1), evs.mean(0), 'r')
        plt.plot(np.arange(1,n_pc_curr+1)/n_pc_curr, mean_evs) #KEEP THIS LINE: proportion of PCs
        
        plt.plot(range(len(mean_evs)), mean_evs) #KEEP THIS LINE: actual PCs
        plt.yscale('log'); plt.xscale('log');
        plt.title('Eigenvalue Spectrum')
        plt.xlabel('PC dimension'); plt.ylabel('Variance')
        # except:
        print(f'n_pc_curr: {len(np.arange(1,n_pc_curr+1))}')
        print(f'evs: {mean_evs.shape}')
        #plot powerspectrum
        plt.subplot(2,5,6)
        #plt.loglog(np.arange(0,0.505,0.005), pows.T, 'k', lw=1, alpha=0.2)
        #plt.loglog(np.arange(0,0.505,0.005), pows.mean(0), 'r')
        plt.plot(np.arange(0,61/120, 1/120), mean_pows)
        plt.yscale('log'); plt.xscale('log');
        plt.title('Power Spectrum')
        plt.xlabel('Frequency (Hz)'); plt.ylabel('Power')

    #Space dimension slopes
    plt.subplot(2,5,2)
    plt.errorbar(subsetsizes[1:], space_slopes.mean(0)[1:], space_slopes.std(0)[1:], color = 'black', alpha = 0.5)
    plt.title('Average eigenvalue spectrum exponent \n at each subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('Exponent')

    #Time dimension slopes
    plt.subplot(2,5,7)
    plt.errorbar(subsetsizes[1:], time_slopes.mean(0)[1:], time_slopes.std(0)[1:], color = 'black', alpha = 0.5)
    plt.title('Average power spectrum exponent \n at each subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('Exponent')

    #PCA goodness of fit
    plt.subplot(2,5,3)
    plt.plot(subsetsizes[:], space_er.T[:], ".", color = 'purple', lw=1, alpha=0.2)
    plt.title('Fit error for eigenspectrum \n at subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('error')

    #FFT goodness of fit
    plt.subplot(2,5,8)
    plt.plot(subsetsizes[:], time_er.T[:],  ".", color = 'purple', lw=1, alpha=0.2)
    plt.title('Fit error for power spectrum \n at subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('error')

    #Pearson (R) Correlation value as function of subset size
    n_trials = pearson_corr.shape[0]
    plt.subplot(2,5,4)
    pearson_corr = np.array([pearson_corr[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, pearson_corr.mean(0), pearson_corr.std(0), color = 'blue', alpha = 0.5)
    plt.plot(subsetsizes, pearson_corr.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, pearson_corr, color = 'blue', alpha = 0.5)
    plt.title('Pearson\'s r as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel('r')

    #Spearman (Rho) Correlation value as function of subset size
    plt.subplot(2,5,9)
    spearman_corr = np.array([spearman_corr[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, spearman_corr.mean(0), spearman_corr.std(0), color = 'blue', alpha = 0.5)
    plt.plot(subsetsizes, spearman_corr.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, spearman_corr, color = 'blue', alpha = 0.5)
    plt.title('Spearman\'s ρ as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel('ρ')

    #Pearson p values
    plt.subplot(2,5,5)
    pearson_p = np.log10(np.array([pearson_p[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, pearson_p.mean(0), pearson_p.std(0), color = 'green', alpha = 0.5)
    plt.plot(subsetsizes, pearson_p.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle = '--', color = 'orange', lw = 2, alpha = 0.75, label = 'p = 0.05')
    plt.legend()
    #plt.semilogy(subsetsizes, pearson_p, color = 'green', alpha = 0.5)
    # plt.yscale('log')
    plt.title('Pearson p value as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel('$log_{10}p$') 

    #Spearman p values
    plt.subplot(2,5,10)
    spearman_p = np.log10(np.array([spearman_p[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, spearman_p.mean(0), spearman_p.std(0), color = 'green', alpha = 0.5)
    plt.plot(subsetsizes, spearman_p.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle = '--', color = 'orange', lw = 2, alpha = 0.75)
    # plt.yscale('log')
    #plt.semilogy(subsetsizes, spearman_p, color = 'green', alpha = 0.5)
    plt.title('Spearman p value as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel('$log_{10}p$') 

    plt.tight_layout()
    plt.draw()
