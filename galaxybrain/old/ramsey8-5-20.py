import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import fooof
from fooof import FOOOFGroup
from scipy import io, signal, stats
from neurodsp.spectral import compute_spectrum

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def fooofy(components, spectra, freq_range):
    """
    A FOOOF Function, gets exponent parameters
    """
    fg = FOOOFGroup(max_n_peaks=0, aperiodic_mode='fixed', verbose = False) #initialize FOOOF object

    #print(spectra.shape, components.shape) #Use this line if things go weird

    fg.fit(components, spectra, freq_range) # THIS IS WHERE YOU SAY WHICH FREQ RANGE TO FIT
    m_array = fg.get_params('aperiodic_params', 'exponent')
    r2_array = fg.get_params('r_squared') #correlation between components (freqs or PCs) and spectra (powers or eigvals)
    #fg.r_squared_
    return m_array, r2_array

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
    summed_neurons = subset.sum(axis= 1) # summing data for ft decomp.
    freqs, powers = compute_spectrum(summed_neurons, fs = fs, nperseg = nperseg, noverlap = noverlap)   #making these parameters now

    return freqs, powers


#set non as default!
def random_subset_decomp(data, subset_size, n_iter, n_pc, pc_range, f_range, verbose = False):

    #Make these parameters for main func later
    fs=1; nperseg=120; noverlap=60

    freqs = np.fft.rfftfreq(nperseg) #added this

    evals_mat = np.zeros((n_iter, n_pc)) # n_iter * |evals|
    powers_mat = np.zeros((n_iter, len(freqs)))

    for i in np.arange(n_iter):
        loc_array = np.sort(np.random.choice(data.shape[1], subset_size, replace=False))
        subset = data.iloc[:,loc_array]

        # decomposition in space
        evals = pca_on_data(subset, n_pc)
        evals_mat[i] = evals

        # decomposition in time
        freqs, powers = ft_on_data(subset, fs, nperseg, noverlap)
        powers_mat[i] = powers

#     print('PCA')
    e_axis = np.arange(1,n_pc+1)
#     print(e_axis.shape, evals_mat.shape, pc_range)
    pca_m_array, pca_r2_array = fooofy(e_axis, evals_mat, pc_range) #space decomposition exponents, and r2
#     print('PSD')
    ft_m_array, ft_r2_array = fooofy(freqs, powers_mat, f_range) #time decomposition exponents, and r2

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

    return evals_mat, pca_m_array, pca_r2_array, powers_mat, ft_m_array, ft_r2_array

def ramsey(data, subset_sizes, n_iters = 150, n_pc = None, pc_range = [0,None], f_range = [0,None], verbose = True):
    n = len(subset_sizes)

    #size = np.arange(0,n)
    #eigs = np.zeros((n_iters, n)) #right dims?
    eigs = []
    #powers = np.zeros((n_iters, n))
    powers = []

    pca_m = np.zeros((n_iters, n)) # dims: n_iters * amount of subset sizes
    ft_m = np.zeros((n_iters, n))

    space_r2 = np.zeros((n_iters, n))
    time_r2 = np.zeros((n_iters, n))

    pearson_r = np.zeros(n)
    pearson_p = np.zeros(n)
    spearman_rho = np.zeros(n)
    spearman_p = np.zeros(n)

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.inferno(np.linspace(0,1,n)))

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

        # if type(pc_range[1]) == float: #if second element of pc_range is float, it is a percentage of pc's
        #     pc_frac = pc_range[1]
        #     curr_pc_range = [pc_range[0],int(n_pc_curr*pc_frac)]
        #
        # elif pc_range[1] == None:
        #     curr_pc_range = None

        #f_range conditions
        if type(f_range[1]) == float:
            curr_f_range = [f_range[0],f_range[1]]

        elif f_range[1] == None:
            curr_f_range = None


        evs, ev_m, ev_r2, pows, pow_m, pow_r2 = random_subset_decomp(data, n_i, n_iters, n_pc_curr , pc_range = curr_pc_range, f_range = curr_f_range) #remember to add parameters later, check function doc for output

        #this was commented out so no data was saved for past 2 experiments
        # eigs[:,i] = evs.mean(0)
        # powers[:,i] = pows.mean(0)
        eigs.append(evs.mean(0))
        powers.append(pows.mean(0))

        pca_m[:,i] = ev_m
        ft_m[:,i] = pow_m

        space_r2[:,i] = ev_r2
        time_r2[:,i] = pow_r2

        #This is where you'd start resampling (iteratively)
        pearson_r[i], pearson_p[i] = stats.pearsonr(ev_m, pow_m)
        spearman_rho[i], spearman_p[i] = stats.spearmanr(ev_m, pow_m)

        if verbose == True:

            if i == 0: #only creates figure for the first iter, then it just plots over
                plt.figure(figsize=(10,5))

            # plot eigenspectrum
            plt.subplot(1,2,1)
            #plt.loglog(np.arange(1,n_pc+1), evs.T, 'k', lw=1, alpha=0.2)
            #plt.loglog(np.arange(1,n_pc+1), evs.mean(0), 'r')
            plt.errorbar(np.arange(1,n_pc_curr+1)/n_pc_curr, evs.mean(0), evs.std(0))
            plt.yscale('log'); plt.xscale('log');

            #plot powerspectrum
            plt.subplot(1,2,2)
            #plt.loglog(np.arange(0,0.505,0.005), pows.T, 'k', lw=1, alpha=0.2)
            #plt.loglog(np.arange(0,0.505,0.005), pows.mean(0), 'r')
            plt.errorbar(np.arange(0,61/120, 1/120), pows.mean(0), pows.std(0))
            plt.yscale('log'); plt.xscale('log');

    return eigs, powers, pca_m, space_r2, ft_m, time_r2, pearson_r, spearman_rho, pearson_p, spearman_p


def plot_comp_spectra_corr(subsetsizes, space_r2, time_r2, n_iters, n_pc, f_range, pc_range, eigs, pows):
    """
    Plot Spectra and correlation between components and spectra w/ subset size (goodness of fit)
    """

    n = len(subsetsizes)
    plt.figure(figsize=(10,8))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.inferno(np.linspace(0,1,n)))

    for i, n_i in enumerate(subsetsizes):
        mean_evs = eigs[i]
        mean_pows = pows[i]
        if n_pc == None: #does this still need to be None?  Will it ever be manually changed?
            n_pc_curr = min(subsetsizes)

        elif type(n_pc) == int and n_pc < n_i:
            n_pc_curr = n_pc

        elif type(n_pc) == float:
            n_pc_curr = int(n_pc*n_i)

        #write conditions for pc_range,  use a function, or outside of this
        # [0,None] for whole range, otherwise check if float for fraction
        if pc_range == [0,None]:
            curr_pc_range = [0, int(min(.5*n_pc_curr, .25*max(subsetsizes*n_pc)))]

        #f_range conditions
        if type(f_range[1]) == float:
            curr_f_range = [f_range[0],f_range[1]]

        elif f_range[1] == None:
            curr_f_range = None

        # plot eigenspectrum
        plt.subplot(2,2,1)
        #plt.loglog(np.arange(1,n_pc+1), evs.T, 'k', lw=1, alpha=0.2)
        #plt.loglog(np.arange(1,n_pc+1), evs.mean(0), 'r')
        plt.plot(np.arange(1,n_pc_curr+1)/n_pc_curr, mean_evs)
        plt.yscale('log'); plt.xscale('log');

        #plot powerspectrum
        plt.subplot(2,2,2)
        #plt.loglog(np.arange(0,0.505,0.005), pows.T, 'k', lw=1, alpha=0.2)
        #plt.loglog(np.arange(0,0.505,0.005), pows.mean(0), 'r')
        plt.plot(np.arange(0,61/120, 1/120), mean_pows)
        plt.yscale('log'); plt.xscale('log');

    plt.subplot(2,2,3)
    plt.plot(subsetsizes[:], space_r2.T[:], ".", color = 'purple', lw=1, alpha=0.2)
    plt.title('Correlation between PCs and eigenvalues \n at subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('r\u00b2')
    plt.subplot(2,2,4)
    plt.plot(subsetsizes[:], time_r2.T[:],  ".", color = 'purple', lw=1, alpha=0.2)
    plt.title('Correlation between frequencies and powers \n at subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('r\u00b2')
    plt.tight_layout()
    plt.draw()

def plot_decomp_features(subsetsizes, space_slopes, time_slopes, pearson_corr, spearman_corr, pearson_p, spearman_p):

    plt.figure(figsize=(16, 8))

    #Avg decomp. slope @ subset size
    plt.subplot(2,3,1)
    plt.errorbar(subsetsizes[1:], space_slopes.mean(0)[1:], space_slopes.std(0)[1:], color = 'black', alpha = 0.5)
    #plt.plot(subsetsizes[1:], space_slopes[:,1:].T, 'ok', alpha = 0.5)
    plt.title('average eigenvalue exponent \n at each subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('Slope')

    plt.subplot(2,3,4)
    plt.errorbar(subsetsizes[1:], time_slopes.mean(0)[1:], time_slopes.std(0)[1:], color = 'black', alpha = 0.5)
    #plt.plot(subsetsizes[1:], time_slopes[:,1:].T, 'ok', alpha = 0.5)
    plt.title('average power exponent \n at each subset size')
    plt.xlabel('Subset Size')
    plt.ylabel('Slope')

    #Pearson (R) Correlation value as function of subset size
    n_trials = pearson_corr.shape[0]
    plt.subplot(2,3,2)
    pearson_corr = np.array([pearson_corr[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, pearson_corr.mean(0), pearson_corr.std(0), color = 'blue', alpha = 0.5)
    plt.plot(subsetsizes, pearson_corr.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, pearson_corr, color = 'blue', alpha = 0.5)
    plt.title('Pearson\'s r as function of subset size')
    plt.xlabel('Subset Size')

    #Spearman (Rho) Correlation value as function of subset size
    plt.subplot(2,3,5)
    spearman_corr = np.array([spearman_corr[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, spearman_corr.mean(0), spearman_corr.std(0), color = 'blue', alpha = 0.5)
    plt.plot(subsetsizes, spearman_corr.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, spearman_corr, color = 'blue', alpha = 0.5)
    plt.title('Spearman\'s ρ as function of subset size')
    plt.xlabel('Subset Size')

    #Pearson p values
    plt.subplot(2,3,3)
    pearson_p = np.log10(np.array([pearson_p[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, pearson_p.mean(0), pearson_p.std(0), color = 'green', alpha = 0.5)
    plt.plot(subsetsizes, pearson_p.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle = '--', color = 'orange', lw = 1, alpha = 0.75)
    #plt.semilogy(subsetsizes, pearson_p, color = 'green', alpha = 0.5)
    # plt.yscale('log')
    plt.title('Pearson p value as function of subset size')
    plt.xlabel('Subset Size')

    #Spearman p values
    plt.subplot(2,3,6)
    spearman_p = np.log10(np.array([spearman_p[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, spearman_p.mean(0), spearman_p.std(0), color = 'green', alpha = 0.5)
    plt.plot(subsetsizes, spearman_p.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle = '--', color = 'orange', lw = 1, alpha = 0.75)
    # plt.yscale('log')
    #plt.semilogy(subsetsizes, spearman_p, color = 'green', alpha = 0.5)
    plt.title('Spearman p value as function of subset size')
    plt.xlabel('Subset Size')

    plt.tight_layout()
    plt.draw()
