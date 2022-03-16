import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import cycler
import os


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

########################
### Helper functions ###
########################

def noticks():
    plt.xticks([]);    plt.yticks([])
    

def pltlabel(title, x, y, size=14):
    plt.title(title, fontsize=size)
    plt.xlabel(x, fontsize=size);    plt.ylabel(y, fontsize=size)


def colorcycler(color_range, num):
    cmap = LinearSegmentedColormap.from_list('mycmap', color_range)(np.linspace(0, 1, num))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', cmap)


def rc_style(font_size=14, n_c=None):
    """n_c: number of cylcer iters"""
    plt.style.use(CURR_DIR+'/mplrc_notebook')
    # plt.rcParams['font.font_size'] = font_size
    if n_c:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cool(np.linspace(0,1,n_c)))

##########################
### Analysis functions ###
##########################

def corr_plot(corr_data, kind, subsetsizes, n_trials):
    label_map = {'Pearson':'r','Spearman':'ρ'}
    corr_data = np.array([corr_data[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, corr_data.mean(0), corr_data.std(0), color='blue', alpha=0.5)
    plt.plot(subsetsizes, corr_data.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, pearson_corr, color = 'blue', alpha = 0.5)
    plt.title(f'{kind}\'s {label_map[kind]} as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel(f'{label_map[kind]}')


def p_plot(p_data, kind, subsetsizes, n_trials):
    """p value plot"""
    p_data = np.log10(np.array([p_data[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, p_data.mean(0), p_data.std(0), color='green', alpha=0.5)
    plt.plot(subsetsizes, p_data.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle='--', color='orange', lw=2, alpha=0.75, label='p = 0.05')
    plt.legend()
    #plt.semilogy(subsetsizes, p_data, color = 'green', alpha = 0.5)
    # plt.yscale('log')
    plt.title(f'{kind} p value as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel('$log_{10}p$') 


def plot_all_measures(data, meta):
    """
    meta should have: 'n_iters', 'n_pc', 'f_range', 'subsetsizes', 'pc_range'
    """
    subsetsizes = data['subsetsizes']
    n_pc = data['n_pc']
    n = len(subsetsizes)
    #stylistic details
    rc_style(n_c=n)
    plt.figure(figsize=(23,15))

    #plot spectra
    for i, n_i in enumerate(subsetsizes):
        mean_evs = data['eigs'][i]
        mean_pows = data['pows'][i]
        if n_pc == None: #does this still need to be None?  Will it ever be manually changed?
            n_pc_curr = min(subsetsizes)

        elif type(n_pc) == int and n_pc < n_i:
            n_pc_curr = n_pc

        elif type(n_pc) == float:
            n_pc_curr = int(n_pc*n_i)

        # plot eigenspectrum
        plt.subplot(4,5,1)
        #plt.loglog(np.arange(1,n_pc+1), evs.T, 'k', lw=1, alpha=0.2)
        #plt.loglog(np.arange(1,n_pc+1), evs.mean(0), 'r')
        plt.plot(np.arange(1,n_pc_curr+1)/n_pc_curr, mean_evs) #KEEP THIS LINE: proportion of PCs
        plt.yscale('log'); plt.xscale('log');
        plt.title('Eigenvalue Spectrum')
        plt.xlabel('PC dimension'); plt.ylabel('Variance')
        #plot powerspectrum
        plt.subplot(4,5,6)
        #plt.loglog(np.arange(0,0.505,0.005), pows.T, 'k', lw=1, alpha=0.2)
        #plt.loglog(np.arange(0,0.505,0.005), pows.mean(0), 'r')
        plt.plot(np.arange(0,61/120, 1/120), mean_pows)
        plt.yscale('log'); plt.xscale('log')
        plt.title('Power Spectrum')
        plt.xlabel('Frequency (Hz)'); plt.ylabel('Power')

    #Space dimension slopes
    plt.subplot(4,5,2)
    plt.errorbar(subsetsizes[1:], data['espec_exp'].mean(0)[1:], data['espec_exp'].std(0)[1:], color='black', alpha=0.5)
    plt.title('Average eigenvalue spectrum exponent \n at each subset size')
    plt.xlabel('Subset Size');      plt.ylabel('Exponent')

    #Time dimension slopes
    plt.subplot(4,5,7)
    plt.errorbar(subsetsizes[1:], data['psd_exp1'].mean(0)[1:], data['psd_exp1'].std(0)[1:], color='black', alpha=0.5)
    plt.title('Average power spectrum exponent \n at each subset size')
    plt.xlabel('Subset Size');      plt.ylabel('Exponent')

    #PCA goodness of fit
    plt.subplot(4,5,3)
    plt.plot(subsetsizes[:], data['pca_er'].T[:], ".", color = 'purple', lw=1, alpha=0.2)
    plt.title('Fit error for eigenspectrum \n at subset size')
    plt.xlabel('Subset Size');      plt.ylabel('error')

    #FFT goodness of fit
    plt.subplot(4,5,8)
    plt.plot(subsetsizes[:], data['ft_er1'].T[:],  ".", color = 'purple', lw=1, alpha=0.2)
    plt.title('Fit error for power spectrum \n at subset size')
    plt.xlabel('Subset Size');      plt.ylabel('error')

    n_trials = data['pearson_corr1'].shape[0]
    #Pearson (R) Correlation value as function of subset size
    plt.subplot(4,5,4)
    corr_plot(data['pearson_corr1'], 'Pearson', subsetsizes, n_trials)

    #Spearman (Rho) Correlation value as function of subset size
    plt.subplot(4,5,9)
    corr_plot(data['spearman_corr1'], 'Spearman', subsetsizes,n_trials)

    #Pearson p values
    plt.subplot(4,5,5)
    p_plot(data['pearson_p1'], 'Pearson', subsetsizes, n_trials)

    #Spearman p values
    plt.subplot(4,5,10)
    p_plot(data['spearman_p1'], 'Spearman', subsetsizes, n_trials)

    ## NOSUM
    plt.subplot(4,5,11)
    plt.plot() #SPECTRUM

    #Space dimension slopes
    plt.subplot(4,5,12)
    plt.plot() 

    #Time dimension slopes
    plt.subplot(4,5,17)
    plt.errorbar(subsetsizes[1:], data['psd_exp2'].mean(0)[1:], data['psd_exp2'].std(0)[1:], color='black', alpha=0.5)
    plt.title('Average power spectrum exponent \n at each subset size')
    plt.xlabel('Subset Size');      plt.ylabel('Exponent')

    #PCA goodness of fit
    plt.subplot(4,5,13)
    plt.plot()

    #FFT goodness of fit
    plt.subplot(4,5,18)
    plt.plot(subsetsizes[:], data['ft_er2'].T[:],  ".", color = 'purple', lw=1, alpha=0.2)
    plt.title('Fit error for power spectrum \n at subset size')
    plt.xlabel('Subset Size');      plt.ylabel('error')


    #Pearson (R) Correlation value as function of subset size
    plt.subplot(4,5,14)
    corr_plot(data['pearson_corr2'], 'Pearson', subsetsizes, n_trials)

    #Spearman (Rho) Correlation value as function of subset size
    plt.subplot(4,5,19)
    corr_plot(data['spearman_corr2'], 'Spearman', subsetsizes, n_trials)

    #Pearson p values
    plt.subplot(4,5,15)
    p_plot(data['pearson_p2'], 'Pearson', subsetsizes, n_trials)

    #Spearman p values
    plt.subplot(4,5,20)
    p_plot(data['spearman_p2'], 'Spearman', subsetsizes, n_trials)

    plt.tight_layout()
    plt.draw()