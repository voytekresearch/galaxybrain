import matplotlib.pyplot as plt
import numpy as np

def corr_plot(corr_data, kind, subsetsizes, n_trials):
    label_map = {'Pearson':'r','Spearman':'ρ'}
    corr_data = np.array([corr_data[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, corr_data.mean(0), corr_data.std(0), color = 'blue', alpha = 0.5)
    plt.plot(subsetsizes, corr_data.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, pearson_corr, color = 'blue', alpha = 0.5)
    plt.title(f'{kind}\'s {label_map[kind]} as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel(f'{label_map[kind]}')

def p_plot(p_data, kind, subsetsizes, n_trials):
    p_data = np.log10(np.array([p_data[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, p_data.mean(0), p_data.std(0), color = 'green', alpha = 0.5)
    plt.plot(subsetsizes, p_data.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle = '--', color = 'orange', lw = 2, alpha = 0.75, label = 'p = 0.05')
    plt.legend()
    #plt.semilogy(subsetsizes, p_data, color = 'green', alpha = 0.5)
    # plt.yscale('log')
    plt.title(f'{kind} p value as function of subset size')
    plt.xlabel('Subset Size'); plt.ylabel('$log_{10}p$') 

def init_plt_style(n):
    '''n: number of cylcer iters'''
    plt.rcParams['mathtext.default'] = 'regular'

    font = {'family' : 'Arial',
       'weight' : 'regular',
       'size'   : 14}
    
    plt.rc('font', **font)
    plt.rcParams['axes.spines.top']=False
    plt.rcParams['axes.spines.right']=False
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cool(np.linspace(0,1,n)))

def plot_all_measures(data):
    """
    """
    subsetsizes = data['subsetsizes']
    n_pc = data['n_pc']
    n = len(subsetsizes)
    #stylistic details
    init_plt_style(n)
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
        plt.yscale('log'); plt.xscale('log');
        plt.title('Power Spectrum')
        plt.xlabel('Frequency (Hz)'); plt.ylabel('Power')

    #Space dimension slopes
    plt.subplot(4,5,2)
    plt.errorbar(subsetsizes[1:], data['espec_exp'].mean(0)[1:], data['espec_exp'].std(0)[1:], color = 'black', alpha = 0.5)
    plt.title('Average eigenvalue spectrum exponent \n at each subset size')
    plt.xlabel('Subset Size');      plt.ylabel('Exponent')

    #Time dimension slopes
    plt.subplot(4,5,7)
    plt.errorbar(subsetsizes[1:], data['psd_exp1'].mean(0)[1:], data['psd_exp1'].std(0)[1:], color = 'black', alpha = 0.5)
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
    plt.errorbar(subsetsizes[1:], data['psd_exp2'].mean(0)[1:], data['psd_exp2'].std(0)[1:], color = 'black', alpha = 0.5)
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