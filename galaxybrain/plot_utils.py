import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sb
from matplotlib.colors import LinearSegmentedColormap
import cycler
import os


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

########################
### Helper functions ###
########################

def noticks():
    plt.xticks([]);    plt.yticks([])
    

def pltlabel(title='', x='', y='', size=14):
    plt.title(title, fontsize=size)
    plt.xlabel(x, fontsize=size);    plt.ylabel(y, fontsize=size)


def logaxes():
    plt.yscale('log');    plt.xscale('log')


def colorcycler(color_range, num, default=True):
    cmap = LinearSegmentedColormap.from_list('mycmap', color_range)(np.linspace(0, 1, num))
    if default: # hard codes it in kernel
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', cmap)
    else:
        return cmap


def rc_style(font_size=14, n_c=None):
    """n_c: number of cylcer iters"""
    plt.style.use(CURR_DIR+'/mplrc_notebook')
    # plt.rcParams['font.font_size'] = font_size
    if n_c:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cool(np.linspace(0,1,n_c)))


def solo_colorbar(colors, values, label):
    """
    colors: list e.g., ['#111d6c', '#e03694']
    range: length 2 list of form [min, max]
    """
    num, range_ = len(values), (values[0], values[-1])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name='',
                                            colors=colors, N=num)
    norm = mpl.colors.Normalize(*range_)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), fraction=0.046, pad=0.04, label=label)


def silent_plot(fx, fx_args, fn):
    """
    save plot directly into directory
        fx: plotting func
        fx_args: data for fx
        fn: filename
     """
    fx(*fx_args)
    plt.savefig(fn)
    plt.close('all')
    plt.pause(0.01)

##########################
### Analysis functions ###
##########################

def corr_plot(corr_data, kind, subsetsizes, p_data=None):
    """if given p_data, annotates point of p>0.05 with a '*' """
    n_trials = corr_data.shape[0]
    label_map = {'Pearson':'r','Spearman':'ρ'}
    corr_data = np.array([corr_data[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, corr_data.mean(0), corr_data.std(0), color='blue', alpha=0.5)
    plt.plot(subsetsizes, corr_data.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, pearson_r, color = 'blue', alpha = 0.5)
    pltlabel(f'{kind}\'s {label_map[kind]} as function of subset size', 'Subset Size', f'{label_map[kind]}')
    if p_data.any(): # .any() because array
        x_offset = subsetsizes[-1] * .03 # tried and true
        for x, y, p in zip(subsetsizes, corr_data.mean(0), p_data.mean(0)):
            if p >= 0.05:
                plt.annotate('*', (x-x_offset,y), size=30)


def p_plot(p_data, kind, subsetsizes):
    n_trials = p_data.shape[0]
    p_data = np.log10(np.array([p_data[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, p_data.mean(0), p_data.std(0), color='green', alpha=0.5)
    plt.plot(subsetsizes, p_data.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle='--', color='orange', lw=2, alpha=0.75, label='p = 0.05')
    plt.legend()
    #plt.semilogy(subsetsizes, p_data, color = 'green', alpha = 0.5)
    # plt.yscale('log')
    pltlabel(f'{kind} p value as function of subset size', 'Subset Size', '$log_{10}p$')


def goodness_of_fit_plot(subsetsizes, data, spec):
    """spec in: ['pca_er', 'ft_er1', 'ft_er2']"""
    if 'pca' in spec:
        title_spec = 'eigenspectrum'
    else:
        title_spec = 'power spectrum'
    plt.plot(subsetsizes[:], data[spec].T[:], ".", color='purple', lw=1, alpha=0.2)
    pltlabel(f'Fit error for {title_spec} \n at subset size', 'Subset Size', 'error')


def plot_all_measures(data, meta):
    """
    meta should have: 'n_iters', 'n_pc', 'f_range', 'subsetsizes', 'pc_range'

    plot layout:
    [ ES  ]  [ subset size vs ES exponent  ]
    [ PSD ]  [ subset size vs PSD exponent ]
    """
    subsetsizes = meta['subsetsizes']
    n_pc = meta['n_pc']
    n = len(subsetsizes)
    #stylistic details
    rc_style(n_c=n)
    plt.figure(figsize=(8,8))

    #plot spectra
    for i, n_i in enumerate(subsetsizes):
        mean_evs  = data['eigs'][i]
        mean_pows = data['pows'][i]
        if n_pc == None: #does this still need to be None?  Will it ever be manually changed?
            n_pc_curr = min(subsetsizes)
        elif isinstance(n_pc, int) and n_pc < n_i:
            n_pc_curr = n_pc
        elif isinstance(n_pc, float):
            n_pc_curr = int(n_pc*n_i)

        # Eigenspectrum
        plt.subplot(2,2,1)
        #plt.loglog(np.arange(1,n_pc+1), evs.T)
        #plt.loglog(np.arange(1,n_pc+1), evs.mean(0))
        plt.plot(np.arange(1,n_pc_curr+1)/n_pc_curr, mean_evs) #KEEP THIS LINE: proportion of PCs
        logaxes()
        pltlabel('Eigenvalue Spectrum', 'PC dimension', 'Variance')
        
        # PSD
        plt.subplot(2,2,3)
        #plt.loglog(np.arange(0,0.505,0.005), pows.T)
        #plt.loglog(np.arange(0,0.505,0.005), pows.mean(0))
        plt.plot(np.arange(0,61/120, 1/120), mean_pows)
        logaxes()
        pltlabel('Power Spectrum', 'Frequency (Hz)', 'Power')

    #Space dimension slopes
    plt.subplot(2,2,2)
    plt.errorbar(subsetsizes[1:], data['pca_m'].mean(0)[1:], data['pca_m'].std(0)[1:], color='black', alpha=0.5)
    pltlabel('Average eigenvalue spectrum exponent \n at each subset size', 'Subset Size', 'Exponent')

    #Time dimension slopes
    plt.subplot(2,2,4)
    plt.errorbar(subsetsizes[1:], data['ft_m1'].mean(0)[1:], data['ft_m1'].std(0)[1:], color='black', alpha=0.5)
    pltlabel('Average power spectrum exponent \n at each subset size', 'Subset Size', 'Exponent')

    plt.tight_layout()
    plt.draw()


def plot_dist(data, rgn, exp_kind, title, kind='violin'):
    '''Plot distirbution either with violin or hist
    data: data dictionary of standard format'''
    colors = list(iter(plt.cm.cool(np.linspace(0, 1, 16))))
    mouse = 'krebs'
    if kind == 'violin':
        sb.violinplot(data=data[mouse][rgn]['data'][f'psd_exp{exp_kind}'], palette=colors, linewidth=0.3)
    else:
        for e in data[mouse][rgn]['data'][f'psd_exp{exp_kind}'].T:
            plt.hist(e,bins=np.arange(0.2,1,0.02),histtype='step', label=f'µ: {e.mean():.2f}\nσ: {e.std():.2f}')
        if title=='Summed':
            plt.legend(loc='center right', bbox_to_anchor=(0, 0.5))
        else:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    plt.title(title)