import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sb
from matplotlib.colors import LinearSegmentedColormap
import cycler
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # for tight_layout() incompatibility with ax object
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


def rc_style(font_size=14):
    """n_c: number of cylcer iters"""
    plt.style.use(CURR_DIR+'/mplrc_notebook')


def colorcycler(color_range, num, default=True):
    cmap = LinearSegmentedColormap.from_list('mycmap', color_range)(np.linspace(0, 1, num))
    if default: # hard codes it in kernel
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', cmap)
    else:
        return cmap


def solo_colorbar(colors, values, label, grad='continuous', orientation='vertical', cax=None):
    """
    colors: list e.g., ['#111d6c', '#e03694']
    range: length 2 list of form [min, max]
    fig: either default plt or plt.figure
    """
    num, range_ = len(values), (values[0], values[-1])
    if grad == 'continuous': # for some reason still looks discrete sometimes
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name='', colors=colors, N=num)
        norm = mpl.colors.Normalize(*range_)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), fraction=0.046, pad=0.04, label=label, orientation=orientation, cax=cax)
    elif grad == 'discrete':
        colors = colorcycler(colors, num, False)
        cmap, norm = mpl.colors.from_levels_and_colors(np.linspace(*range_, num+1), colors)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label=label, format='%.1f', orientation=orientation, cax=cax)


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

def corr_plot(corr, kind, subsetsizes, p_vals=None):
    """if given p_vals, annotates point of p>0.05 with a '*' """
    n_trials = corr.shape[0]
    label_map = {'Pearson':'r','Spearman':'ρ'}
    corr = np.array([corr[i] for i in range(n_trials)], dtype=float)
    plt.errorbar(subsetsizes, corr.mean(0), corr.std(0), color='blue', alpha=0.5)
    plt.plot(subsetsizes, corr.T, 'bo', alpha=0.5)
    #plt.plot(subsetsizes, pearson_r, color = 'blue', alpha = 0.5)
    pltlabel(f'{kind}\'s {label_map[kind]} as function of subset size', 'Subset Size', f'{label_map[kind]}')
    if p_vals.any(): # .any() because array
        x_off = max(subsetsizes)  * .03 # tried and true
        y_off = max(corr.mean(0)) * .1 # doesn't work as well?
        for x, y, p in zip(subsetsizes, corr.mean(0), p_vals.mean(0)):
            if p >= 0.05:
                plt.annotate('*', (x-x_off, y-y_off), size=30)


def p_plot(p_vals, kind, subsetsizes):
    n_trials = p_vals.shape[0]
    p_vals = np.log10(np.array([p_vals[i] for i in range(n_trials)], dtype=float))
    plt.errorbar(subsetsizes, p_vals.mean(0), p_vals.std(0), color='green', alpha=0.5)
    plt.plot(subsetsizes, p_vals.T, 'go', alpha=0.5)
    plt.axhline(np.log10(0.05), linestyle='--', color='orange', lw=2, alpha=0.75, label='p = 0.05')
    plt.legend()
    #plt.semilogy(subsetsizes, p_vals, color = 'green', alpha = 0.5)
    # plt.yscale('log')
    pltlabel(f'{kind} p value as function of subset size', 'Subset Size', '$log_{10}p$')


def goodness_of_fit_plot(subsetsizes, data, spec_er):
    """
    spec_er in: ['pca_er', 'ft_er1', 'ft_er2']
    TODO: make violin plot
    """
    if 'pca' in spec_er:
        title_spec = 'eigenspectrum'
    else:
        title_spec = 'power spectrum'
    plt.plot(subsetsizes[:], data[spec_er].T[:], ".", color='purple', lw=1, alpha=0.2)
    pltlabel(f'Fit error for {title_spec} \n at subset size', 'Subset Size', 'error (MAE)')


def exp_plot(data, key, kind='violin', meta=None):
    """
    Plot exponent distr over % of neurons
    data : data dictionary of standard format
    key  : str in ('pca_m', 'ft_m1', 'ft_m2') 
    kind : str in ('violin', 'hist', 'errorbar')
    # TODO (maybe) fill-between plot type
    *Note: originally done with kind == 'errorbar'
    """
    colors = list(iter(plt.cm.cool(np.linspace(0, 1, 16))))
    if kind == 'violin':
        ax = sb.violinplot(data=data[key], palette=colors, linewidth=0.3, cut=0, inner='box')
        plt.xticks([]) # no xticks necessary because of colorbar
    elif kind == 'hist':
        for e in data[key].T:
            plt.hist(e,bins=np.arange(0.2,1,0.02),histtype='step', label=f'µ: {e.mean():.2f}\nσ: {e.std():.2f}')
    elif kind == 'errorbar':
        subsetsizes = meta['subsetsizes']
        plt.errorbar(subsetsizes[1:], data[key].mean(0)[1:], data[key].std(0)[1:], color='black')
        

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
    subset_fractions = np.linspace(0,1,n)
    cmap = plt.cm.cool(subset_fractions)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap)

    fig = plt.figure(figsize=(8,8))

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
        pltlabel('log Eigenvalue Spectrum', 'PC dimension', 'Variance')
        
        # PSD
        plt.subplot(2,2,3)
        #plt.loglog(np.arange(0,0.505,0.005), pows.T)
        #plt.loglog(np.arange(0,0.505,0.005), pows.mean(0))
        plt.plot(np.arange(0,61/120, 1/120), mean_pows)
        logaxes()
        pltlabel('log Power Spectrum', 'Frequency (Hz)', 'Power')
    
    
    #Space dimension slopes
    plt.subplot(2,2,2)
    exp_plot(data, 'pca_m')
    pltlabel('Eigenvalue spectrum exponent \n at each subset size', '', 'Exponent')

    #Time dimension slopes (SUMMED)
    plt.subplot(2,2,4)
    exp_plot(data, 'ft_m1')
    pltlabel('Power spectrum exponent \n at each subset size', '', 'Exponent')

    # colorbar
    cax = fig.add_axes([1, 0.3, 0.02, 0.35])
    hexes = [mpl.colors.rgb2hex(c) for c in cmap] # VERY hacky way of getting hex values of cmap cool
    solo_colorbar([hexes[0], hexes[-1]], subset_fractions, 'fraction of neurons', orientation='vertical', cax=cax)

    plt.tight_layout()
    plt.draw()


~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
## Analysis summary plots ##
~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
from data_utils import MICE_META, ALL_REGIONS


def _array_sig(data):
    """ 
    Compute if 50% of p values are <= 0.05
    i.e., if a specific mouse region is significantly correlated
    returns bool
    """
    bools = data <= 0.05
    return len([1 for i in bools if i == True]) >= 0.5*len(data)


def avg_corr_bar(data, mice=MICE_META.keys()):
    """
    data: data_dict
    TODO: significance stars for sigbool
    NOTE: x axis plots differently upon import due to ALL_REGIONS being a set
    """
    corr_df = pd.DataFrame(index=ALL_REGIONS, columns=mice) # cols should be ['Mouse 1', 'Mouse 2', 'Mouse 3']
    # populate corr_df
    for m in mice:
        for r in data[m]:
            d = data[m][r]['data']
            mu_psn_r, mu_psn_p,\
            mu_spn_r, mu_spn_p = [d[k].mean(0) for k in ('pearson_corr', 'pearson_p',
                                                        'spearman_corr', 'spearman_p')]
            sig_bool = _array_sig(mu_psn_p) or _array_sig(mu_spn_p)
            ## might want to avg across one type of corr only
            corr_df[m][r] = np.mean([np.nanmean(mu_psn_r), 
                                    np.nanmean(mu_spn_r)]) #avg of avg of the 2 correlations for that region

    # bar plot        
    corr_df.plot.bar(rot=0, color=['#FFABAB','#C5A3FF', '#85E3FF'])
    plt.tick_params(axis="x", which="both", bottom=False)
    plt.xticks(rotation=-45, horizontalalignment='left', fontsize=16)
    for xc in np.arange(.5,10.5,1):
        plt.axvline(x=xc, color='k', linestyle='-', alpha = 0.5, linewidth=.3)
    plt.title('Average Inter-spectral Correlation')


def all_corr_plot(data, mice=MICE_META.keys()):
    """
    TODO: make it start at x=0
    """
    fractions = np.linspace(.0625,1,16)
    ckeys = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ckeys.append('#000000')

    fig = plt.figure(figsize=(13,4))

    for i_m, m in enumerate(mice):
        for r in data[m]:
            d = data[m][r]['data']
            plt.subplot(1,3,i_m+1)
            #plt.plot(fractions, psn_r.mean(0), marker[i_m]+'-', ms=10, color=ckeys[np.where(self.all_regions==region[0])[0][0]], alpha = 0.5)
            plt.plot(fractions, d['pearson_corr'].mean(0), color=ckeys[ALL_REGIONS.index(r)], alpha=0.5, lw=3)
            plt.title('Pearson\'s r (Mouse {})'.format(i_m+1))
            plt.xlim([.0625,1])
            plt.xlabel('Fraction of neurons'); plt.ylabel('r')

    # hack for all legend labels
    for i_r, r in enumerate(ALL_REGIONS):
        plt.plot(1,1/16,color=ckeys[i_r], label=r)

    # sort both labels and handles alpha
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), reverse=True))
    plt.legend(handles, labels, bbox_to_anchor=(1.1, 1.2)) # (x, y)
    plt.tight_layout()


