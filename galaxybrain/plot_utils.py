"""
NOTE:  nans in the correlations because once you are sampling entire population it should be nan
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
import seaborn as sb
import cycler

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
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', color_range)(np.linspace(0, 1, num))
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
FRACTIONS = np.linspace(.0625,1,16)


def corr_plot(corr, kind, xvals=FRACTIONS, p_vals=None, ax=plt):
    """if given p_vals, annotates point of p<0.05 with a '*' 
    xvals used to be subsetsizes"""
    n_trials = corr.shape[0]
    label_map = {'Pearson':'r','Spearman':'ρ'}
    corr = np.array([corr[i] for i in range(n_trials)], dtype=float)
    ax.errorbar(xvals, corr.mean(0), corr.std(0), color='blue', alpha=0.5)
    ax.plot(xvals, corr.T, 'bo', alpha=0.5, markersize=8, lw=5)
    #ax.plot(subsetsizes, pearson_r, color = 'blue', alpha = 0.5)
    pltlabel(f'{kind}\'s {label_map[kind]} as function of subset size', 'fraction sampled', f'{label_map[kind]}')
    if p_vals.any(): # .any() because array
        #TODO need to find a better way to scale location of marker
        x_off = max(xvals)  * .02
        y_off = max(corr.mean(0)) * .1 # doesn't work as well?
        for x, y, p in zip(xvals, corr.mean(0), p_vals.mean(0)):
            if p < 0.05:
                ax.annotate('*', (x-x_off, y-y_off), size=60)


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


def exp_plot(data, key, kind='violin', meta=None, ax=plt):
    """
    Plot exponent distr over % of neurons
    data : data dictionary of standard format
    key  : str in ('es_exponent', 'psd_exponent1', 'psd_exponent2') 
    kind : str in ('violin', 'hist', 'errorbar')
    # TODO (maybe) fill-between plot type
    *Note: originally done with kind == 'errorbar'
    """
    colors = list(iter(plt.cm.cool(np.linspace(0, 1, 16))))
    if kind == 'violin':
        ax = sb.violinplot(data=data[key], palette=colors, linewidth=0.3, cut=0, inner='box')
        # ax.xticks([]) # no xticks necessary because of colorbar
        ax.set_xticks([], minor=False)
    elif kind == 'hist':
        for e in data[key].T:
            ax.hist(e,bins=np.arange(0.2,1,0.02),histtype='step', label=f'µ: {e.mean():.2f}\nσ: {e.std():.2f}')
    elif kind == 'errorbar':
        subsetsizes = meta['subsetsizes']
        ax.errorbar(subsetsizes[1:], data[key].mean(0)[1:], data[key].std(0)[1:], color='black')
        
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def plot_all_measures(data, meta, kind='mouse', title=''):
    """
    data has keys 'data', 'meta'
    meta is for all datasets and has keys: 'n_iters', 'n_pc', 'f_range', 'subsetsizes', 'pc_range'
    kind: 'mouse' (includes sum & nonsum PSD data), 'mouse_old', or 'sim' (ising or shuffle)
    plot layout:
    [ ES  ]  [ subset size vs ES exponent  ]   [ correlation ]
    [ PSD ]  [ subset size vs PSD exponent ]
    """

    subsetsizes = data['meta']['subsetsizes'] # since every sub data has its own meta
    data = data['data']
    n_pc = meta['n_pc']
    n = len(subsetsizes)
    dsuffix = '1'
    ## Plot style
    fig=plt.figure(figsize=(19,8)) 
    subset_fractions = np.linspace(0,1,n)
    cmap = plt.cm.cool(subset_fractions)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap)

    gs = GridSpec(2,3, width_ratios=[1,1,4], wspace=.8) #
    gs1 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:,:2], hspace=0.5, wspace=0.3)
    gs2 = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[:,2])

    ## Spectra
    for (ip, spec), labs in zip(enumerate(['eigs', 'pows']), [['log ES', 'PC dimension', 'Variance'],
                                                            ['log PSD', 'Frequency (Hz)', 'Power']]):
        ax = fig.add_subplot(gs1[ip,0])
        for i, n_i in enumerate(subsetsizes):
            if isinstance(n_pc, int) and n_pc < n_i:
                n_pc_curr = n_pc
            elif isinstance(n_pc, float):
                n_pc_curr = int(n_pc*n_i)
            print(spec)
            xvals = np.arange(1,n_pc_curr+1)/n_pc_curr if spec == 'eigs'\
            else np.arange(0,61/120, 1/120)
            # Eigenspectrum
            ax.plot(xvals, data[spec][i]) #KEEP THIS LINE: proportion of PCs
            logaxes()
            pltlabel(*labs)

    ## Exponent distributions
    for (ip, exp), labs in zip(enumerate(['es_exponent', 'psd_exponent'+dsuffix]), [['ES exponent \n at each subset size', '', 'Exponent'],
                                                            ['PSD exponent \n at each subset size', '', 'Exponent']]):
        ax = fig.add_subplot(gs1[ip,1])
        exp_plot(data, exp, ax=ax)
        pltlabel(*labs)
        
    ## colorbar for first two cols
    cax = fig.add_axes([0.45, 0.3, 0.01, 0.35])
    hexes = [mpl.colors.rgb2hex(c) for c in cmap] # VERY hacky way of getting hex values of cmap cool
    solo_colorbar([hexes[0], hexes[-1]], subset_fractions, 'fraction sampled', 
                orientation='vertical', cax=cax)

    ## Interspec Correlation
    ax = fig.add_subplot(gs2[:])
    corr_plot(data['pearson_corr'+dsuffix], 'Pearson',
            p_vals=data['pearson_p'+dsuffix], ax=ax)

    plt.suptitle(title)
    plt.tight_layout()
    #TODO reset color map


~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
## Analysis summary plots ##
~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
from .data_utils import MICE_META, ALL_REGIONS
MARKERS   = 'o*^' # one for each mouse

def _array_sig(data):
    """ 
    Compute if 50% of p values are < 0.05
    i.e., if a specific mouse region is significantly correlated
    returns bool
    """
    bools = data < 0.05
    return len([1 for i in bools if i == True]) >= 0.5*len(data)


def _ckeys():
    ckeys = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ckeys.append('#000000')
    return ckeys


def _sorted_lgnd(xy=(1,1)):
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), reverse=True))
    plt.legend(handles, labels, bbox_to_anchor=xy) # (x, y)


def avg_corr_bar(data, mice=MICE_META.keys()):
    """
    data: data_dict
    TODO: significance stars for sigbool
    https://stackoverflow.com/questions/40489821/how-to-write-text-above-the-bars-on-a-bar-plot-python/40491960
    https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    https://matplotlib.org/examples/api/barchart_demo.html
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


def all_corr_plot(data, mice=MICE_META.keys(), corr_type='pearson'):
    """
    1 x n_mice line plot of correlation over subsets
    TODO: make it start at x=0
    """
    ckeys = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ckeys.append('#000000')

    fig = plt.figure(figsize=(13,4))

    for i_m, m in enumerate(mice):
        for r in data[m]:
            d = data[m][r]['data']
            plt.subplot(1,3,i_m+1)
            plt.plot(FRACTIONS, d[f'{corr_type}_corr'].mean(0), color=ckeys[ALL_REGIONS.index(r)], alpha=0.5, lw=3) # marker[i_m]+'-', ms=10
            plt.title('Pearson\'s r (Mouse {})'.format(i_m+1))
            plt.xlim([.0625,1])
            plt.ylabel('r')

    fig.supxlabel('fraction sampled', x=.45, y=.1)

    # hack for all legend labels
    for i_r, r in enumerate(ALL_REGIONS):
        plt.plot(1,1/16,color=ckeys[i_r], label=r)

    # sort both labels and handles alpha
    _sorted_lgnd(xy=(1.1,1.2))
    plt.tight_layout()


def corr_heat_map(data, mice=MICE_META.keys(), corr_type='pearson'):
    """Nx1 heat map"""
    heat_maps = [pd.DataFrame(index=ALL_REGIONS, columns=FRACTIONS) for i in range(len(mice))]
    for i_m, m in enumerate(mice):
        for r in data[m]:
            d = data[m][r]['data']
            for i_f, f in enumerate(FRACTIONS):
                heat_maps[i_m][f][r] = d[f'{corr_type}_corr'].mean(0)[i_f]

    plt.figure(figsize=(15,4))
    for i_m in range(len(mice)):
        plt.subplot(1,3,i_m+1)
        h_map = heat_maps[i_m]
        cmap = mpl.cm.bwr.set_bad(color='#ababab')
        im = plt.imshow(h_map.fillna(np.nan), 'bwr', extent=[.0625,1,-1,1], aspect='auto')
        plt.xlabel('fraction sampled')
        # NOTE: used to be linspace to 12? 
        plt.yticks(ticks=np.linspace(-1,1,11), verticalalignment='baseline', labels=np.flip(h_map.index), fontsize=16)
        plt.title('Mouse {}'.format(i_m + 1))
        plt.colorbar(im,fraction=0.046, pad=0.04)
            
        plt.tight_layout()


def avg_exp_plot(data, mice=MICE_META.keys()):
    """
    data : data_dict
    """
    title_lu = {'es_exponent':'ES exponent', 'psd_exponent':'PSD exponent'}
    ckeys = _ckeys()
    fig = plt.figure(figsize=(9,4))
    for i, m in enumerate(mice):
        for r in data[m]:
            d = data[m][r]['data']
            for pi, exp in enumerate(['es_exponent', 'psd_exponent']):
                plt.subplot(1,2,pi+1)
                plt.plot(FRACTIONS, d[exp].mean(0), MARKERS[i]+'-', ms=10, color=ckeys[ALL_REGIONS.index(r)], alpha=0.6)
                plt.xscale('log')
                plt.ylabel(title_lu[exp])
                if exp == 'psd_exponent': # TODO: shouldn't have to do this manually
                    plt.ylim([-.023,-0.006])
    
    for i_r, r in enumerate(ALL_REGIONS):
        plt.plot(0,1,color=ckeys[i_r], label=r)

    fig.supxlabel('fraction sampled', x=.45, y=.1)
    _sorted_lgnd(xy=(1,1.1))
    plt.tight_layout()
        

~~~~~~~~~~~~~~~~~~~0
# Ising Model Plots 
~~~~~~~~~~~~~~~~~~~0

TEMP_COLOR_RANGE = ['#2186cf', '#ad4b59'] # b to r (cold to hot)
CRIT_T           = f'{2.27:.2f}' # as float
CRIT_C           = 'k' 

def plot_ising_spectra(data, spec, temps='all', subset_ix=15, ax=plt):
    """plot spectra over temps given
    spec        : 'eigs' or 'pows'
    temps       : list of str temps or 'all'
    subset_ix : index in range [0, 16) """
    if temps == 'all':
        temps = [k for k in data if k != 'meta']

    data = {k : data[k] for k in temps} # filter
    colors = colorcycler(TEMP_COLOR_RANGE, len(data), False)
    colors[temps.index(CRIT_T)] = mpl.colors.to_rgba(CRIT_C)
    
    for t, c in zip(data, colors):
        # conditionals for plot
        lw    =  4  if t==CRIT_T else 2
        alpha =  1  if t==CRIT_T else 0.9
        try:
            ax.plot(data[t]['data'][spec][subset_ix], color=c, lw=lw, alpha=alpha)
        except:
            print(t, spec, subset_ix)
        logaxes()


def measure_over_temps(data, data_key, temps, ax=plt, colorbar=False):
    """
    Plot % sampled vs a measure (data_key) over multiple temperatures
    data_key: data for each temperature 
    """
    data = [data[t]['data'][data_key].mean(0) for t in temps]
    colors = colorcycler(TEMP_COLOR_RANGE, len(temps), False)
    colors[temps.index(CRIT_T)] = mpl.colors.to_rgba(CRIT_C)
    for r, t, c in zip(data, temps, colors):
        lw    =  4  if t == CRIT_T else 2
        alpha =  1  if t == CRIT_T else 0.9
        ax.plot(FRACTIONS, r, color=c, lw=lw, alpha=alpha)
    plt.xlim([FRACTIONS[0], FRACTIONS[-1]])
    if colorbar:
        ftemps = list(map(float, temps)) #need float for colorbar
        cmap, norm = mpl.colors.from_levels_and_colors(ftemps, colors[1:] ) # weird indexing offset by 1
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='temperature')