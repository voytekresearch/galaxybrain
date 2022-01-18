import numpy as np
from scipy import io, signal
import pandas as pd
import sys, os
import json

sys.path.append('../')
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import ramsey
import plot_utils

import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
matplotlib.rc('figure', max_open_warning = 0)

MICE_REGIONS = {'krebs': {'all': 1462,'CP': 176,'HPF': 265,'LS': 122,'MB': 127,'TH': 227,'V1': 334},
                'robbins': {'all': 2688,  'FrMoCtx': 647,  'HPF': 333,  'LS': 133,  'RSP': 112,  'SomMoCtx': 220,  'TH': 638,  'V1': 251,  'V2': 124}, 
                'waksman': {'all': 2296, 'CP': 134, 'HPF': 155, 'TH': 1878}}

def load_mouse_data(datafolder, i_m, return_type='binned', bin_width=0.01, smooth_param=[0.2, 0.025]):
    """ Load neuropixel data from Stringer et al., 2019

    Parameters
    ----------
    datafolder : str
        Location of data.
    i_m : int
        Which mouse (0,1,2)
    return_type : str
        Return as 'spiketimes', 'binned' (default), or 'smoothed'.
    bin_width : float
        Bin width in seconds, defaults 0.01s.
    smooth_param : [float, float]
        Smoothing parameter for Gaussian window, [win_len, std].

    Returns
    -------
        data, cluster_info.
    """
    mice = ['Krebs', 'Waksman', 'Robbins']
    cluLocs = io.loadmat(datafolder+'cluLoc.mat', squeeze_me = True)
    probe_id = cluLocs['iprobeAll']
    probe_loc = cluLocs['brainLocNameAll']

    # load data and cluster info
    mouse_data = io.loadmat(datafolder+'spks/spks%s_Feb18.mat'%mice[i_m], squeeze_me = True)['spks']
    clu_info = pd.DataFrame(np.array([probe_id[i_m],probe_loc[i_m]]).T, columns=['probe', 'region'])

    print('Grabbing Spikes...')
    spikes_all = []
    for probe in range(len(mouse_data)):
        st = mouse_data[probe][0]
        clu = mouse_data[probe][1]
        # add spike time into each
        spikes_all += [np.sort(st[clu==k]) for k in np.unique(clu)]

    if return_type is 'spiketimes':
        return spikes_all, clu_info

    print('Binning Spikes...')
    t_beg, t_end = _beg_end(spikes_all)
    t_bins = np.arange(t_beg,t_end,bin_width)
    spk_binned = [np.histogram(spks,t_bins)[0] for spks in spikes_all]
    df_spk = pd.DataFrame(np.array(spk_binned).T, index=t_bins[:-1])

    # compute populate rate in various areas
    pop_rate = df_spk.sum(1)
    for reg, grp in clu_info.groupby('region'):
        df_spk.insert(0,reg,df_spk[grp.index].sum(1))
    df_spk.insert(0,'all',pop_rate)

    if return_type is 'binned':
        return df_spk, clu_info

    if return_type is 'smoothed':
        print('Smoothing...')
        win_len, win_std = smooth_param
        win = signal.windows.gaussian(int(win_len/bin_width)+1, win_std/bin_width)
        win/=win.sum()
        bin_smoothed = signal.convolve(df_spk, win[:,None], mode='same')
        df_spk_smo = pd.DataFrame(bin_smoothed, index=t_bins[:-1])
        return df_spk_smo, clu_info

def return_pops(data, df_info):
    """ Return different populations in a list.

    Parameters
    ----------
    data : matrix or dataframe, time x neuron.
    df_info : dataframe
        Cluster information for which region each cluster falls into.

    Returns
    -------
    list
        List of spiketime matrices, grouped based on region.

    """
    pop_list, region_labels = [], []
    for reg, grp in df_info.groupby('region'):
        if type(data) is type(pd.DataFrame()):
            # dataframe is passed in, all good
            pop_list.append(np.squeeze(data[grp.index.values].values))
        else:
            # assume the group indices line up with the data array indices
            pop_list.append(np.squeeze(data[grp.index.values]))
        region_labels.append(reg)
    return pop_list, region_labels

def _beg_end(spikes_all):
    spikes_concat = np.concatenate(spikes_all)
    return np.floor(spikes_concat.min()), np.ceil(spikes_concat.max())

def spike_dict(mice_ix=range(3)):
    """
    Uses load_mouse_data to return a dictionary of mouse data and region info
    args:
        mice_ix: list of mouse specific indices (0,1,2)
        
    Returns:
        Dict of form: {mouse_name : ( (spike_dataframe, region_indices_dict), 
                                    {region1:count, region2:count...} )
                        }
        See mouse_iter() for use case
        """
    
    datafolder = '../data/spikes/'

    region_sizes = ( [('all', 1462), ('CP', 176), ('HPF', 265), ('LS', 122), ('MB', 127), ('TH', 227), ('V1', 334)], #krebs
                     [('all', 2688), ('FrMoCtx', 647), ('HPF', 333), ('LS', 133), ('RSP', 112), ('SomMoCtx', 220), ('TH', 638), ('V1', 251), ('V2', 124)], #robbins
                     [('all', 2296), ('CP', 134), ('HPF', 155), ('TH', 1878)] ) #waksman
    raster_dict = {}
    for i_m, name in zip(mice_ix, ('krebs', 'robbins','waksman')):
        print(f'Mouse {i_m+1}')
        df_spk, df_info = load_mouse_data(datafolder, i_m, return_type='binned', bin_width=1)
        region_indices = {}
        for region in df_info.region.unique():
            region_indices.update({region:np.where(df_info['region'] == str(region))[0]})
            
        spk_list, region_labels = return_pops(df_spk, df_info)
        print(list(zip(region_labels, [s.shape[1] for s in spk_list])), 'Total:',sum([s.shape[1] for s in spk_list]))
        su_start_ind = len(region_labels)+1
        raster_dict[name] = ( (df_spk[df_spk.columns[su_start_ind:]], region_indices), region_sizes[i_m] )

    return raster_dict

def mouse_iter(raster_dict, mouse_key, burn_in):
    """Yields mouse spikes in an oft used loop"""
    mouse = raster_dict[mouse_key][0]
    for region in raster_dict[mouse_key][1]:
        region_name, region_count = region[0], region[1]
        print(region_name)
        if region_name == 'all':
            mouse_raster = mouse[0].iloc[burn_in:-burn_in]
        else:
            mouse_raster = mouse[0][mouse[1][region_name]].iloc[burn_in:-burn_in]
            
        yield np.array(mouse_raster), region_name, region_count

#####################################
### Analysis result data handling ###
#####################################

def load_npz(file, kind, decomp_arr):
    """
    Designated for easier loading of the saved Ramsey data
    file typically looks like f'../data/experiments/CP/ramsey_{i+1}.npz'
    sim/noise data don't have: b, summed data (#2 suffix)
        and decomp_arr appends: [pca_m, ft_m, psn_r, spn_r, psn_p, spn_p]
    """
    if kind == 'mouse':
        with np.load(file, allow_pickle=True) as data:
            eigs = data['eigs']
            pows = data['pows']
            pca_m = data['pca_m']
            pca_er = data['pca_er']
            pca_b = data['pca_b']
            ft_m1 = data['ft_m1']
            ft_er1 = data['ft_er1']
            ft_b1 = data['ft_b1']
            ft_m2 = data['ft_m2']
            ft_er2 = data['ft_er2']
            ft_b2 = data['ft_b2']
            pearson_r1 = data['pearson_r1']
            spearman_rho1 = data['spearman_rho1']
            pearson_p1 = data['pearson_p1']
            spearman_p1 = data['spearman_p1']
            pearson_r2 = data['pearson_r2']
            spearman_rho2 = data['spearman_rho2']
            pearson_p2 = data['pearson_p2']
            spearman_p2 = data['spearman_p2']

        decomp_arr.append([pca_m, ft_m1, ft_m2, 
                        pearson_r1, spearman_rho1, pearson_p1, spearman_p1,
                        pearson_r2, spearman_rho2, pearson_p2, spearman_p2,
                            pca_b, ft_b1, ft_b2])
        return eigs, pows, pca_er, pca_b, ft_er1, ft_b1, ft_er2, ft_b2, decomp_arr
    
    elif kind == 'mouse_old': # "old" data format with different field names 
        with np.load(file, allow_pickle=True) as data:
            eigs = data['eigs']
            pows = data['pows']
            space_r2 = data['space_r2']
            time_r2 = data['time_r2']
            pca_m = data['pca_m']
            ft_m = data['ft_m']
            psn_r = data['pearson_r']
            spn_r = data['spearman_rho']
            psn_p = data['pearson_p']
            spn_p = data['spearman_p']
        decomp_arr.append([pca_m, ft_m, psn_r, spn_r, psn_p, spn_p]) # in the past r2 went into here??
        return eigs, pows, space_r2, time_r2, decomp_arr

    elif kind in ('sim', 'noise'):
        with np.load(file, allow_pickle=True) as data:
            eigs = data['eigs']
            pows = data['pows']
            space_er = data['space_er']
            time_er = data['time_er']
            pca_m = data['pca_m']
            ft_m = data['ft_m']
            psn_r = data['pearson_r']
            spn_r = data['spearman_rho']
            psn_p = data['pearson_p']
            spn_p = data['spearman_p']
        decomp_arr.append([pca_m, ft_m, psn_r, spn_r, psn_p, spn_p])
        return eigs, pows, space_er, time_er, decomp_arr

def load_results(dir_, kind='mouse', plot=None, analysis_args=None):
    '''
    loads, [optionally] plots analysis results
    return of data dictionary currently not implemented
    dir_ : dir of data
    only plots if directory of figures given
    '''
    data_dict = dict()
    
    if analysis_args == None:
        with open(f'{dir_}/analysis_args.json','r') as f:
            analysis_args = json.load(f)
    print(analysis_args['ramsey_params'])
    if 'shuffle' in analysis_args and analysis_args.get('shuffle') != False:
        n_loop = analysis_args['shuffle'][1]
    else:
        n_loop = analysis_args['num_trials']

    if 'mouse' in kind: #"mouse" or "mouse_old"
        for mouse in analysis_args['mouse_in']:
            data_dict[mouse] = {}
            for region, count in MICE_REGIONS[mouse].items():
                decomp_arr = []
                subset_sizes = np.linspace(30,count,16, dtype=int)
                for i in range(n_loop):
                    #TODO supposed to decomp_arr to decomp_arr
                    if kind == 'mouse':
                        eigs, pows, pca_er, pca_b, ft_er1, ft_b1, ft_er2, ft_b2, decomp_arr = load_npz(f'{dir_}/{mouse}/{region}/ramsey_{i+1}.npz', kind, decomp_arr)
                    else:
                        eigs, pows, space_r2, time_r2, decomp_arr = load_npz(f'{dir_}/{mouse}/{region}/ramsey_{i+1}.npz', kind, decomp_arr)
                
                decomp_arr = np.array(decomp_arr, dtype=object)
                meta = {'count':count, **analysis_args['ramsey_params'], 'pc_range':[0,None], 'subsetsizes':subset_sizes}
                if kind == 'mouse':
                    data = { 
                            'pca_er':pca_er, 'ft_er1':ft_er1, 'ft_er2':ft_er2,
                            'eigs':eigs, 'pows':pows, 'pca_b':pca_b, 'ft_b1':ft_b1, 'ft_b2':ft_b2,
                            'espec_exp': decomp_arr[:,0].mean(0), 'psd_exp1': decomp_arr[:,1].mean(0), 'psd_exp2': decomp_arr[:,2].mean(0),
                            'pearson_corr1':decomp_arr[:,3], 'spearman_corr1':decomp_arr[:,4], 
                            'pearson_p1':decomp_arr[:,5], 'spearman_p1':decomp_arr[:,6], 
                            'pearson_corr2':decomp_arr[:,7], 'spearman_corr2':decomp_arr[:,8], 
                            'pearson_p2':decomp_arr[:,9], 'spearman_p2':decomp_arr[:,10], 
                            }
                else:
                    data = { 
                            'space_r2':space_r2, 'time_r2':time_r2,
                            'eigs':eigs, 'pows':pows,
                            'espec_exp': decomp_arr[:,0].mean(0), 'psd_exp': decomp_arr[:,1].mean(0),
                            'pearson_corr':decomp_arr[:,2], 'spearman_corr':decomp_arr[:,3], 
                            'pearson_p':decomp_arr[:,4], 'spearman_p1':decomp_arr[:,5],
                            }

                data_dict[mouse][region] = {'meta': meta, 'data':data} #appending a big tuple that includes the data
                
                if plot: #saves plots in folder
                    ramsey.plot_all_measures(data, meta)
                    plt.savefig(f'{plot}/{mouse}_{region}_measures')
                    plt.close('all')
                    plt.pause(0.01)

    elif kind == 'sim': #ising model
        N = analysis_args['N']
        for t in analysis_args['temps']:
            decomp_arr = []
            subset_sizes = np.linspace(30,N-10,16, dtype=int)
            for i in range(n_loop):
                eigs, pows, space_er, time_er, decomp_arr = load_npz(f'{dir_}/{t:.2f}/ramsey_{i+1}.npz', kind, decomp_arr)
            
            decomp_arr = np.array(decomp_arr, dtype=object)
            meta = {'subsetsizes':subset_sizes, **analysis_args['ramsey_params']}
            data = {'space_er':space_er, 'time_er':time_er, 'pc_range':[0,None], 'eigs':eigs, 'pows':pows, 'espec_exp': decomp_arr[:,0].mean(0), 'psd_exp': decomp_arr[:,1].mean(0), 'pearson_corr':decomp_arr[:,2], 'spearman_corr':decomp_arr[:,3], 'pearson_p':decomp_arr[:,4], 'spearman_p':decomp_arr[:,5]}
            data_dict[f'{t:.2f}'] = {'meta':meta, 'data':data}
            
            if plot:
                ramsey.plot_all_measures(data) #warning: doesn't have summed equivalent
                plt.savefig(f'{plot}/{t:.2f}_measures.png')
                plt.close('all')
                plt.pause(0.01)
            
    elif kind == 'noise':
        decomp_arr = []
        subset_sizes = np.linspace(30,1462,16, dtype=int)
        for i in range(n_loop):
            eigs, pows, space_er, time_er, decomp_arr = load_npz(f'{dir_}/ramsey_{i+1}.npz')
        decomp_arr = np.array(decomp_arr)
        data_dict = {**analysis_args['ramsey_params'], **{'subsetsizes':subset_sizes, 'space_er':space_er, 'time_er':time_er, 'pc_range':[0,None], 'eigs':eigs, 'pows':pows, 'espec_exp': decomp_arr[:,0].mean(0), 'psd_exp': decomp_arr[:,1].mean(0), 'pearson_corr':decomp_arr[:,2], 'spearman_corr':decomp_arr[:,3], 'pearson_p':decomp_arr[:,4], 'spearman_p':decomp_arr[:,5]}}
        
        if plot:
            plot_utils.plot_all_measures(data_dict)
            plt.savefig(f'{plot}/_measures.png')
            plt.close('all')
            plt.pause(0.01)

    return data_dict

