import numpy as np
from scipy import io, signal
import pandas as pd
import sys, os
from collections import defaultdict
import json

from pathlib import Path
here_dir = str(Path(__file__).parent.absolute())
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
matplotlib.rc('figure', max_open_warning = 0)


# brain_region : size
MICE_META   = {'krebs': {'all': 1462,'CP': 176,'HPF': 265,'LS': 122,'MB': 127,'TH': 227,'V1': 334},
               'robbins': {'all': 2688,  'FrMoCtx': 647,  'HPF': 333,  'LS': 133,  'RSP': 112,  'SomMoCtx': 220,  'TH': 638,  'V1': 251,  'V2': 124}, 
               'waksman': {'all': 2296, 'CP': 134, 'HPF': 155, 'TH': 1878}}
ALL_REGIONS = list({k2 for v1 in MICE_META.values() for k2 in v1}) # list to use indices later

### Helper functions

def _beg_end(spikes_all):
    spikes_concat = np.concatenate(spikes_all)
    return np.floor(spikes_concat.min()), np.ceil(spikes_concat.max())


def _return_pops(data, df_info):
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
    cluLocs = io.loadmat(datafolder+'/cluLoc.mat', squeeze_me = True)
    probe_id = cluLocs['iprobeAll']
    probe_loc = cluLocs['brainLocNameAll']

    # load data and cluster info
    mouse_data = io.loadmat(datafolder+'/spks/spks%s_Feb18.mat'%mice[i_m], squeeze_me = True)['spks']
    clu_info = pd.DataFrame(np.array([probe_id[i_m],probe_loc[i_m]]).T, columns=['probe', 'region'])

    print('Grabbing Spikes...')
    spikes_all = []
    for probe in range(len(mouse_data)):
        st = mouse_data[probe][0]
        clu = mouse_data[probe][1]
        # add spike time into each
        spikes_all += [np.sort(st[clu==k]) for k in np.unique(clu)]

    if return_type == 'spiketimes':
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

    if return_type == 'binned':
        return df_spk, clu_info

    if return_type == 'smoothed':
        print('Smoothing...')
        win_len, win_std = smooth_param
        win = signal.windows.gaussian(int(win_len/bin_width)+1, win_std/bin_width)
        win/=win.sum()
        bin_smoothed = signal.convolve(df_spk, win[:,None], mode='same')
        df_spk_smo = pd.DataFrame(bin_smoothed, index=t_bins[:-1])
        return df_spk_smo, clu_info


def shuffle_data(data, axis): 
    """Helper function to shuffle data"""
    if axis == 'time':
        t_ind = np.arange(data.shape[0]) 
        np.random.shuffle(t_ind)
        raster_curr = data.iloc[t_ind]
    elif axis == 'space':
        raster_curr = data.apply(np.random.permutation, axis=1, result_type='broadcast') #need broadcast to maintain shape
    return raster_curr

    
class MouseData:
    mice_names = list(MICE_META.keys())
    def __init__(self, mouse_in=mice_names, burn_in=20, phantom=False):
        """
        Uses load_mouse_data to return a dictionary of mouse data and region info
        args:
            mouse_in: list of mouse names
            burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb)
            
        Returns:
            Dict of form: {mouse_name : (spike_dataframe, region_indices_dict }
            See mouse_iter() for use case
        """
        
        datafolder       = here_dir+'/../data/spikes'
        self.mouse_in    = mouse_in
        self.burn_in     = burn_in
        self.raster_dict = {}
        self.phantom     = phantom
        if phantom:
            global MICE_META
            # limit to one iteration
            MICE_META = {'phantom':{'phantom_region':None}}
            self.mouse_in = ['phantom'] 
            return
        for name in mouse_in:
            i_m = self.mice_names.index(name)
            print(f'Mouse {i_m+1}')
            df_spk, df_info = load_mouse_data(datafolder, i_m, return_type='binned', bin_width=1)
            region_indices = {}
            for region in df_info.region.unique():
                region_indices.update({region:np.where(df_info['region'] == str(region))[0]})
                
            spk_list, region_labels = _return_pops(df_spk, df_info)
            print(list(zip(region_labels, [s.shape[1] for s in spk_list])), 'Total:',sum([s.shape[1] for s in spk_list]))
            su_start_ind = len(region_labels)+1
            self.raster_dict[name] = (df_spk[df_spk.columns[su_start_ind:]], region_indices)

    def get_labels(self):
        for m in self.mouse_in:
            for r in MICE_META[m]:
                yield f'{m}_{r}'

    def get_spikes(self, mouse_name=None, region='all', label=None):
        """
        return single raster for mouse-region pair
        df shape: n_time x n_neuron
        label: input str like 'mousename_region' (used in some scripts)
        (decoupled from mouse_iter in case you want to use in a notebook for one dataset)
        """
        if self.phantom:
            return self.dummy_data()
        if label:
            mouse_name, region = label.split('_')
        if mouse_name not in self.mouse_in:
            raise KeyError(f'{mouse_name} not loaded')
        ix = self.burn_in
        spike_df, region_idx = self.raster_dict[mouse_name]
        if region == 'all':
            mouse_raster = spike_df.iloc[ix:-ix]
        else:
            mouse_raster = spike_df[region_idx[region]].iloc[ix:-ix]

        return mouse_raster

    def mouse_iter(self, mouse_name):
        """
        Yields mouse spike df, region name
        (for any analysis loop)
        """
        for region in MICE_META[mouse_name]:
            mouse_raster = self.get_spikes(mouse_name, region)
            yield mouse_raster, region

    def dummy_data(self):
        """fake data for debugging"""
        return pd.DataFrame(np.random.poisson(1, size=(500,100)))


#####################################
### Analysis result data handling ###
#####################################

def load_results(dir_, kind='mouse', plot='', analysis_args=None):
    """
    loads, [optionally] plots analysis results
    dir_ : dir of data
    only plots if directory of figures given

    returns data dictionary of form:
    {'mouse' : {'region' : {'meta' : {'count':int, 'subsetsizes':array},
                            'data' : {'pca_m':array, ...} }
                } }
    """
    
    def format_data(f_prefix, both_psd=True):
        """load, reshape/average certain data across trials"""
        one_time_keys = ['eigs', 'pows', 'es_error']
                    
        if both_psd: # at the time of adding both psds, also added offset
            one_time_keys += ['psd_error1', 'psd_offset1', 'psd_error2', 'psd_offset2', 'es_offset']
        else:
            one_time_keys += ['psd_error']

        decomp_dict = defaultdict(lambda: [])
        for i in range(1, n_loop+1):
            with np.load(f'{f_prefix}/{i}.npz', allow_pickle=True) as data:
                for k in set(data.files) - set(one_time_keys):
                    decomp_dict[k].append(data[k])
                other_spec_data = {k : data[k] for k in one_time_keys }
        decomp_dict = {k : np.array(decomp_dict[k]) for k in decomp_dict} # to access np methods
        for k in list(decomp_dict):
            # Average the slopes across trials
            if '_exponent' in k: 
                decomp_dict[k] = decomp_dict[k].mean(0)
            if 'knee' in k:
                decomp_dict[k] = decomp_dict[k].mean(0)
        return {**decomp_dict, **other_spec_data}

    data_dict = dict()

    if analysis_args == None:
        with open(f'{dir_}/analysis_args.json','r') as f:
            analysis_args = json.load(f)

    
    print(analysis_args['ramsey_kwargs'])
    if 'shuffle' in analysis_args and analysis_args.get('shuffle') != False:
        n_loop = analysis_args['shuffle'][1]
    else:
        n_loop = analysis_args['num_trials']

    if 'mouse' in kind: #"mouse" or "mouse_shuffle"
        if kind == 'mouse_shuffle':
            both_psd = False
        else:
            both_psd = True
        for mouse in analysis_args['mouse_kwargs']['mouse_in']:
            data_dict[mouse] = {}
            for region, count in MICE_META[mouse].items():
                subset_sizes = np.linspace(30, count, 16, dtype=int)
                meta = {'count':count, 'subsetsizes':subset_sizes}
                data = format_data(f'{dir_}/{mouse}_{region}', both_psd)
                data_dict[mouse][region] = {'meta':meta, 'data':data} #appending a big tuple that includes the data

    elif kind == 'sim': #ising model
        N = analysis_args['N']
        for t in next(os.walk(dir_))[1]:
            subset_sizes = np.linspace(100, N, 10, dtype=int) #used to be N-10
            meta = {'subsetsizes':subset_sizes}
            try:
                data = format_data(f'{dir_}/{t}')
            except KeyError: #psd_error2:
                print(f'{t} doesn\'t have psd_error2')
                continue
            data_dict[t] = {'meta':meta, 'data':data}
            
    elif kind == 'noise': # gaussian noise
        subset_sizes = np.linspace(30, 1462, 16, dtype=int)
        meta  = {'subsetsizes' : subset_sizes}
        data = format_data(f'{dir_}')
        ## since only one kind of noise, just one key
        data_dict['all'] = {'data':data, 'meta':meta} 

    data_dict['meta'] = {**analysis_args['ramsey_kwargs']} # meta for all data
    
    return data_dict
