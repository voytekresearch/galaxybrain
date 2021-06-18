import numpy as np
from scipy import io, signal
import pandas as pd
import sys, os
import json

sys.path.append('../')
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import ramsey

import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
matplotlib.rc('figure', max_open_warning = 0)

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

def export_data(data,filename):
    #filename is a str, exports to csv
    data.to_csv(r'../data/'+filename+'.csv',encoding='utf-8', index=True)
    print('Saved data to ../data')

def _load_data(files = None, show_dir = False):
    """ 
    EXPERIMENTAL
    Load data from your data folder
    If show_dir is True, displays a list of files in your directory.  Omits the extension.
    If show_dir is False, assumes input is a string of format 'file1, file2 ...'
    Only really works for .csv for now
    """
    l = os.listdir('../data')
    l.remove('.DS_Store')
    if show_dir == True:
        for i, x in enumerate(l):
            x = x[:x.index('.')]
            l[i] = x
        print('Files in ../data: ', l)
        files = input("Enter file name(s). Separate with commas for multiple files.  Enter 'all' (w/o quotes)  to load all. \n")
        print('>Uploading...')
        if files == 'all':
            for z in l:
                globals()[z] = pd.read_csv("../data/"+ z +".csv", index_col = 0)
            print('>Done.')

        else:
            files = files.split(',')
            files = [i.translate({ord(i):None for i in ' '}) for i in files] # takes out spaces

            for z in files:
                globals()[z] = pd.read_csv("../data/"+ z +".csv", index_col = 0)
            print('>Done.')

    if show_dir == False:
        print('>Uploading...')
        if files == 'all':
            for z in l:
                globals()[z[:z.index('.')]] = pd.read_csv("../data/"+ z, index_col = 0)
            print('>Done.')
        else:
            files = files.split(',')
            files = [i.translate({ord(i):None for i in ' '}) for i in files] # takes out spaces
            for z in files:
                globals()[z] = pd.read_csv("../data/"+ z +".csv", index_col = 0)
            print('>Done.')

def load_npz(file):
    """
    Designated for easier loading of the saved Ramsey data
    File typically looks like '../data/experiments/CP/ramsey_'+str(i+1)+'.npz'
    """
    decomp_arr = []
    with np.load(file) as data:
        pca_m = data['pca_m']
        s_r2 = data['space_r2']
        ft_m = data['ft_m']
        t_r2 = data['time_r2']
        psn_r = data['pearson_r']
        spn_r = data['spearman_rho']
        psn_p = data['pearson_p']
        spn_p = data['spearman_p']
    decomp_arr.append([pca_m, ft_m, psn_r, spn_r, psn_p, spn_p])
    return np.array(decomp_arr)

mice_regions = {'krebs': {'all': 1462,'CP': 176,'HPF': 265,'LS': 122,'MB': 127,'TH': 227,'V1': 334},
                'robbins': {'all': 2688,  'FrMoCtx': 647,  'HPF': 333,  'LS': 133,  'RSP': 112,  'SomMoCtx': 220,  'TH': 638,  'V1': 251,  'V2': 124}, 
                'waksman': {'all': 2296, 'CP': 134, 'HPF': 155, 'TH': 1878}}

def load_and_plot(dir_, type_='mouse', mice = mice_regions.keys(), plot=None,analysis_args=None):
    '''
    loads, plots analysis results
    return of data dictionary currently not implemented
    dir_ : dir of data
    only plots if directory of figures given
    '''
    data_dict = dict()
    
    if analysis_args == None:
        with open(f'{dir_}/analysis_args.json','r') as f:
            analysis_args = json.load(f)
    
    if 'shuffle' in analysis_args:
        n_loop = analysis_args['shuffle'][1]
    else:
        n_loop = analysis_args['num_trials']

    if type_ == 'mouse':    
        for mouse in mice:
            data_dict[mouse] = []
            for region, count in mice_regions[mouse].items():
                decomp_arr = []
                subset_sizes = np.linspace(30,count,16, dtype=int)
                for i in range(n_loop):
                    with np.load(f'{dir_}/{mouse}/{region}/ramsey_' + str(i+1) + '.npz', allow_pickle=True) as data:
                        eigs = data['eigs']
                        pows = data['pows']
                        space_er = data['space_er']
                        time_er = data['time_er']
                        pca_m = data['pca_m']
                        ft_m = data['ft_m']
                        pca_b = data['pca_b']
                        ft_b = data['ft_b']
                        psn_r = data['pearson_r']
                        spn_r = data['spearman_rho']
                        psn_p = data['pearson_p']
                        spn_p = data['spearman_p']
                        curr_pc_range = data['curr_pc_range']

                    decomp_arr.append([pca_m, ft_m, psn_r, spn_r, psn_p, spn_p, pca_b, ft_b,])

                decomp_arr = np.array(decomp_arr)
                data = {**analysis_args['ramsey_params'], **{'subsetsizes':subset_sizes, 
                        'space_er':space_er, 'time_er':time_er, 'pc_range':[0,None], 
                        'eigs':eigs, 'pows':pows, 'pca_b':pca_b, 'ft_b':ft_b, 'espec_exp': decomp_arr[:,0].mean(0), 
                        'psd_exp': decomp_arr[:,1].mean(0), 'pearson_corr':decomp_arr[:,2], 
                        'spearman_corr':decomp_arr[:,3], 'pearson_p':decomp_arr[:,4], 
                        'spearman_p':decomp_arr[:,5], 'curr_pc_range':curr_pc_range}}
                data_dict[mouse].append((region, count, data)) #appending a big tuple that includes the data
                
                if plot:
                    ramsey.plot_all_measures(subset_sizes, space_er=space_er, time_er=time_er, 
                                pc_range=[0,None], 
                                eigs=eigs, 
                                pows=pows,
                                space_slopes=decomp_arr[:,0].mean(0), 
                                time_slopes=decomp_arr[:,1].mean(0), 
                                pearson_corr=decomp_arr[:,2], 
                                spearman_corr=decomp_arr[:,3], 
                                pearson_p=decomp_arr[:,4], 
                                spearman_p=decomp_arr[:,5],
                                **analysis_args['ramsey_params'])

                    plt.savefig(f'{plot}/{mouse}_{region}_measures')

                    plt.close('all')
                    plt.pause(0.01)
    
    elif type_ == 'sim':
        N = analysis_args['N']
        for t in analysis_args['temps']:
            data_dict[f'{t:.2f}'] = []
            decomp_arr = []
            subset_sizes = np.linspace(30,N-10,16, dtype=int)
            for i in range(n_loop):
                with np.load(f'{dir_}/{t:.2f}/ramsey_{i+1}.npz', allow_pickle=True) as data:
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

            decomp_arr = np.array(decomp_arr)
            data = {**analysis_args['ramsey_params'], **{'subsetsizes':subset_sizes, 'space_er':space_er, 'time_er':time_er, 'pc_range':[0,None], 'eigs':eigs, 'pows':pows, 'espec_exp': decomp_arr[:,0].mean(0), 'psd_exp': decomp_arr[:,1].mean(0), 'pearson_corr':decomp_arr[:,2], 'spearman_corr':decomp_arr[:,3], 'pearson_p':decomp_arr[:,4], 'spearman_p':decomp_arr[:,5]}}
            data_dict[f'{t:.2f}'].append(data)
            
            if plot:
                ramsey.plot_all_measures(subset_sizes, space_er=space_er, time_er=time_er, 
                            pc_range=[0,None], 
                            eigs=eigs, 
                            pows=pows,
                            space_slopes=decomp_arr[:,0].mean(0), 
                            time_slopes=decomp_arr[:,1].mean(0), 
                            pearson_corr=decomp_arr[:,2], 
                            spearman_corr=decomp_arr[:,3], 
                            pearson_p=decomp_arr[:,4], 
                            spearman_p=decomp_arr[:,5],
                            **analysis_args['ramsey_params'])

                plt.savefig(f'{plot}/{t:.2f}_measures.png')

                plt.close('all')
                plt.pause(0.01)
            
    elif type_ == 'noise':
        # data_dict = [] #don't need dict because no regions. Yes it's poorly named
        decomp_arr = []
        subset_sizes = np.linspace(30,1462,16, dtype=int)
        for i in range(n_loop):
            with np.load(f'{dir_}/ramsey_{i+1}.npz', allow_pickle=True) as data:
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

        decomp_arr = np.array(decomp_arr)
        data_dict = {**analysis_args['ramsey_params'], **{'subsetsizes':subset_sizes, 'space_er':space_er, 'time_er':time_er, 'pc_range':[0,None], 'eigs':eigs, 'pows':pows, 'espec_exp': decomp_arr[:,0].mean(0), 'psd_exp': decomp_arr[:,1].mean(0), 'pearson_corr':decomp_arr[:,2], 'spearman_corr':decomp_arr[:,3], 'pearson_p':decomp_arr[:,4], 'spearman_p':decomp_arr[:,5]}}
        # data_dict.append(data)
        
        if plot:
            ramsey.plot_all_measures(subset_sizes, space_er=space_er, time_er=time_er, 
                        pc_range=[0,None], 
                        eigs=eigs, 
                        pows=pows,
                        space_slopes=decomp_arr[:,0].mean(0), 
                        time_slopes=decomp_arr[:,1].mean(0), 
                        pearson_corr=decomp_arr[:,2], 
                        spearman_corr=decomp_arr[:,3], 
                        pearson_p=decomp_arr[:,4], 
                        spearman_p=decomp_arr[:,5],
                        **analysis_args['ramsey_params'])

            plt.savefig(f'{plot}/_measures.png')

            plt.close('all')
            plt.pause(0.01)

    return data_dict