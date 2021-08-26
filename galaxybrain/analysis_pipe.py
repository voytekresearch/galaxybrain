import numpy as np
import sys, os
import json
sys.path.append('../')
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from data_utils import load_mouse_data, return_pops
import ramsey

import multiprocessing as mp

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#Parallel stuff
cores = mp.cpu_count()
pool = mp.Pool(7)

def shuffle_data(data, axis): 
    """Helper function to shuffle data"""
    if axis == 'time':
        t_ind = np.arange(data.shape[0]) 
        np.random.shuffle(t_ind)
        raster_curr = data.iloc[t_ind]
    elif axis == 'space':
        raster_curr = data.apply(np.random.permutation, axis=1, result_type='broadcast') #need broadcast to maintain shape
    return raster_curr

def run_analysis(output_dir, mice_regions, num_trials, ramsey_params, burn_in = 20, shuffle = False, mouse_in = ['krebs', 'robbins', 'waksman'],parallel=True):
    """
    output_dir: assumes you've made expNUM folder
    ramsey_params: params besides data and subsetsizes
    burn_in: skipping beginning and end of recordings (see DataDiagnostic.ipynb
    shuffle = (axis, num_shuffles)
    
    **This function refers to some variables declared below in __main__**
    """
    #var housekeeping
    n_iters = ramsey_params['n_iters']
    n_pc = ramsey_params['n_pc']
    f_range = ramsey_params['f_range']
    
    #analysis + saving data 
    for mouse_key in mouse_in:
        mouse = mice_regions[mouse_key][0] #this is the data
        ### TODO : maybe move this outside and append mouse names to labels ###
        parallel_args = [] #just need to keep track of indices
        parallel_labels = [] # for going through results and saving data later
        for region in mice_regions[mouse_key][1]:
            
            region_name = region[0]; region_count = region[1]
            print(region_name)
            os.makedirs(f'{output_dir}/{mouse_key}/{region_name}')

            if region_name == 'all':
                mouse_raster = mouse[0].iloc[burn_in:-burn_in]
            else:
                mouse_raster = mouse[0][mouse[1][region_name]].iloc[burn_in:-burn_in] #mouse[1] are indices

            subsetsizes = np.linspace(30,region_count,16, dtype=int)
            
            if shuffle:
                for s in range(shuffle[1]):
                    curr_raster = shuffle_data(mouse_raster, shuffle[0]) 
                    #curr_output = {'eigs':[],'pows':[],'pca_m':[],'s_er':[],'ft_m':[],'t_er':[],'psn_r':[], 'spn_r':[], 'psn_p':[], 'spn_p':[]}

                    if parallel:
                        [parallel_args.append((curr_raster,subsetsizes)) for n in range(num_trials)]
                        [parallel_labels.append((region_name, s)) for n in range(num_trials)]
                        
                    else:
                        for n in range(num_trials):
                            eigs, pows, pca_m, pca_er, pca_b, ft_m, ft_er, ft_b, psn_r, spn_r, psn_p, spn_p  = ramsey.ramsey(curr_raster, subsetsizes, **ramsey_params)
                            curr_output.append([eigs, pows, pca_m, pca_er, ft_m, ft_er, psn_r, spn_r, psn_p, spn_p, pca_b, ft_b])
                        
                        # OUTDATED (parallel is up to date) AVG ACROSS TRIALS HERE
                        curr_output = np.array(curr_output)
                        np.savez(f'{output_dir}/{mouse_key}/{region_name}/ramsey_{s+1}',eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
                                                                                        pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
                                                                                        pca_m=curr_output[:,2].mean(0), space_er=curr_output[:,3].mean(0), 
                                                                                        ft_m=curr_output[:,4].mean(0), time_er=curr_output[:,5].mean(0), 
                                                                                        pearson_r=curr_output[:,6].mean(0), spearman_rho=curr_output[:,7].mean(0), 
                                                                                        pearson_p=curr_output[:,8].mean(0), spearman_p=curr_output[:,9].mean(0))
            else:
                if parallel:
                    [parallel_args.append((mouse_raster,subsetsizes)) for n in range(num_trials)]
                    [parallel_labels.append((region_name, n)) for n in range(num_trials)]
                else:
                    for i in range(num_trials):
                        eigs, pows, pca_m, s_er, ft_m, t_er, psn_r, spn_r, psn_p, spn_p = ramsey.ramsey(mouse_raster, subsetsizes, **ramsey_params)
                        np.savez(f'{output_dir}/{mouse_key}/{region_name}/ramsey_{i+1}', eigs=eigs, pows=pows, 
                                                                                        pca_m=pca_m, space_er=s_er, 
                                                                                        ft_m=ft_m, time_er=t_er, 
                                                                                        pearson_r=psn_r, spearman_rho=spn_r, pearson_p=psn_p, spearman_p=spn_p)
            
            
        if parallel:
            results = [pool.apply(ramsey.ramsey, args = (_curr_raster, _subsetsizes, n_iters, n_pc, f_range)) for (_curr_raster,_subsetsizes) in parallel_args]
            pool.close()
            if shuffle:
                for i in np.arange(0,len(results), num_trials):
                    region_name, s = parallel_labels[i][0], parallel_labels[i][1]
                    curr_output = np.array(results[i:i+num_trials]) #slice across trials to avg after
                    np.savez(f'{output_dir}/{mouse_key}/{region_name}/ramsey_{s+1}',eigs=np.array([curr_output[:,0][i] for i in range(num_trials)]).mean(0), # this properly takes the mean over trials
                                                                                    pows=np.array([curr_output[:,1][i] for i in range(num_trials)]).mean(0), # ^
                                                                                    pca_m=curr_output[:,2].mean(0), pca_er=curr_output[:,3].mean(0), pca_b=curr_output[:,4].mean(0), 
                                                                                    ft_m1=curr_output[:,5].mean(0), ft_er1=curr_output[:,6].mean(0), ft_b1=curr_output[:,7].mean(0), 
                                                                                    ft_m2=curr_output[:,8].mean(0), ft_er2=curr_output[:,9].mean(0), ft_b2=curr_output[:,10].mean(0), 
                                                                                    pearson_r1=curr_output[:,11].mean(0), spearman_rho1=curr_output[:,13].mean(0), 
                                                                                    pearson_p1=curr_output[:,12].mean(0), spearman_p1=curr_output[:,14].mean(0),
                                                                                    pearson_r2=curr_output[:,15].mean(0), spearman_rho2=curr_output[:,17].mean(0), 
                                                                                    pearson_p2=curr_output[:,16].mean(0), spearman_p2=curr_output[:,18].mean(0))
                                                                                     #curr_pc_range=curr_output[:,12][0] # saving the first because they are all identical
                
            else:
                for i in range(len(results)):
                    region_name, tn = parallel_labels[i][0], parallel_labels[i][1]
                    curr_output = np.array(results[i])
                    np.savez(f'{output_dir}/{mouse_key}/{region_name}/ramsey_{tn+1}',   eigs=curr_output[0], pows=curr_output[1], 
                                                                                        pca_m=curr_output[2], pca_er=curr_output[3], pca_b=curr_output[4], 
                                                                                        ft_m1=curr_output[5], ft_er1=curr_output[6],ft_b1=curr_output[7], 
                                                                                        ft_m2=curr_output[8], ft_er2=curr_output[9],ft_b2=curr_output[10], 
                                                                                        pearson_r1=curr_output[11], spearman_rho1=curr_output[13], 
                                                                                        pearson_p1=curr_output[12], spearman_p1=curr_output[14],
                                                                                        pearson_r2=curr_output[15], spearman_rho2=curr_output[17], 
                                                                                        pearson_p2=curr_output[16], spearman_p2=curr_output[18])  #curr_pc_range=curr_output[12]

            
### SCRIPT ###
if __name__ == '__main__':
    #load mouse data
    datafolder = '../data/spikes/' #where the mouse data is
    all_mice = []
    for i_m in range(3):
        print('Mouse ' + (str(i_m+1)))
        all_mice.append([])
        df_spk, df_info = load_mouse_data(datafolder, i_m, return_type='binned', bin_width=1)
        region_indices = {}
        for region in df_info.region.unique():
            region_indices.update({region:np.where(df_info['region'] == str(region))[0]})

        spk_list, region_labels = return_pops(df_spk, df_info)
        print(list(zip(region_labels, [s.shape[1] for s in spk_list])), 'Total:',sum([s.shape[1] for s in spk_list]))

        su_start_ind = len(region_labels)+1

        all_mice[i_m].append(df_spk[df_spk.columns[su_start_ind:]])
        all_mice[i_m].append(region_indices)

    krebs = all_mice[0]; robbins = all_mice[1]; waksman = all_mice[2]

    mice_regions = {'krebs': [krebs, [('all', 1462), ('CP', 176), ('HPF', 265), ('LS', 122), ('MB', 127), ('TH', 227), ('V1', 334)]],
                'robbins': [robbins, [('all', 2688), ('FrMoCtx', 647), ('HPF', 333), ('LS', 133), ('RSP', 112), ('SomMoCtx', 220), ('TH', 638), ('V1', 251), ('V2', 124)]],
                'waksman': [waksman, [('all', 2296), ('CP', 134), ('HPF', 155), ('TH', 1878)]] }

    analysis_args={'output_dir' : '../../../../projects/ps-voyteklab/brirry/data/experiments/NOSUM',
                    'mice_regions' : mice_regions,
                    'ramsey_params' : {'n_iters' : 95, 'n_pc' : 0.8, 'f_range' : [0,0.4]},
                    'num_trials' : 4,
                    'mouse_in' : ['krebs'],
                    'shuffle' : ('space',5)}
    
    run_analysis(**analysis_args)

    analysis_args['mice_regions'] = 'mice_regions_var'

    with open(f"{analysis_args['output_dir']}/analysis_args.json",'w') as f:
        json.dump(analysis_args, f, indent=1)


