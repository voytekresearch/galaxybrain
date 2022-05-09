# imported from old notebook

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SummaryPlots: 
    def __init__(self):
        self.mice_regions = {'krebs': [[], [('all', 1462), ('CP', 176), ('HPF', 265), ('LS', 122), ('MB', 127), ('TH', 227), ('V1', 334)]],
                             'robbins': [[], [('all', 2688), ('FrMoCtx', 647), ('HPF', 333), ('LS', 133), ('RSP', 112), ('SomMoCtx', 220), ('TH', 638), ('V1', 251), ('V2', 124)]],
                             'waksman': [[], [('all', 2296), ('CP', 134), ('HPF', 155), ('TH', 1878)]] }
        
        all_regions = []
        for i_m, mouse_key in enumerate(['krebs', 'robbins', 'waksman']):
            for region in self.mice_regions[mouse_key][1]:
                all_regions.append(region[0])
        self.all_regions = np.unique(all_regions)

        #self.corr_df = pd.DataFrame(index = self.all_regions, columns = ['Mouse 1', 'Mouse 2', 'Mouse 3'])

    
    def mouse_region_iterator(self, plot_type, verbose, spec = None, CKEYS = None, marker = None):
        """
        Iterates through mouse-region pairs to extract relevant statistics.  For certain plots does some of the plotting.
        plot_type: either 'heat map', 'line_corr', bar_plot' or 'avg pca exp'
        CKEYS, marker are specific plotting vars (for 'avg pca exp')
        """
        debug = []
        fractions = np.linspace(.0625,1,16)
        if plot_type == "heat map":
            self.mouse_heat_maps = [pd.DataFrame(index = self.all_regions, columns = fractions), pd.DataFrame(index = self.all_regions, columns = fractions), pd.DataFrame(index = self.all_regions, columns = fractions)]
        elif plot_type == "bar_plot":
            self.corr_df = pd.DataFrame(index = self.all_regions, columns = ['Mouse 1', 'Mouse 2', 'Mouse 3'])
            
        for i_m, mouse_key in enumerate(['krebs', 'robbins', 'waksman']):
            if verbose:
                print('=============== \n', 'MOUSE:', mouse_key, '\n=============== ')
            for region in self.mice_regions[mouse_key][1]:
                region_name = region[0]; region_count = region[1]
                if verbose:
                    print(mouse_key, region_name, '(size: ', str(region_count) +')')

                subset_sizes = np.linspace(30,region_count,16, dtype=int)
                
                decomp_arr = []
                for i in range(4):
                    with np.load('../data/experiments/exp5/' + mouse_key + '/' + region_name + '/ramsey_'+str(i+1)+'.npz', allow_pickle=True) as data:
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

                    decomp_arr.append([pca_m, ft_m, psn_r, spn_r, psn_p, spn_p, space_r2, time_r2])

                decomp_arr = np.array(decomp_arr)

                mouse_df_key = 'Mouse {}'.format(i_m+1)
                
                
                psn_r, spn_r = decomp_arr[:,2], decomp_arr[:,3]
                psn_p, spn_p = decomp_arr[:,4], decomp_arr[:,5]
                n_trials = psn_r.shape[0]
                
                psn_r = np.array([psn_r[i] for i in range(n_trials)], dtype=float)
                spn_r = np.array([spn_r[i] for i in range(n_trials)], dtype=float)

                psn_p = np.array([psn_p[i] for i in range(n_trials)], dtype=float)
                spn_p = np.array([spn_p[i] for i in range(n_trials)], dtype=float)
    
                
                if plot_type == 'heat map':
                    #for heat map
                    for i_f, f in enumerate(fractions):
                        self.mouse_heat_maps[i_m][f][region_name] = psn_r.mean(0)[i_f]

                    
                elif plot_type == 'avg pca exp':
                    spec_index_table = {'eig': 0,'pow': 1}
                    i_feat = spec_index_table[spec]
                    plt.plot(subset_sizes, decomp_arr.mean(0)[i_feat].mean(0), marker[i_m]+'-', ms=10, color=CKEYS[np.where(self.all_regions==region[0])[0][0]])
            #         plt.plot(subset_sizes, decomp_arr.mean(0)[i_feat].mean(0)/decomp_arr.mean(0)[i_feat].mean(0)[0], marker[i_m]+'-', ms=10, color=CKEYS[np.where(all_regions==region[0])[0][0]])
            #         plt.plot(subset_sizes, decomp_arr.mean(0)[3], marker[i_m]+'-', ms=10, color=CKEYS[np.where(all_regions==region[0])[0][0]])
                    plt.xscale('log');#plt.yscale('log')



    def avg_exp(self, spec, save = False, verbose = False):
        """Plot f(subset_size) = average spectral exponent
        spec = 'eig' or 'pow' 
        """
        CKEYS = plt.rcParams['axes.prop_cycle'].by_key()['color']
        CKEYS.append('#000000')
        marker='o*^'
        plt.figure(figsize=(12,10))

        self.mouse_region_iterator(plot_type = 'avg pca exp', spec = spec, verbose = verbose, CKEYS = CKEYS, marker = marker)

        for i_r, r in enumerate(self.all_regions):
            plt.plot(100,1,color=CKEYS[i_r], label=r)
        plt.legend()
        plt.show()