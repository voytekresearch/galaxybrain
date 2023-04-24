import numpy as np
from collections import defaultdict
from fooof import FOOOFGroup, FOOOF
from scipy import stats
from neurodsp.spectral import compute_spectrum
from joblib import Parallel, delayed, cpu_count
import warnings
warnings.filterwarnings("ignore")
import time
import gc
from multiprocessing import Pool, TimeoutError


def fooofy(components, spectra, fit_range, group=True, fit_kwargs={}, return_params='default'):
    """
    fit FOOOF model on given spectrum and return params
    Parameters:
        components (1d array): frequencies or PC dimensions
        spectra (2d array): PSDs or variance explained
        fit_range (list): range for x axis of spectrum to fit like [0, 0.4]
        group: whether to use FOOOFGroup or not
    """
    if return_params == 'default':
        return_params = [['aperiodic_params', 'exponent'],
                          ['error'], # MAE
                          ['aperiodic_params', 'offset']]
    if group:
        fg = FOOOFGroup(max_n_peaks=0, verbose=False, **fit_kwargs)
    else:
        fg = FOOOF(max_n_peaks=0, verbose=False, **fit_kwargs)

    fg.fit(components, spectra, fit_range)
    return {p[-1]: fg.get_params(*p) 
                    for p in return_params}


def pca(data, n_pc=None):
    """
    Decomposition in space
    """
    from sklearn.decomposition import PCA

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # logging.debug('pca: init')
    pca = PCA(n_pc)
    # logging.debug('pca: fit')
    pop_pca = pca.fit(data)
    # logging.debug('pca: get var')
    evals = pop_pca.explained_variance_ratio_
    del PCA,pca # avoid potential memory leak? lol
    gc.collect()
    return evals


def ft(subset, **ft_kwargs):
    """
    Welch method over both summed and non summed neurons
    returns: freqs, powers_summed, powers_chans (2d array n_chans x n_freqs)

    fourier kwargs for mouse:
        fs       = 1
        nperseg  = 120
        noverlap = NPERSEG/2
    for ising model 10000 long
        fs       = 1
        nperseg  = 2000
        noverlap = int(.8*NPERSEG)
    """
    if not isinstance(subset, np.ndarray):
        subset = np.array(subset)
    summed_neurons = subset.sum(axis= 1) # summing data for ft decomp.
    _freqs, powers_summed = compute_spectrum(summed_neurons, **ft_kwargs, method='welch') #powers_sum is an array
    _freqs, powers_chans  = compute_spectrum(subset.T,       **ft_kwargs, method='welch') #returns a matrix!

    return powers_summed, powers_chans


class Ramsey:
    def __init__(self, logger, data, n_iter, n_pc, ft_kwargs, pc_range, f_range, fooof_kwargs={}, data_type='mouse', parallel=True):
        """
        f_range: range of frequencies to fit (default freqs usually go up to 0.5) or None
        fooof_kwargs = { 'es' : {
                    'return_params': [...],
                    'fit_kwargs': {...}
                    }
                'psd': SAME^
                }
        """
        self.logger = logger
        self.data = data
        self.n_iter = n_iter
        self.n_pc = n_pc
        self.ft_kwargs = ft_kwargs
        self.pc_range = pc_range
        self.f_range = f_range # TODO make input percentages, then transform here
        self.fooof_kwargs = fooof_kwargs
        self.data_type = data_type
        self.parallel = parallel

    def subset_iter(self):
        """Do random_subset_decomp over incrementing subset sizes
        slope dims: n_iter * amount of subset sizes
        b: offsets
        returns: eigs, pows (2D)
        fit results and stats
        """
        data = self.data[0](self.data[1])

        if self.data_type == 'mouse':
            subset_sizes = np.linspace(30, data.shape[1], 16, dtype=int)
        elif self.data_type == 'ising':
            subset_sizes = np.linspace(100, data.shape[1], 10, dtype=int)
        del data
        gc.collect()
        eigs        = []
        powers_sum  = []
        n_subsets     = len(subset_sizes)
        fit_results = defaultdict(lambda: np.zeros((self.n_iter, n_subsets)))
        stats_      = defaultdict(lambda: np.zeros(n_subsets))
        if self.parallel:
            self.logger.info(f'working with {cpu_count()} processors')
        for i, num in enumerate(subset_sizes):
            self.logger.debug(f'subset {i+1}')
            n_pc_curr = int(self.n_pc*num)

            # [0,None] for whole range, otherwise check if float for fraction
            if self.pc_range == [0,None]:
                pc_range_curr = [0, int(min(.5*n_pc_curr, .25*max(subset_sizes*self.n_pc)))]
            elif isinstance(self.pc_range[1], float): # cut up to percent
                pc_range_curr = [self.pc_range[0], int(n_pc_curr*self.pc_range[1])]
            # DEBUG can't fooof fit if range is too small
            if pc_range_curr[1] < 3:
                self.logger.warning(f'skipping subset {num}')
                continue

            spectra_i, results_i = self.random_subset_decomp(subset_size=num, n_pc_sub=n_pc_curr, pc_range_sub=pc_range_curr)
            #append average across iterations
            eigs.append(spectra_i['evals'].mean(0)) 
            powers_sum.append(spectra_i['psd'].mean(0))

            for measure, dat in results_i.items():
                fit_results[measure][:,i] =  dat

            # calculate correlation across fits of random subsets
            for it in [1,2]: #summed and non summed
                try:
                    stats_[f'pearson_corr{it}'][i],  stats_[f'pearson_p{it}'][i]  = stats.pearsonr( results_i['es_exponent'], results_i[f'psd_exponent{it}'])
                    stats_[f'spearman_corr{it}'][i], stats_[f'spearman_p{it}'][i] = stats.spearmanr(results_i['es_exponent'], results_i[f'psd_exponent{it}'])
                except ValueError: # can't compute correlation
                    self.logger.warning(f"NaNs at subset iter: {i}, num: {num}")
                except KeyError: # skip psd_exponent nosum because of 0s spec
                    pass

        # NOTE: need to unpack dict in shuffle case (key order conserved python 3.6)
        return {'eigs': eigs, 
                'pows': powers_sum, 
                **fit_results, 
                **stats_}

    def random_subset_decomp(self, subset_size, n_pc_sub, pc_range_sub):
        """
        returned data include 1 pca exponent and 2 PSD exponents
        performs iter_task of main computational load
            - not all tasks are distributed as reaching CPU limit has diminishing returns

        Returns
            spectra := {evals, psd, psd_chan}
            fit_dict := {es_<fit_item>..., psd_<fit_item>...}
        """
        freqs = np.fft.rfftfreq(self.ft_kwargs['nperseg'])
        pcs   = np.arange(1,n_pc_sub+1)

        evals_mat       = np.empty((self.n_iter, n_pc_sub)) # n_iter * |evals|
        sum_powers_mat  = np.empty((self.n_iter, len(freqs)))
        chan_powers_mat = np.empty((self.n_iter, subset_size, len(freqs)))

        def iter_task():
            data = self.data[0](self.data[1])

            loc_array = np.sort(np.random.choice(data.shape[1], subset_size, replace=False))
            subset = np.array(data.iloc[:,loc_array]) #converted to array for testing jit
            self.logger.debug('starting fft')
            ft_results = ft(subset, **self.ft_kwargs)
            self.logger.debug('starting pca')
            with Pool(processes=1) as pool:
                res = pool.apply_async(pca, (subset, n_pc_sub))
                while True:
                    try:
                        pca_results = res.get(timeout=10)
                        break
                    except TimeoutError:
                        self.logger.warning('pool async timed out')

            del subset, data
            gc.collect()
            return (pca_results, *ft_results)
        
        # TODO make sure n_jobs doesn't need to correspond to num delayed tasks
        if self.parallel:
            results = []
            n_desired, n_available = 10, cpu_count()
            n_batch = self.n_iter//n_desired
            for _ in range(n_batch):
                self.logger.info('commencing a batch')
                curr_results = Parallel(n_jobs=n_available)(delayed(iter_task)() for _ in range(n_desired)) # n_parallel_jobs should be equal to n_jobs
                self.logger.info(f'results len {len(curr_results)}')
                results = [*results, *curr_results]
                time.sleep(1)
        else: #probably local testing:
            results = [iter_task() for _ in range(self.n_iter)]


        evals, powers_sum, powers_chans = list(zip(*results))
        for i in range(self.n_iter):
            evals_mat[i] = evals[i]
            sum_powers_mat[i]  = powers_sum[i]
            chan_powers_mat[i] = powers_chans[i]
            
        es_fooof_kwargs, psd_fooof_kwargs = self.fooof_kwargs.get('es', {}), self.fooof_kwargs.get('psd', {})
        es_fit   = fooofy(pcs, evals_mat,      pc_range_sub, **es_fooof_kwargs)
        psd_fit1 = fooofy(freqs,  sum_powers_mat, self.f_range,  **psd_fooof_kwargs)
        include_psd_fit2 = True
        try:
            psd_fit2_list= [fooofy(freqs, chan_powers_mat[:,chan], self.f_range, **psd_fooof_kwargs) 
                                                            for chan in range(subset_size)] # list comp here iterates over each neuron
            psd_fit2 = defaultdict(lambda: [])
            for d in psd_fit2_list:
                for key, params in d.items():
                    psd_fit2[key].append(params)
            psd_fit2 = {k:np.mean(v, axis=0) for k,v in psd_fit2.items()}
        except Exception as e: # Fitting fails because the input power spectra data is mostly 0s (after logging, contains NaNs or Infs) 
            self.logger.warning(f'could not get summed psd at {subset_size}')
            include_psd_fit2 = False

        spectra = {'evals':evals_mat,'psd':sum_powers_mat, 'psd_chan':chan_powers_mat}
        fit_dict = {**{f'es_{k}'   :v for k,v in es_fit.items()}, # renaming keys for each measure
                    **{f'psd_{k}1':v for k,v in psd_fit1.items()}}
        if include_psd_fit2:
            fit_dict = {**fit_dict, 
                        **{f'psd_{k}2':v for k,v in psd_fit2.items()}}

        return spectra, fit_dict