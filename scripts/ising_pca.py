from ..galaxybrain.ramsey import pca
import h5py
import numpy as np

ising = h5py.File('../data/spikes/ising.hdf5', 'r') # keys := ['1.97', '2.07', '2.17', '2.27', '2.37', '2.47', '2.57']
PATH = '/home/brirry/ising_data'
with h5py.File(f'{PATH}/ising_evals.hdf5', 'a') as f:
    for temp in ising:
        tensor = np.array(ising[temp]) # shape := (Time x N x N)
        flattened = tensor.reshape(tensor.shape[0], -1) # shape := (Time x N^2)
        evals = pca(flattened) # n_pc will be smallest number of axis in latest case 
        f.create_dataset(f'{temp:.2f}', data=evals, dtype='f')
    
