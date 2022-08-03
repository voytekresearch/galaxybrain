import sys
import pathlib
import os
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(), 'galaxybrain'))
#print(pathlib.Path(__file__).parent.resolve().parent.resolve())
from ramsey import pca
import h5py
import numpy as np

ising = h5py.File('/home/brirry/galaxybrain/data/spikes/ising.hdf5', 'r') # keys := ['1.97', '2.07', '2.17', '2.27', '2.37', '2.47', '2.57']
PATH = '/home/brirry/galaxybrain/data/experiments/ising_pca'
if not os.path.exists(PATH): os.makedirs(PATH)
with h5py.File(f'{PATH}/ising_evals.hdf5', 'a') as f:
    for temp in ising:
        tensor = np.array(ising[temp]) # shape := (Time x N x N)
        flattened = tensor.reshape(tensor.shape[0], -1) # shape := (Time x N^2)
        evals = pca(flattened) # n_pc will be smallest number of axis in latest case
        f.create_dataset(f'{temp:.2f}', data=evals, dtype='f')