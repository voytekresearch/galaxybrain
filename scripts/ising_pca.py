from ..galaxybrain.ramsey import PCA
import h5py
import numpy as np

ising = h5py.File('../data/spikes/ising.hdf5', 'r') # keys := ['1.97', '2.07', '2.17', '2.27', '2.37', '2.47', '2.57']

for temp in ising:
    tensor = np.array(ising[temp]) # shape := (Time x N x N)
    flattened = tensor.reshape(tensor.shaoe[0], -1) # shape := (Time x N^2)
    

