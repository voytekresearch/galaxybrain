from joblib import Parallel, delayed, cpu_count
import time
from sklearn.decomposition import PCA
import numpy as np


def pca(data, n_pc=None):
    """
    Decomposition in space
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    pop_pca = PCA(n_pc).fit(data)
    evals = pop_pca.explained_variance_ratio_

    return evals



start= time.perf_counter()
fake_data = np.random.poisson(1, size=(500, 100))
def func():
    print('doing pca!')
    pca(fake_data)

Parallel(n_jobs=5)(delayed(func)() for _ in range(10))
print(time.perf_counter()-start)
