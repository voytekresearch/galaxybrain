from joblib import Parallel, delayed, cpu_count
import time
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import time
import argparse



"""plan:
- time one run of PCA on the ising matrix in separate script
- see if joblib speeds this up on cluster
- see if multi trials with MPI speeds even more

"""


def computation():
    # emulate holding it in memory with subsample here
    start = time.perf_counter()
    d = np.copy(DATA)
    pca = PCA(int(N_PC*min(d.shape)))
    pop_pca = pca.fit(d)
    evals = pop_pca.explained_variance_ratio_
    
    duration = f'{time.perf_counter() - start:.1f}'
    end_time = datetime.now().strftime("%H:%M:%S")
    return (f'{duration}s at {end_time}')


def iter_computation():
    return Parallel(n_jobs=N_CORE)(delayed(computation)() for _ in range(N_CORE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='serial', action='store_true')
    parser.add_argument('-j', dest='joblib_only', action='store_true')
    parser.add_argument('-m', dest='mpi_and_joblib', action='store_true')
    cl_args = parser.parse_args()
    
    N_PC = 0.8
    N_CORE = 3#cpu_count() # also number of sub iterations
    DATA = np.random.randint(0,2,size=(10000,4096))

    ### test runtime one time
    if cl_args.serial:
        print(computation())
    elif cl_args.joblib_only:
        print(f'Working with {N_CORE} ppn')
        print(iter_computation())
    elif cl_args.mpi_and_joblib:
        print(f'Working with {N_CORE} ppn')
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        my_rank = comm.Get_rank()
        num_proc = comm.Get_size()
        if my_rank != 0:
            comm.send(iter_computation(), dest=0)
        else:
            for proc_id in range(1,num_proc):
                print(comm.recv(source=proc_id))
