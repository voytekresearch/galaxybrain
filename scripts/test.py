from joblib import Parallel, delayed, cpu_count
import time
from datetime import datetime


from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_proc = comm.Get_size()


def computation():
    time.sleep(5)
    return f'{datetime.now().strftime("%H:%M:%S")}'


def function():
    return Parallel(n_jobs=3)(delayed(computation)() for _ in range(3))


if my_rank != 0:
    comm.send(function(), dest=0)
else:
    for proc_id in range(1,num_proc):
        print(comm.recv(source=proc_id))

"""mpirun -np 4 python script.py"""