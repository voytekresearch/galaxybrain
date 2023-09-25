import sys
import subprocess
if 'sdsc' in subprocess.run('hostname', capture_output=True).stdout.decode('utf8'):
    sys.path.append('/home/brirry/galaxybrain')
import argparse
from scripts.analysis_pipe import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mouse', action='store_true')
    parser.add_argument('-t', dest='test',  action='store_true') # test mouse
    parser.add_argument('-i', dest='ising', action='store_true')
    parser.add_argument('-p', dest='mpi', action='store_true')
    cl_args = parser.parse_args()
    main(cl_args)