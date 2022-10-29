#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising"
#PBS -l nodes=4:ppn=25
#PBS -l walltime=32:00:00
#PBS -o /home/brirry/logs/out.log
#PBS -e /home/brirry/logs/err.log
#PBS -V
#PBS -M bfbarry@ucsd.edu
#PBS -m abe
ROOT="/home/brirry/galaxybrain"

# python ${ROOT}/scripts/analysis_pipe.py -i >> /home/brirry/logs/analysis.log 2>&1

mpirun -v -machinefile $PBS_NODEFILE -np 4 --map-by ppr:1:node python ${ROOT}/scripts/analysis_pipe.py -i -p >> /home/brirry/logs/analysis.log 2>&1