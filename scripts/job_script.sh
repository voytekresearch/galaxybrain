#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising"
#PBS -l nodes=4:ppn=10
#PBS -l walltime=1:00:00
#PBS -o /home/brirry/logs/out.log
#PBS -e /home/brirry/logs/err.log
#PBS -V
#PBS -M bfbarry@ucsd.edu
#PBS -m abe
ROOT="/home/brirry/galaxybrain"

# python ${ROOT}/scripts/analysis_pipe.py -i >> /home/brirry/logs/analysis.log 2>&1
exp_dir=${ROOT}/data/experiments/ising_better_fit # this should be an arg to avoid repetition
rm -rf $exp_dir
mkdir $exp_dir
mpirun -v -machinefile $PBS_NODEFILE -np 4 --map-by ppr:1:node python ${ROOT}/scripts/analysis_pipe.py -i -p >> /home/brirry/logs/analysis.log 2>&1