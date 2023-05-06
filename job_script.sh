#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising"
#PBS -l nodes=4:ppn=5
#PBS -l walltime=1:00:00
#PBS -o /home/brirry/logs/out.log
#PBS -e /home/brirry/logs/err.log
#PBS -V
#PBS -M bfbarry@ucsd.edu
#PBS -m abe
ROOT="/home/brirry/galaxybrain"
LOG_PATH="/home/brirry/logs/analysis.log"
### USAGE
# change PBS -N, exp_dir, and script flag (-m, -i etc)
# python ${ROOT}/scripts/analysis_pipe.py -i >> /home/brirry/logs/analysis.log 2>&1
exp_dir=${ROOT}/data/experiments/ising # this should be an arg to avoid repetition
rm -rf $exp_dir
mkdir $exp_dir
# -np corresponds to num_trials
echo "`date` BEGIN \n" >> ${LOG_PATH}
cd $ROOT && mpirun -v -machinefile $PBS_NODEFILE -np 4 --map-by ppr:1:node python ${ROOT}/main.py -i -p >> ${LOG_PATH} 2>&1