#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising"
#PBS -l nodes=1:ppn=8 
#PBS -l walltime=32:00:00
#PBS -o /home/brirry/logs/out.log
#PBS -e /home/brirry/logs/err.log
#PBS -V
#PBS -m abe
ROOT="/home/brirry/galaxybrain"

python ${ROOT}/scripts/analysis_pipe.py -m