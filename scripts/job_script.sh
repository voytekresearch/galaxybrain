#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising"
#PBS -l nodes=1:ppn=28
#PBS -l walltime=32:00:00
#PBS -o /home/brirry/logs/analysis.log
#PBS -e /home/brirry/logs/analysis.log
#PBS -V
#PBS -m abe
ROOT="/home/brirry/galaxybrain"

python ${ROOT}/scripts/analysis_pipe.py -i >> /home/brirry/logs/analysis.log 2>&1