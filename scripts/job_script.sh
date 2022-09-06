#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising"
#PBS -l nodes=1:ppn=7  
#PBS -l walltime=32:00:00
#PBS -o output.out
#PBS -e error.err
#PBS -V
#PBS -m abe
ROOT="/home/brirry/galaxybrain"

python ${ROOT}/scripts/ising_pipe.py -a
