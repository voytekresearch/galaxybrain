#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_test2"
#PBS -l nodes=1:ppn=7  
#PBS -l walltime=32:00:00
#PBS -o output1.txt
#PBS -e error1.txt
#PBS -V
#PBS -m abe
cd ./galaxybrain
d="mouse"
if [ "$d" = "mouse" ]
then
    echo "running analysis_pipe.py"
    python analysis_pipe.py
elif [ "$d" = "noise" ]
then
    python noise_pipe.py
else
    echo "running sim_analysis.py"
    python ising_pipe.py
fi
