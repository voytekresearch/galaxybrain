#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising_mpi"
#PBS -l nodes=4:ppn=10
#PBS -l walltime=1:00:00
#PBS -o /home/brirry/logs/out.log
#PBS -e /home/brirry/logs/err.log
#PBS -V
#PBS -M barry.brian.f@gmail.com
#PBS -m abe

rm -rf /home/brirry/galaxybrain/data/experiments/ising
mkdir /home/brirry/galaxybrain/data/experiments/ising
echo "`date` BEGIN \n" >> /home/brirry/logs/analysis.log
cd /home/brirry/galaxybrain && mpirun -v -machinefile $PBS_NODEFILE -np 4 --map-by ppr:10:node python /home/brirry/galaxybrain/main.py -i -p  >> /home/brirry/logs/analysis.log 2>&1
