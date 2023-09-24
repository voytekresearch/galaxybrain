import subprocess
import os
import yaml


with open('pipeline_config.yaml', 'r') as f:
    job_args = yaml.safe_load(f)['job_args']
NUM_NODES = job_args['NUM_NODES']
NUM_PROC = job_args['NUM_PROC']
WALLTIME = job_args['WALLTIME']
PROJECT_ROOT = job_args['PROJECT_ROOT']
LOG_PATH = job_args['LOG_PATH']
ANALYSIS_TYPE = job_args['ANALYSIS_TYPE']

JOB_DESCRIPTOR = '_'.join(ANALYSIS_TYPE)
DATA_DIR = os.path.join(PROJECT_ROOT, job_args['DATA_DIR_POSTFIX'])

arg_map = {'mouse' : 'm',
           'test'  : 't',
           'ising' : 'i',
           'mpi'   : 'p',}
arg_str = ''
for i in ANALYSIS_TYPE:
    if i not in arg_map:
        print(f'ANALYSIS_TYPE {ANALYSIS_TYPE} not in arg_map')
        exit()
    arg_str += f'-{arg_map[i]} '

sh_script = F"""#!/bin/bash
#PBS -q hotel
#PBS -N \"galaxybrain_{JOB_DESCRIPTOR}\"
#PBS -l nodes={NUM_NODES}:ppn={NUM_PROC}
#PBS -l walltime={WALLTIME}
#PBS -o /home/brirry/logs/out.log
#PBS -e /home/brirry/logs/err.log
#PBS -V
#PBS -M barry.brian.f@gmail.com
#PBS -m abe

rm -rf {DATA_DIR}/*
echo \"`date` BEGIN \\n\" >> {LOG_PATH}
cd {PROJECT_ROOT} && mpirun -v -machinefile $PBS_NODEFILE -np {NUM_NODES} --map-by ppr:{NUM_PROC}:node python {PROJECT_ROOT}/main.py {arg_str} >> {LOG_PATH} 2>&1
"""

# save out .sh file
with open('job_script.sh', 'w') as f:
    f.write(sh_script)

res = subprocess.run('qsub job_script.sh', shell=True, capture_output=True, text=True)
print(res.stdout, res.stderr)
# submit with subprocess