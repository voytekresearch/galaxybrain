NUM_NODES = 4
NUM_PROC = 10
WALLTIME = "1:00:00"
PROJECT_ROOT = "/home/brirry/galaxybrain"


arg_map = {'mouse' : 'm',
           'test'  : 't',
           'ising' : 'i',
           'mpi'   : 'p',}
ANALYSIS_TYPE = 'ising mpi'
if ANALYSIS_TYPE not in arg_map:
    print(f'ANALYSIS_TYPE {ANALYSIS_TYPE} not in arg_map')
    exit()
ANALYSIS_TYPE = arg_map[ANALYSIS_TYPE]


PBS_config = F"""
#!/bin/bash
#PBS -q hotel
#PBS -N "galaxybrain_ising"
#PBS -l nodes={NUM_NODES}:ppn={NUM_PROC}
#PBS -l walltime={WALLTIME}
#PBS -o /home/brirry/logs/out.log
#PBS -e /home/brirry/logs/err.log
#PBS -V
#PBS -M bfbarry@ucsd.edu
#PBS -m abe
"""

bash_script = f"""

ROOT=\"/home/brirry/galaxybrain\"
LOG_PATH=\"/home/brirry/logs/analysis.log\"
### USAGE
# change PBS -N, exp_dir, and script flag (-m, -i etc)
# python ${ROOT}/scripts/analysis_pipe.py -i >> /home/brirry/logs/analysis.log 2>&1
exp_dir=${ROOT}/data/experiments/ising # this should be an arg to avoid repetition
rm -rf $exp_dir
mkdir $exp_dir
# -np corresponds to num_trials
echo \"`date` BEGIN \n\" >> ${LOG_PATH}
cd $ROOT && mpirun -v -machinefile $PBS_NODEFILE -np 40 --map-by ppr:10:node python ${ROOT}/main.py -{ANALYSIS_TYPE} -p >> ${LOG_PATH} 2>&1
"""

# save out .sh file

# submit with subprocess

# TODO first recreate current script and make sure they match with diff