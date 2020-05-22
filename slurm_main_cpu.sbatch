#!/bin/bash
#
#SBATCH --job-name=ice_cpu
#SBATCH --output=slurm_log/ice_cpu.%A.out
#SBATCH --error=slurm_log/ice_cpu.%A.err
#
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=8G
#SBATCH --time=0-12:00
#

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2

if [ -z ${SLURM_ARRAY_TASK_ID} ]
then
        echo "SLURM array variable not set"
        python main.py "$@"
else
        echo "SLURM array variable is set"
        echo "${SLURM_ARRAY_TASK_ID}"
        python main.py --seed ${SLURM_ARRAY_TASK_ID} --n-sims 1 "$@"
fi

