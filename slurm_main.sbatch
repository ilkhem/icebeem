#!/bin/bash
#
#SBATCH --job-name=augice
#SBATCH --output=slurm_log/augice.%A_%a.out
#SBATCH --error=slurm_log/augice.%A_%a.err
#
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH --time=0-23:00
#
#SBATCH --gres=gpu:1
#SBATCH --exclusive

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

