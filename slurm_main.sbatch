#!/bin/bash
#
#SBATCH --job-name=icebeem
#SBATCH --output=slurm_log/icebeem.%A_%a.out
#SBATCH --error=slurm_log/icebeem.%A_%a.err
#
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --time=0-6:00
#
#SBATCH --gres=gpu:1

if [ -z ${SLURM_ARRAY_TASK_ID} ]
then
        echo "SLURM array variable not set"
        python main.py "$@"
else
        echo "SLURM array variable is set"
        echo "${SLURM_ARRAY_TASK_ID}"
        python main.py --seed ${SLURM_ARRAY_TASK_ID} --nSims 1 "$@"
fi

