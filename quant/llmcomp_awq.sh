#!/bin/bash
#SBATCH --job-name=awq
#SBATCH --comment="AWQ quantization"
#SBATCH --nodes=1
#SBATCH --nodelist=cubox01,cubox03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=112
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/quant_awq_%j.log

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# Huggingface cache directory
export HF_HOME=./cache/hf

srun python -m quant.llmcomp_awq