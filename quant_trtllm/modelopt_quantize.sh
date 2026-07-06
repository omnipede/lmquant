#!/bin/bash
#SBATCH --job-name=mopt_fp8
#SBATCH --comment="Per tensor, per token FP8 quantization"
#SBATCH --nodes=1
#SBATCH --nodelist=cubox02,cubox03,cubox04
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/modelopt_quantize_%j.log

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# Huggingface cache directory
export HF_HOME=./cache/hf

srun python -m quant_trtllm.modelopt_quantize