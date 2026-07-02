#!/bin/bash
#SBATCH --job-name=trt_qnt
#SBATCH --comment="TensorRT-LLM quantization"
#SBATCH --nodes=1
#SBATCH --nodelist=cubox08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/trtllm_quantize_%j.log

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Job name:= ${SLURM_JOB_NAME:-local}"
echo "Job id:= ${SLURM_JOB_ID:-local}"
echo "Nodelist:= ${SLURM_JOB_NODELIST:-local}"
echo "Run started at:-"
date
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

set -euo pipefail

IMAGE="nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc20"
WORKDIR="/purestorage/AILAB/AI_2/hgseo/work/remote/QT"

mkdir -p logs

srun \
    --mpi=pmix \
    --container-image="${IMAGE}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    --container-workdir="${WORKDIR}" \
    --container-writable \
    --no-container-entrypoint \
    bash -lc '
        export PMIX_MCA_gds=hash;
        export HF_HOME=./cache/hf
        ulimit -l unlimited;
        ulimit -s 65536;
        echo "Container shell on $(hostname)";
        python quant_trtllm/trtllm_quantize.py --model_dir google/gemma-4-31B-it \
            --dtype bfloat16 \
            --qformat fp8_pc_pt \
            --output_dir ./models/gemma-4-31B-it-TRT_FP8 \
            --tp_size 2
    '
