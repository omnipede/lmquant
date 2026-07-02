#!/bin/bash
#SBATCH --job-name=awq_31b
#SBATCH --comment="LLM server"
#SBATCH --nodes=1
#SBATCH --nodelist=cubox01,cubox02,cubox03,cubox04,cubox05,cubox06,cubox07,cubox08,cubox09,cubox10,cubox11,cubox12,cubox13,cubox14,cubox15,cubox16
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/gem4_awq_31b_%j.log

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# Huggingface cache directory
export HF_HOME=./cache/hf

# vLLM config
export VLLM_USE_DEEP_GEMM=0

# Model
MODEL=./models/gemma-4-31B-it-AWQ
SERVED_MODEL_NAME=google/gemma-4-31B-it

# Run vLLM server
python -m vllm.entrypoints.openai.api_server \
    --port 8003 \
    --model $MODEL \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 2 \
    --max-num-seqs 128 \
    --reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4 \
    --disable-custom-all-reduce \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --default-chat-template-kwargs '{"enable_thinking": true}'