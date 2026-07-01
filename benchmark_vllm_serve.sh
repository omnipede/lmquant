#!/bin/bash
#SBATCH --job-name=bnch_llm
#SBATCH --comment="vLLM OpenAI-compatible server benchmark"
#SBATCH --nodes=1
#SBATCH --nodelist=nv178
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=logs/bench_vllm_%j.log

set -euo pipefail

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Job name:= ${SLURM_JOB_NAME:-local}"
echo "Job id:= ${SLURM_JOB_ID:-local}"
echo "Nodelist:= ${SLURM_JOB_NODELIST:-local}"
echo "Run started at:-"
date
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Target OpenAI-compatible server.
BASE_URL="${BASE_URL:-http://172.100.100.29:8000}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
BACKEND="${BACKEND:-openai-chat}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
API_KEY="${API_KEY:-EMPTY}"

# Synthetic workload defaults. These map to vLLM's random dataset options.
DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
NUM_WARMUPS="${NUM_WARMUPS:-16}"
REQUEST_RATES="${REQUEST_RATES:-1 2 4 8 16 inf}"
MAX_CONCURRENCY_LIST="${MAX_CONCURRENCY_LIST:-1 2 4 8 16 32}"

# Metrics and output.
METRIC_PERCENTILES="${METRIC_PERCENTILES:-50,90,95,99}"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}"
READY_CHECK_TIMEOUT_SEC="${READY_CHECK_TIMEOUT_SEC:-300}"
RESULT_DIR="${RESULT_DIR:-logs/benchmarks/${SLURM_JOB_ID:-local}}"
LABEL_PREFIX="${LABEL_PREFIX:-vllm-serve}"
SAVE_DETAILED="${SAVE_DETAILED:-1}"
PLOT_TIMELINE="${PLOT_TIMELINE:-0}"
PLOT_DATASET_STATS="${PLOT_DATASET_STATS:-0}"

# Optional pass-through knobs. Examples:
#   EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
#   GOODPUT_SLO='ttft:500 tpot:50 e2el:10000'
#   METADATA='model_variant=fp8 tp=8'
#   EXTRA_ARGS='--temperature 0.0 --ignore-eos'
EXTRA_BODY="${EXTRA_BODY:-}"
GOODPUT_SLO="${GOODPUT_SLO:-}"
METADATA="${METADATA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs "$RESULT_DIR"
export OPENAI_API_KEY="$API_KEY"

echo "Benchmark configuration:"
echo "  BASE_URL=${BASE_URL}"
echo "  MODEL=${MODEL}"
echo "  BACKEND=${BACKEND}"
echo "  ENDPOINT=${ENDPOINT}"
echo "  DATASET_NAME=${DATASET_NAME}"
echo "  INPUT_LEN=${INPUT_LEN}"
echo "  OUTPUT_LEN=${OUTPUT_LEN}"
echo "  NUM_PROMPTS=${NUM_PROMPTS}"
echo "  NUM_WARMUPS=${NUM_WARMUPS}"
echo "  REQUEST_RATES=${REQUEST_RATES}"
echo "  MAX_CONCURRENCY_LIST=${MAX_CONCURRENCY_LIST}"
echo "  PERCENTILE_METRICS=${PERCENTILE_METRICS}"
echo "  METRIC_PERCENTILES=${METRIC_PERCENTILES}"
echo "  RESULT_DIR=${RESULT_DIR}"

COMMON_ARGS=(
    --backend "$BACKEND"
    --base-url "$BASE_URL"
    --endpoint "$ENDPOINT"
    --model "$MODEL"
    --dataset-name "$DATASET_NAME"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --num-warmups "$NUM_WARMUPS"
    --ready-check-timeout-sec "$READY_CHECK_TIMEOUT_SEC"
    --percentile-metrics "$PERCENTILE_METRICS"
    --metric-percentiles "$METRIC_PERCENTILES"
    --save-result
    --result-dir "$RESULT_DIR"
)

if [[ "$SAVE_DETAILED" == "1" ]]; then
    COMMON_ARGS+=(--save-detailed)
fi

if [[ "$PLOT_TIMELINE" == "1" ]]; then
    COMMON_ARGS+=(--plot-timeline)
fi

if [[ "$PLOT_DATASET_STATS" == "1" ]]; then
    COMMON_ARGS+=(--plot-dataset-stats)
fi

if [[ -n "$EXTRA_BODY" ]]; then
    COMMON_ARGS+=(--extra-body "$EXTRA_BODY")
fi

if [[ -n "$GOODPUT_SLO" ]]; then
    # shellcheck disable=SC2206
    GOODPUT_ARGS=($GOODPUT_SLO)
    COMMON_ARGS+=(--goodput "${GOODPUT_ARGS[@]}")
fi

if [[ -n "$METADATA" ]]; then
    # shellcheck disable=SC2206
    METADATA_ARGS=($METADATA)
    COMMON_ARGS+=(--metadata "${METADATA_ARGS[@]}")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_CLI_ARGS=($EXTRA_ARGS)
    COMMON_ARGS+=("${EXTRA_CLI_ARGS[@]}")
fi

for request_rate in $REQUEST_RATES; do
    for max_concurrency in $MAX_CONCURRENCY_LIST; do
        safe_rate="${request_rate//./p}"
        result_filename="${LABEL_PREFIX}-rps_${safe_rate}-conc_${max_concurrency}.json"

        echo
        echo "============================================================"
        echo "Running vllm bench serve: request_rate=${request_rate}, max_concurrency=${max_concurrency}"
        echo "Result file: ${RESULT_DIR}/${result_filename}"
        echo "Started at: $(date)"
        echo "============================================================"

        vllm bench serve \
            "${COMMON_ARGS[@]}" \
            --request-rate "$request_rate" \
            --max-concurrency "$max_concurrency" \
            --label "${LABEL_PREFIX}-rps_${safe_rate}-conc_${max_concurrency}" \
            --result-filename "$result_filename"

        echo "Finished at: $(date)"
    done
done

echo
echo "Benchmark completed at:-"
date
echo "Results saved under: ${RESULT_DIR}"
