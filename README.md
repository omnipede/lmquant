
# Benchmark
```sh
BASE_URL=http://172.100.100.23:8001 MODEL=google/gemma-4-31B-it NUM_PROMPTS=128 INPUT_LEN=4096 OUTPUT_LEN=4096 REQUEST_RATES=inf MAX_CONCURRENCY_LIST="64" EXTRA_ARGS="--ignore-eos" sbatch benchmark_vllm_serve.sh
```