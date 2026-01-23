
MODEL="qwen-max-latest"
REFERENCE_MODELS="deepseek-v3.1-huawei,qwen-max-latest,gemini-2.5-pro,gpt-4.1,aws.claude-sonnet-4.5"
ANSWER_FILE="outputs/mt_bench/amoa-r_1-model_5-agg_qwen-ful.jsonl"
PARALLEL=32
ROUNDS=1
PROVIDER="friday"
MAX_TOKENS=4096
CACHE_DIR='amoa-mtbench-cache-refer_5-agg_qwen-ful'

MODEL_LIST="amoa-r_1-model_5-agg_qwen-ful"
EVAL_PARALLEL=32

python generate_for_mtbench_amoa_api_cache.py \
    --model "${MODEL}" \
    --reference-models "${REFERENCE_MODELS}" \
    --answer-file "${ANSWER_FILE}" \
    --parallel ${PARALLEL} \
    --rounds ${ROUNDS} \
    --provider "${PROVIDER}" \
    --max-tokens ${MAX_TOKENS} \
    --cache-dir "${CACHE_DIR}"


python eval_mt_bench.py --model-list ${MODEL_LIST} --parallel ${EVAL_PARALLEL}

python show_mt_bench_result.py