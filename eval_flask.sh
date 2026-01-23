MODEL="gpt-4.1"
OUTPUT_PATH="outputs/flask/amoa-api-ref_5-agg_gpt-r1-ful.jsonl"
REFERENCE_MODELS="deepseek-v3.1-huawei,qwen-max-latest,gemini-2.5-pro,gpt-4.1,aws.claude-sonnet-4.5"
ROUNDS=1
NUM_PROC=20
CACHE_DIR='amoa-api-flask-refer_5-agg_gpt-ful-cacehe'

FLASK_DIR="FLASK/gpt_review"
EVAL_QUESTIONS="../evaluation_set/flask_hard_evaluation.jsonl"
EVAL_ANSWERS="../../${OUTPUT_PATH}"
EVAL_OUTPUT="../../outputs/flask/amoa-api-ref_5-agg_gpt-r1-ful-eval.jsonl"
AGGREGATE_INPUT="../../outputs/flask/amoa-api-ref_5-agg_gpt-r1-ful-eval.jsonl"
STATS_OUTPUT="outputs/stats/amoa-api-ref_5-agg_gpt-r1-ful-eval_skill.csv"

mkdir -p outputs/flask


python generate_for_flask_amoa_api_cache.py \
    --model="${MODEL}" \
    --output-path="${OUTPUT_PATH}" \
    --reference-models="${REFERENCE_MODELS}" \
    --rounds ${ROUNDS} \
    --num-proc ${NUM_PROC} \
    --cache_dir "${CACHE_DIR}"

cd "${FLASK_DIR}"


python gpt4_eval.py \
    -q "${EVAL_QUESTIONS}" \
    -a "${EVAL_ANSWERS}" \
    -o "${EVAL_OUTPUT}"


python aggregate_skill.py -m "${AGGREGATE_INPUT}"


cat "${STATS_OUTPUT}"
