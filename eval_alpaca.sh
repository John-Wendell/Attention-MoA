
# 配置变量 - 生成阶段
MODEL="aws.claude-sonnet-4.5"
OUTPUT_PATH="outputs/alpaca/amoa-r_1-model_5-agg_claude-ful.json"
REFERENCE_MODELS="deepseek-v3.1-huawei,qwen-max-latest,gemini-2.5-pro,gpt-4.1,aws.claude-sonnet-4.5"
ROUNDS=1
NUM_PROC=40
MAX_TOKENS=4096
CACHE_DIR='amoa-alpaca-cache-refer_5-agg_claude-ful'

# 配置变量 - 评估阶段
EVAL_MODEL_OUTPUTS="${OUTPUT_PATH}"
EVAL_REFERENCE_OUTPUTS="alpaca_eval/results/gpt4_1106_preview/model_outputs.json"
EVAL_OUTPUT_PATH="leaderboard-amoa-r_1-model_5-agg_claude-ful.json"

# 运行生成脚本
python generate_for_alpaca_amoa_api_cache.py \
    --model="${MODEL}" \
    --output-path="${OUTPUT_PATH}" \
    --reference-models="${REFERENCE_MODELS}" \
    --rounds ${ROUNDS} \
    --num-proc ${NUM_PROC} \
    --max_tokens ${MAX_TOKENS} \
    --cache_dir "${CACHE_DIR}"

# 运行评估
alpaca_eval --model_outputs "${EVAL_MODEL_OUTPUTS}" --reference_outputs "${EVAL_REFERENCE_OUTPUTS}" --output_path "${EVAL_OUTPUT_PATH}"
