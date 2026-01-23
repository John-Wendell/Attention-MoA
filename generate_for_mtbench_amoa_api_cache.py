"""Generate answers with local models and API models with caching and attention mechanisms.

Usage:
python3 generate_for_mt_bench_attention_res_api_cache.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures
import threading
import hashlib
from typing import List
from concurrent.futures import ThreadPoolExecutor

import shortuuid
from loguru import logger
import tqdm

# FastChat imports
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.llm_judge.gen_model_answer import reorg_answer_file

from utils import (
    generate_together,
    generate_openai,
    generate_with_references_A,
    generate_with_references_AKV,
    generate_with_references_final,
    generate_with_res,
    generate_friday,
    DEBUG,
)

# Global lock for file writing to prevent race conditions
answer_file_lock = threading.Lock()

# Add cache-related functions
def get_cache_key(question_id: str, turn_index: int, i_round: int) -> str:
    """Generate cache key"""
    key_str = f"{question_id}_{turn_index}_{i_round}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_references_cache_path(cache_dir: str, reference_model: str, cache_key: str) -> str:
    """Get references cache file path"""
    model_cache_dir = os.path.join(cache_dir, "references", reference_model)
    return os.path.join(model_cache_dir, f"{cache_key}.json")

def get_final_output_cache_path(cache_dir: str, model: str, cache_key: str) -> str:
    """Get final output cache file path"""
    model_cache_dir = os.path.join(cache_dir, "final_output", model)
    return os.path.join(model_cache_dir, f"{cache_key}.json")

def get_res_output_cache_path(cache_dir: str, model: str, cache_key: str) -> str:
    """Get res output cache file path"""
    model_cache_dir = os.path.join(cache_dir, "res_output", model)
    return os.path.join(model_cache_dir, f"{cache_key}.json")


def load_references_from_cache(cache_path: str) -> List[str]:
    """Load references list from cache"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return [json.load(f)]
        except Exception as e:
            logger.warning(f"Failed to load references cache from {cache_path}: {e}")
    return None

def save_references_to_cache(cache_path: str, references: List[str]) -> None:
    """Save references list to cache"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            if references:
                json.dump(references[0], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save references cache to {cache_path}: {e}")

def load_text_from_cache(cache_path: str) -> str:
    """Load text from cache"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load text cache from {cache_path}: {e}")
    return None

def save_text_to_cache(cache_path: str, text: str) -> None:
    """Save text to cache"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(text, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save text cache to {cache_path}: {e}")

def _process_attention_for_model(reference_model, reference_models, qs, messages, temperature, max_tokens):
    """Process attention computation for a single reference_model"""
    ats_for_model = {}
    k = qs[reference_model]
    for reference_query_model in reference_models:
        q = qs[reference_query_model]
        if (q is not None) and (reference_model != reference_query_model):
            a = generate_with_references_A(
                model=reference_model,
                messages=messages,
                q=q,
                k=k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            a = None
        ats_for_model[reference_query_model] = a
    return reference_model, ats_for_model

def _process_AKV_for_model(reference_model, reference_models, ats, qs, messages, temperature, max_tokens):
    """Process final answer computation for a single reference_model"""
    ats4reference_model = []
    for reference_A_model in reference_models:
        a = ats[reference_A_model][reference_model]
        if a is not None:
            ats4reference_model.append(a)
    q = qs[reference_model]
    reference = generate_with_references_AKV(
        model=reference_model,
        messages=messages,
        ats4reference_model=ats4reference_model,
        q=q,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return reference



def get_answer(
    question: dict,
    model: str,
    reference_models: List[str],
    num_choices: int,
    max_tokens: int,
    answer_file: str,
    rounds: int,
    provider: str,
    cache_dir: str = None,
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []

    if provider == "together":
        generate_fn = generate_together
    elif provider == "openai":
        generate_fn = generate_openai
    elif provider == "friday":
        generate_fn = generate_friday
    else:
        assert False


    for i in range(num_choices):

        turns = []
        messages = []

        for j in range(len(question["turns"])):

            qs = question["turns"][j]
            
            messages.append({"role": "user", "content": qs})
            # print('origin messages: ',messages)
            references = []
            history = []

            # Initialize akv_futures to avoid scope issues
            akv_futures = {}

            if len(reference_models) > 0:

                prev_references = []

                for i_round in range(rounds):
                    # print(question)
                    if DEBUG:
                        logger.info(
                            f"Round {i_round+1}/{rounds} to collecting reference responses."
                        )

                    # Generate cache key for current round
                    cache_key = get_cache_key(question["question_id"], j, i_round)
                    
                    cached_references_map = {}
                    missing_models = []
                    
                    if cache_dir:
                        for reference_model in reference_models:
                            cache_path = get_references_cache_path(cache_dir, reference_model, cache_key)
                            cached_ref = load_references_from_cache(cache_path)
                            
                            if cached_ref is not None:
                                logger.info(f"Loading references from cache for {reference_model}: {cache_path}")
                                cached_references_map[reference_model] = cached_ref[0]
                            else:
                                missing_models.append(reference_model)
                    else:
                        missing_models = list(reference_models)
                    
                    all_cached = (len(missing_models) == 0)

                    if all_cached:
                        references = [cached_references_map[m] for m in reference_models]
                        logger.info(f"All references loaded from cache for round {i_round}")
                    else:
                        references = []
                        
                        ### QKV computation ###
                        qs = {}
                        # Compute QKV for all reference_models in parallel
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            qkv_futures = {
                                reference_model: executor.submit(
                                    generate_with_references_final,
                                    model=reference_model,
                                    messages=messages,
                                    references=prev_references,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                )
                                for reference_model in reference_models
                            }
                            
                            for reference_model, future in qkv_futures.items():
                                qs[reference_model] = future.result()

                        ### Attention computation ###
                        ats = {}
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            attn_futures = [
                                executor.submit(
                                    _process_attention_for_model,
                                    reference_model,
                                    reference_models,
                                    qs,
                                    messages,
                                    temperature,
                                    max_tokens,
                                )
                                for reference_model in reference_models
                            ]
                            for future in attn_futures:
                                ref_model, ats_for_model = future.result()
                                ats[ref_model] = ats_for_model

                        ### A*KV Final Answer Computation ###
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            # Reset akv_futures for this round
                            akv_futures = {}
                            for reference_model in missing_models:
                                future = executor.submit(
                                    _process_AKV_for_model,
                                    reference_model,
                                    reference_models,
                                    ats,
                                    qs,
                                    messages,
                                    temperature,
                                    max_tokens,
                                )
                                akv_futures[reference_model] = future
                            
                            for reference_model in reference_models:
                                if reference_model in cached_references_map:
                                    references.append(cached_references_map[reference_model])
                                else:
                                    reference = akv_futures[reference_model].result()
                                    if reference is not None:
                                        references.append(reference)
                                        if cache_dir:
                                            cache_path = get_references_cache_path(cache_dir, reference_model, cache_key)
                                            save_references_to_cache(cache_path, [reference])

                    # Generate final output
                    final_cache_key = get_cache_key(f"{question['question_id']}_final", j, i_round)
                    final_cache_path = get_final_output_cache_path(cache_dir, model, final_cache_key) if cache_dir else None
                    
                    cached_output = None
                    if all_cached and final_cache_path:
                        cached_output = load_text_from_cache(final_cache_path)
                    
                    final_output_from_cache = False

                    if cached_output is not None:
                        logger.info(f"Loading final output from cache: {final_cache_path}")
                        output = cached_output
                        final_output_from_cache = True
                    else:
                        try:
                            output = generate_with_references_final(
                                model=model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                generate_fn=generate_fn,
                                references=references,
                            ).strip()
                        except Exception as e:
                            logger.warning(f"generate_with_references_final failed: {e}")
                            output = None
                            # Try to read the result from references for the model with the same name
                            if model in reference_models:
                                try:
                                    # Find the index of model in reference_models
                                    model_index = reference_models.index(model)
                                    # Get the result from the corresponding position in references
                                    if model_index < len(references):
                                        output = references[model_index]
                                        logger.info(f"Using reference result for model {model} from references list")
                                except (ValueError, IndexError) as idx_err:
                                    logger.warning(f"Failed to get reference result for model {model}: {idx_err}")
                                    output = None

                            # If getting from references fails, try to get from cache
                            if output is None and model in reference_models:
                                if model in cached_references_map:
                                    output = cached_references_map[model]
                                    logger.info(f"Using cached reference result for model {model}")
                                elif model in akv_futures:
                                    try:
                                        output = akv_futures[model].result()
                                        logger.info(f"Using future result for model {model}")
                                    except Exception as future_err:
                                        logger.warning(f"Failed to get future result for model {model}: {future_err}")
                                        output = None

                            if output is None:
                                output = ""
                                logger.warning(f"All fallback methods failed for model {model}, using empty string")

                        if final_cache_path:
                            save_text_to_cache(final_cache_path, output)

                    if i_round == 0:
                        prev_references = [output]
                        history.append(output)
                        references = []
                    else:
                        # Generate res output
                        res_cache_key = get_cache_key(f"{question['question_id']}_res", j, i_round)
                        res_cache_path = get_res_output_cache_path(cache_dir, model, res_cache_key) if cache_dir else None
                        
                        cached_res_output = None
                        if final_output_from_cache and res_cache_path:
                            cached_res_output = load_text_from_cache(res_cache_path)
                        
                        if cached_res_output is not None:
                            logger.info(f"Loading res output from cache: {res_cache_path}")
                            res_output = cached_res_output
                        else:
                            res_output = generate_with_res(
                                model=model,
                                messages=messages,
                                max_tokens=max_tokens,
                                pre_output=history,
                                cur_output=[output],
                                generate_fn=generate_fn,
                            ).strip()
                            if res_cache_path:
                                save_text_to_cache(res_cache_path, res_output)


                        if "AMoA should be stopped" in res_output:
                            print('Early Stopped')
                            break
                        else:
                            output = res_output
                            prev_references = [output]
                            history.append(output)
                            references = []
            else:
                try:
                    output = generate_with_references_final(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        generate_fn=generate_fn,
                        references=references,
                    ).strip()
                except Exception as e:
                    logger.warning(f"generate_with_references_final failed (no reference models): {e}")
                    output = ""
            # print('output: ',output)
            messages.append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    # Use lock for thread-safe file writing
    with answer_file_lock:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--reference-models", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--provider", type=str, default="together")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="The directory to store cache."
    )
    args = parser.parse_args()

    question_file = f"FastChat/fastchat/llm_judge/data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"outputs/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file}")

    if args.reference_models is None:
        reference_models = []
    else:
        reference_models = args.reference_models.split(",")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                reference_models,
                args.num_choices,
                args.max_tokens,
                answer_file,
                args.rounds,
                args.provider,
                args.cache_dir,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
