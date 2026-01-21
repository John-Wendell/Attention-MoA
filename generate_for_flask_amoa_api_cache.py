import json
import os
import hashlib
import datasets
from fire import Fire
from functools import partial
from typing import List
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    generate_with_references_QKV,
    generate_with_references_A,
    generate_with_references_A_self,
    generate_with_references_AKV,
    generate_with_references_final,
    generate_with_res,
    DEBUG,
)


def get_cache_key(question_id: str, i_round: int) -> str:
    """Generate cache key"""
    key_str = f"{question_id}_{i_round}"
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
    """Process attention computation for a single reference_model (inner loop remains serial)"""
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
                # temperature=temperature,
                max_tokens=max_tokens,
            )
        elif (q is not None) and (reference_model == reference_query_model):
            a = generate_with_references_A_self(
                model=reference_model,
                messages=messages,
                q=q,
                # temperature=temperature,
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
        # temperature=temperature,
        max_tokens=max_tokens,
    )
    return reference


def _process_Q_for_model(reference_model, messages, prev_references, temperature, max_tokens):
    """Process Q generation for a single reference_model"""
    reference = generate_with_references_final(
        model=reference_model,
        messages=messages,
        references=prev_references,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return reference_model, reference


def process_fn(
    item,
    model,
    reference_models=None,
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
    num_workers=4,
    cache_dir=None,
):
    # Handle mutable default parameters
    if reference_models is None:
        reference_models = []

    messages = [{"role": "user", "content": item["text"]}]
    question_id = item["question_id"]

    references = item.get("references", [])

    history = []
    output = None  # Initialize output variable to prevent undefined error

    if len(references) == 0 and len(reference_models) > 0:

        prev_references = []

        for i_round in range(rounds):

            if DEBUG:
                logger.info(
                    f"Round {i_round+1}/{rounds} to collecting reference responses."
                )

            # Generate cache key for current round
            cache_key = get_cache_key(question_id, i_round)

            # Try to load references for all reference_models from cache

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
                # All references are in cache, use them directly
                references = [cached_references_map[m] for m in reference_models]
                logger.info(f"All references loaded from cache for round {i_round}")
            else:
                # Need to recompute missing parts
                references = []

                ### QKV computation ###
                qs = {}
                # Note: Even if some models already have cached results, we still need their Q to compute Attention
                # Therefore, generate Q for all reference_models here
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            _process_Q_for_model,
                            reference_model,
                            messages,
                            prev_references,
                            temperature,
                            max_tokens,
                        )
                        for reference_model in reference_models
                    ]
                    for future in futures:
                        ref_model, reference = future.result()
                        qs[ref_model] = reference

                ### Attention computation (parallel processing of outer loop) ###
                ats = {}
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
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
                    for future in futures:
                        ref_model, ats_for_model = future.result()
                        ats[ref_model] = ats_for_model

                ### A*KV Final Answer Computation ###
                # Modified: only compute missing_models, use cached ones directly

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {}
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
                        futures[reference_model] = future

                    # Assemble references in the order of reference_models
                    for reference_model in reference_models:
                        if reference_model in cached_references_map:
                            references.append(cached_references_map[reference_model])
                        else:
                            reference = futures[reference_model].result()
                            if reference is not None:
                                references.append(reference)
                                # Save to cache
                                if cache_dir:
                                    cache_path = get_references_cache_path(cache_dir, reference_model, cache_key)
                                    save_references_to_cache(cache_path, [reference])

            # Generate final output
            final_cache_key = get_cache_key(str(question_id) + "_final", i_round)
            final_cache_path = get_final_output_cache_path(cache_dir, model, final_cache_key) if cache_dir else None


            # Modified: only try to load final output cache when all references are from cache (all_cached=True)
            # If references are recomputed, force regeneration of final output
            cached_output = None
            if all_cached and final_cache_path:
                cached_output = load_text_from_cache(final_cache_path)

            final_output_from_cache = False # Flag whether output is from cache, used for subsequent res output judgment

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
                        references=references,
                    ).strip()
                except:
                    try:
                        output = generate_with_references_final(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            references=[],
                        ).strip()
                    except:
                        output = None
                # Save to cache (only save non-None values)
                if final_cache_path and output is not None:
                    save_text_to_cache(final_cache_path, output)

            if i_round == 0:
                prev_references = [output]
                history.append(output)
                references = []
            else:  # i_round > 0
                # Restore history from previous rounds (regardless of whether current round is loaded from cache)
                # Try to load final output from all previous rounds into history
                if len(history) < i_round:
                    logger.info(f"Restoring history from cache for rounds 0 to {i_round-1}")
                    for prev_round in range(i_round):
                        if prev_round >= len(history):
                            prev_cache_key = get_cache_key(str(question_id) + "_final", prev_round)
                            if cache_dir:
                                prev_cache_path = get_final_output_cache_path(cache_dir, model, prev_cache_key)
                                prev_output = load_text_from_cache(prev_cache_path)
                                if prev_output is not None:
                                    history.append(prev_output)
                                else:
                                    logger.warning(f"Failed to restore history for round {prev_round}")
                                    break
                            else:
                                logger.warning(f"Cannot restore history for round {prev_round}: cache_dir is None")
                                break

                # Generate res output
                res_cache_key = get_cache_key(str(question_id) + "_res", i_round)
                res_cache_path = get_res_output_cache_path(cache_dir, model, res_cache_key) if cache_dir else None

                # Modified: only try to load res output cache when final output is from cache
                # If final output is new, res output must also be recomputed
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
                        cur_output=[output]
                    ).strip()

                    # Save to cache
                    if res_cache_path:
                        save_text_to_cache(res_cache_path, res_output)

                if res_output is not None and "AMoA should be stopped" in res_output:
                    print('Early Stopped')
                    break
                else:
                    output = res_output
                    prev_references = [output]
                    history.append(output)
                    references = []

    return {
        "text": output,
    }


def main(
    model: str,
    output_path: str,
    reference_paths: str = None,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    rounds: int = 1,
    num_proc: int = 16,
    provider: str = "together",
    cache_dir: str = "./cache",
):

    if reference_paths is None:
        reference_paths = []
    else:
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")

    eval_set = []
    # with open("FLASK/evaluation_set/flask_evaluation.jsonl") as f:
    with open("FLASK/evaluation_set/flask_hard_evaluation.jsonl") as f:
        for line in f:
            if line.strip() == "":
                continue
            item = json.loads(line)
            eval_set.append({"question_id": item["idx"], "text": item["instruction"]})

    eval_set = datasets.Dataset.from_list(eval_set)

    if len(reference_paths):

        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths:
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(
                    f"Reading reference outputs: {reference_path} ({len(reference_responses)})"
                )
                for i_reference_response, reference_response in enumerate(
                    reference_responses
                ):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(
                            reference_response["output"]
                        )

        eval_set = eval_set.add_column(f"references", references)

    elif len(reference_models):

        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )

    logger.info(f"Start.")

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=model,
            reference_models=reference_models,
            temperature=temperature,
            max_tokens=max_tokens,
            rounds=rounds,
            num_workers=num_proc,
            cache_dir=cache_dir,
        ),
        batched=False,
        num_proc=num_proc,
    )

    logger.info(f"Saving outputs to {output_path}.")

    eval_set.to_json(output_path)


if __name__ == "__main__":

    Fire(main)
