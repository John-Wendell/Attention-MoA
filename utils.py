import os
import json
import time
import requests
import openai
import copy
import gc
import torch
import re
import random

from loguru import logger
# from json_repair import repair_json

DEBUG = int(os.environ.get("DEBUG", "0"))  # Debug mode flag


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    top_p=1.0,
    top_k=100,
    api_key_list=None,
    timeout=2000,
    retry=True,
):

    if api_key_list is None:

        api_key_list = ['Your_API_KEY']


    base_url = 'Your_Base_Url'

    # Construct Friday API request parameters
    base_app_id = random.choice(api_key_list)
    headers = {
        'Authorization': f'Bearer {base_app_id}',
        'Content-Type': 'application/json'
    }

    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "thinking": {"type":"disabled"}
    }

    # Retry logic
    max_retries = 5 if retry else 1
    interval = 10

    for attempt in range(max_retries):
        try:
            response = requests.post(url=base_url, json=payload, headers=headers, timeout=timeout)
            status = response.status_code

            if status == 200:
                try:
                    parsed = response.json()
                    message = parsed["choices"][0]["message"]
                    output = message.get('content', '').strip()
                    if DEBUG:
                        logger.debug(f"Output: `{output[:20]}...`")
                    return output
                except Exception as e:
                    logger.error(f"Failed to parse response: {e}")
                    # Parse failure is considered failure, continue retrying

            # Handle non-200 status
            status_message = response.text
            try:
                status_message = response.json().get("message", status_message)
            except:
                pass
        except Exception as e:
            logger.error(f"Request exception: {e}")

        # If not yet successful and still have retry attempts
        if attempt < max_retries - 1:
            logger.info(f"Request failed, will retry {max_retries - 1 - attempt} more times!")
            time.sleep(interval)
            interval += 5
        else:
            return None

    return None


def inject_references_to_messages(
    messages,
    references,
    model=None,
):

    messages = copy.deepcopy(messages)
    # print('messages before injection: ', messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    # For Gemma models, inject the prompt into the last user message to avoid context misalignment in multi-turn conversations
    if model and ('gemma' in model.lower()):
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                messages[i]['content'] = system + "\n\n" + messages[i]['content']
                break
    elif messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages
    # print('messages after injection: ', messages)
    return messages

def inject_references_to_messages_final(
    messages,
    references,
    model=None,
):
    messages = copy.deepcopy(messages)
    if len(references) > 0:
        # print('messages before injection: ', messages)

        system = f"""You have been provided with a set of responses from various large language models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

        for i, reference in enumerate(references):
            system += f"\nResponse from model {i+1}:\n {reference}\n"

        if model and ('gemma' in model.lower()):
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]['role'] == 'user':
                    messages[i]['content'] = system + "\n\n" + messages[i]['content']
                    break
        elif messages[0]["role"] == "system":

            messages[0]["content"] += "\n\n" + system

        else:

            messages = [{"role": "system", "content": system}] + messages
        # print('messages after injection: ', messages)
    return messages

def inject_references_to_messages_QKV(
    messages,
    references,
    model=None,
):

    messages = copy.deepcopy(messages)
    with open('attention_prompt/qkv_prompt.md', 'r') as f:
        qkv_prompt = f.read()

    if len(references) > 0:
        system = qkv_prompt
        system += f"""\n\nYou have been provided with a set of responses from various large language models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability."""
        system += '\n\n' + """Responses from models:\n"""
        for i, reference in enumerate(references):
            system += f"\nResponse from model {i+1}:\n {reference}\n"
    else:
        system = qkv_prompt
    
    if model and ('gemma' in model.lower()):
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                messages[i]['content'] = system + "\n\n" + messages[i]['content']
                break
    else:
        messages = [{"role": "system", "content": system}] + messages
    # print('messages after QKV injection: ', messages)
    return messages

def inject_references_to_messages_A(
    messages,
    q,
    # v,
    k
):

    messages = copy.deepcopy(messages)
    # with open('attention_prompt/a_prompt_sys.md', 'r') as f:
    #     a_prompt_sys = f.read()
    with open('attention_prompt/a_prompt_user.md', 'r') as f:
        a_prompt_user = f.read()
    # system = a_prompt.format(model_query=q)
    # if messages[0]["role"] == "system":
    #     messages[0]["content"] += "\n\n" + system
    # else:
    #     messages = [{"role": "system", "content": system}] + messages
    
    # prompt = a_prompt.format(user_query=messages[-1]['content'],model_query=q)
    # messages[-1]['content'] = prompt
    # print('messages: \n', messages)
    if len(messages)>1:
        conv_history = ''
        for i in range(len(messages)-1):
            role = messages[i]["role"]
            content = messages[i]["content"]
            conv_history += f'{role}: {content}\n'
    else:
        conv_history = 'None'
    # print('conv_history:\n',conv_history)
    prompt = a_prompt_user.format(user_query=messages[-1]['content'], 
                                  conv_history=conv_history,
                                  model_answer_other=q, 
                                  model_answer_own=k)
    # messages = [{"role": "system", "content": a_prompt_sys}, {"role": "user", "content": prompt}]
    messages = [{"role": "user", "content": prompt}]
    return messages

def inject_references_to_messages_A_self(
    messages,
    q,
    # v,
    # k
):

    messages = copy.deepcopy(messages)
    # with open('attention_prompt/a_self_prompt_sys.md', 'r') as f:
    #     a_prompt_sys = f.read()
    with open('attention_prompt/a_self_prompt_user.md', 'r') as f:
        a_prompt_user = f.read()   
    # system = a_prompt.format(model_query=q)
    # if messages[0]["role"] == "system":
    #     messages[0]["content"] += "\n\n" + system
    # else:
    #     messages = [{"role": "system", "content": system}] + messages
    
    # prompt = a_prompt.format(user_query=messages[-1]['content'],model_query=q)
    # messages[-1]['content'] = prompt
    if len(messages)>1:
        conv_history = ''
        for i in range(len(messages)-1):
            role = messages[i]["role"]
            content = messages[i]["content"]
            conv_history += f'{role}: {content}\n'
    else:
        conv_history = 'None'
        
    prompt = a_prompt_user.format(user_query=messages[-1]['content'], 
                                  conv_history=conv_history,
                                  model_answer_own=q)
    # messages = [{"role": "system", "content": a_prompt_sys}, {"role": "user", "content": prompt}]
    messages = [{"role": "user", "content": prompt}]
    
    return messages
def inject_references_to_messages_AKV(
    messages: list[dict],
    ats4reference_model: list,
    q: str,
    # k: str,
    # v: str,
    model=None,
):
    messages = copy.deepcopy(messages)
    with open('attention_prompt/akv_prompt_sys.md', 'r') as f:
        akv_prompt_sys = f.read()
    # with open('attention_prompt/akv_prompt_user.md', 'r') as f:
    #     akv_prompt_user = f.read()
        
    model_attention = ""
    for i in range(len(ats4reference_model)):
        model_attention += f"\nSuggestions from model {i+1}:\n {ats4reference_model[i]}\n"
        
    # system = akv_prompt.format(model_query=q, model_cot=k, model_answer=v, model_attention=model_attention)
    akv_prompt_sys = akv_prompt_sys.format(model_query=q, 
                                           model_attention=model_attention)

    if model and ('gemma' in model.lower()):
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                messages[i]['content'] = akv_prompt_sys + "\n\n" + messages[i]['content']
                break
    elif messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + akv_prompt_sys
    else:
        messages = [{"role": "system", "content": akv_prompt_sys}] + messages
        
    return messages

def inject_references_to_messages_res(
    messages: list[dict],
    pre_output: list,
    cur_output: list,
    # k: str,
    # v: str,
    model=None,
    is_es=True
):
    messages = copy.deepcopy(messages)
    if is_es==True:
        with open('attention_prompt/res_prompt_sys_early_stop.md', 'r') as f:
            sys_prompt = f.read()
    else:
        with open('attention_prompt/res_prompt_sys_ful.md', 'r') as f:
            sys_prompt = f.read()
        
    # pre_output_sum = 'Responses of previous round:\n'
    # for i, reference in enumerate(pre_output):
    #     pre_output_sum += f"\nResponse of history round {i+1}:\n {reference}\n"
        
    # cur_output_sum = ''
    # for i, reference in enumerate(cur_output):
    #     cur_output_sum += f"\nThe response of the latest round:\n {reference}\n"

    # sys_prompt = sys_prompt.format(pre_output=pre_output_sum,
    #                             cur_output=cur_output_sum)
    
    pre_output.extend(cur_output)
    pre_output_sum = 'Responses of historical round:\n'
    for i, reference in enumerate(pre_output):
        pre_output_sum += f"\nResponse of historical round {i+1}:\n {reference}\n"
    sys_prompt = sys_prompt.format(pre_output=pre_output_sum)

    if model and 'gemma' in model.lower():
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                messages[i]['content'] = sys_prompt + "\n\n" + messages[i]['content']
                break
    elif messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + sys_prompt
    else:
        messages = [{"role": "system", "content": sys_prompt}] + messages
    return messages

def generate_with_references(
    model,
    messages,
    references=None,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_openai,
):
    if references is None:
        references = []

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references, model=model)
    # print('messages of aggregator: \n', messages)
    
    result = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if 'gpt-oss' in model.lower():
        result = re.sub(r'analysis.*?assistantfinal', '', result, flags=re.DOTALL)
    if 'qwen3' in model.lower():
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return result
    
def generate_with_references_final(
    model,
    messages,
    references=None,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_openai,
):
    if references is None:
        references = []

    if len(references) > 0:

        messages = inject_references_to_messages_final(messages, references, model=model)
    # print('\n messages of final: \n', messages)
    result = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if 'gpt-oss' in model.lower():
        result = re.sub(r'analysis.*?assistantfinal', '', result, flags=re.DOTALL)
    if 'qwen3' in model.lower():
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return result
    
def generate_with_references_QKV(
    model,
    messages,
    references=None,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_openai,
):
    if references is None:
        references = []

    messages = inject_references_to_messages_QKV(messages, references, model=model)
    # print('\n messages of QKV: \n', messages)
    result = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if 'gpt-oss' in model.lower():
        result = re.sub(r'analysis.*?assistantfinal', '', result, flags=re.DOTALL)
    if 'qwen3' in model.lower():
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return result
    

def generate_with_references_A(
    model,
    messages,
    q,
    k,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_openai,
):


    messages = inject_references_to_messages_A(messages, q,k)
    # print('\n messages of A: \n', messages)
    result = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if 'gpt-oss' in model.lower():
        result = re.sub(r'analysis.*?assistantfinal', '', result, flags=re.DOTALL)
    if 'qwen3' in model.lower():
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return result
    
    
def generate_with_references_A_self(
    model,
    messages,
    q,
    # k,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_openai,
):


    messages = inject_references_to_messages_A_self(messages, q)
    # print('\n messages of A: \n', messages)
    result = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if 'gpt-oss' in model.lower():
        result = re.sub(r'analysis.*?assistantfinal', '', result, flags=re.DOTALL)
    if 'qwen3' in model.lower():
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return result
    
    
def generate_with_references_AKV(
    model,
    messages,
    ats4reference_model,
    q,
    # k,
    # v,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_openai,
):


    messages = inject_references_to_messages_AKV(messages, ats4reference_model, q, model=model)
    # print('\n messages of AKV: \n', messages)
    result = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if 'gpt-oss' in model.lower():
        result = re.sub(r'analysis.*?assistantfinal', '', result, flags=re.DOTALL)
    if 'qwen3' in model.lower():
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return result
    
def generate_with_res(
    model,
    messages,
    pre_output,
    cur_output,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_openai,
):
    messages = inject_references_to_messages_res(messages, pre_output, cur_output, model=model,is_es=False)
    result = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if 'gpt-oss' in model.lower():
        result = re.sub(r'analysis.*?assistantfinal', '', result, flags=re.DOTALL)
    if 'qwen3' in model.lower():
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return result
    

# def get_QKV(reference_output):
    
#     try:
#         reference_json = json.loads(repair_json(reference_output))
#     except:
#         # print('reference_output: ', reference_output)
#         reference_json = None

#     # if isinstance(reference_json, dict):
#     #     q = str(reference_json.get('Query_of_CoT', ''))
#     #     k = str(reference_json.get('Chain_of_Thought', ''))
#     #     v = str(reference_json.get('Answer', ''))
#     # else:
#     #     q = None
#     #     k = None
#     #     v = None
#     if isinstance(reference_json, dict):
#         q = str(reference_json.get('Query_of_Answer', ''))
#         v = str(reference_json.get('Answer', ''))
#     else:
#         q = None
#         v = None
#     # print('q: \n', q)
#     # print('k: ', k)
#     # print('v: \n', v)
#     return q, v


# VLLM



def generate_with_references_final_vllm(
    model,
    messages_list,  # list[list[dict]], outer layer is dataset, inner layer is conversation
    references_list=None,  # list[list[str]], references corresponding to each sample
    temperature=0.7,
    generate_fn=None,  # vllm batch generation function
):
    """
    Batch process dataset function
    
    Args:
        model: model name
        messages_list: list[list[dict]], outer list is dataset size, inner list is conversation history of each sample
        references_list: list[list[str]], list of reference answers corresponding to each sample, if None then no references are injected
        max_tokens: maximum number of tokens to generate
        temperature: temperature parameter
        generate_fn: vllm batch generation function, needs to support batch input
        
    Returns:
        list[str]: generation results for each sample
    """
    
    if generate_fn is None:
        logger.error("Need to provide vllm batch generation function")
        return None
    
    # Process messages for each sample, inject references
    processed_messages_list = []
    
    for idx, messages in enumerate(messages_list):
        # Get references for current sample
        if references_list is not None and idx < len(references_list):
            references = references_list[idx]
            if len(references) > 0:
                # Inject references
                processed_messages = inject_references_to_messages_final(messages, references, model=model)
            else:
                processed_messages = copy.deepcopy(messages)
        else:
            processed_messages = copy.deepcopy(messages)
        
        processed_messages_list.append(processed_messages)
    

    # Batch call vllm for generation
    try:
        print(len(processed_messages_list))
        outputs = generate_fn(
            model=model,
            messages_list=processed_messages_list,
            temperature=temperature,
        )
        return outputs
    except Exception as e:
        logger.error(f"vllm batch generation failed: {e}")
        return None
    
def generate_with_references_AKV_vllm(
    model,
    messages_list,  # list[list[dict]]
    ats4reference_model_list,  # list[list[str]], attention suggestions for each sample
    q_list,  # list[str], query for each sample
    temperature=0.7,
    generate_fn=None,
):
    """
    Batch process AKV injection function
    
    Args:
        model: model name
        messages_list: list[list[dict]], batch conversation history
        ats4reference_model_list: list[list[str]], attention suggestions for each sample
        q_list: list[str], query for each sample
        max_tokens: maximum number of tokens to generate
        temperature: temperature parameter
        generate_fn: vllm batch generation function
        
    Returns:
        list[str]: generation results for each sample
    """
    
    if generate_fn is None:
        logger.error("Need to provide vllm batch generation function")
        return None
    
    # Process messages for each sample, inject AKV
    processed_messages_list = []
    
    for idx, messages in enumerate(messages_list):
        ats4reference_model = ats4reference_model_list[idx] if idx < len(ats4reference_model_list) else []
        q = q_list[idx] if idx < len(q_list) else ""
        
        processed_messages = inject_references_to_messages_AKV(
            messages, 
            ats4reference_model, 
            q, 
            model=model
        )
        processed_messages_list.append(processed_messages)
    
    # Add no-cot system prompt

    # Batch call vllm for generation
    try:
        outputs = generate_fn(
            model=model,
            messages_list=processed_messages_list,
            temperature=temperature,
        )
        return outputs
    except Exception as e:
        logger.error(f"vllm batch generation failed: {e}")
        return None


def generate_with_references_A_self_vllm(
    model,
    messages_list,  # list[list[dict]]
    q_list,  # list[str], query for each sample
    temperature=0.7,
    generate_fn=None,
):
    """
    Batch process A_self injection function
    
    Args:
        model: model name
        messages_list: list[list[dict]], batch conversation history
        q_list: list[str], query for each sample
        max_tokens: maximum number of tokens to generate
        temperature: temperature parameter
        generate_fn: vllm batch generation function
        
    Returns:
        list[str]: generation results for each sample
    """
    
    if generate_fn is None:
        logger.error("Need to provide vllm batch generation function")
        return None
    
    # Process messages for each sample, inject A_self
    processed_messages_list = []
    # print('messages_list\n', messages_list)
    # print('q_list\n', q_list)
    for idx, messages in enumerate(messages_list):
        q = q_list[idx] if idx < len(q_list) else ""
        processed_messages = inject_references_to_messages_A_self(messages, q)
        processed_messages_list.append(processed_messages)
    


    # Batch call vllm for generation
    try:
        outputs = generate_fn(
            model=model,
            messages_list=processed_messages_list,
            temperature=temperature,
        )
        return outputs
    except Exception as e:
        logger.error(f"vllm batch generation failed: {e}")
        return None


def generate_with_references_A_vllm(
    model,
    messages_list,  # list[list[dict]]
    q_list,  # list[str], query for each sample (model_answer_other)
    k_list,  # list[str], key for each sample (model_answer_own)
    temperature=0.7,
    generate_fn=None,
):
    """
    Batch process A injection function
    
    Args:
        model: model name
        messages_list: list[list[dict]], batch conversation history
        q_list: list[str], query for each sample (model_answer_other)
        k_list: list[str], key for each sample (model_answer_own)
        max_tokens: maximum number of tokens to generate
        temperature: temperature parameter
        generate_fn: vllm batch generation function
        
    Returns:
        list[str]: generation results for each sample
    """
    
    if generate_fn is None:
        logger.error("Need to provide vllm batch generation function")
        return None
    
    # Process messages for each sample, inject A
    processed_messages_list = []
    
    for idx, messages in enumerate(messages_list):
        q = q_list[idx] if idx < len(q_list) else ""
        k = k_list[idx] if idx < len(k_list) else ""
        
        processed_messages = inject_references_to_messages_A(messages, q, k)
        processed_messages_list.append(processed_messages)


    # Batch call vllm for generation
    try:
        outputs = generate_fn(
            model=model,
            messages_list=processed_messages_list,
            temperature=temperature,
        )
        return outputs
    except Exception as e:
        logger.error(f"vllm batch generation failed: {e}")
        return None


def generate_with_res_vllm(
    model,
    messages_list,  # list[list[dict]]
    pre_output_list,  # list[list[str]], historical output of each sample
    cur_output_list,  # list[list[str]], current output of each sample
    temperature=0.7,
    generate_fn=None,
):
    """
    Batch process residual injection function
    
    Args:
        model: model name
        messages_list: list[list[dict]], batch conversation history
        pre_output_list: list[list[str]], historical output of each sample
        cur_output_list: list[list[str]], current output of each sample
        max_tokens: maximum number of tokens to generate
        temperature: temperature parameter
        generate_fn: vllm batch generation function
        
    Returns:
        list[str]: generation results for each sample
    """
    
    if generate_fn is None:
        logger.error("Need to provide vllm batch generation function")
        return None
    
    # Process messages for each sample, inject residual
    processed_messages_list = []
    
    for idx, messages in enumerate(messages_list):
        pre_output = pre_output_list[idx] if idx < len(pre_output_list) else []
        cur_output = cur_output_list[idx] if idx < len(cur_output_list) else []
        
        processed_messages = inject_references_to_messages_res(
            messages, 
            pre_output, 
            cur_output, 
            model=model,
            is_es=False
        )
        processed_messages_list.append(processed_messages)
    

    # Batch call vllm for generation
    try:
        outputs = generate_fn(
            model=model,
            messages_list=processed_messages_list,
            temperature=temperature,
        )
        return outputs
    except Exception as e:
        logger.error(f"vllm batch generation failed: {e}")
        return None

def generate_mixed_vllm_batch(
    model,
    task_list,  # list[dict] containing messages and type
    max_tokens=2048,
    max_model_len=None,
    temperature=0.7,
    top_p=1.0,
    top_k=100,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=False,
    dtype="auto",
    enforce_eager=True,
    generate_fn=None, # receives parameters passed from vllm_worker, but does not directly use it as inference function
    stage_name=None,
    save_dir="./batch_inference_results",
):
    """
    Mixed task batch inference supporting different types of prompt injection
    """
    # Core fix: prioritize using the passed-in generate_fn, if not provided fall back to generate_vllm_batch
    if generate_fn is None:
        inference_fn = generate_vllm_batch
    else:
        inference_fn = generate_fn

    # Ensure temperature length matches task_list length
    if isinstance(temperature, list):
        if len(temperature) == 0:
            temperature = 0.7
        elif len(temperature) < len(task_list):
            # Cyclically fill in
            num_repeats = (len(task_list) + len(temperature) - 1) // len(temperature)
            temperature = (temperature * num_repeats)[:len(task_list)]
        elif len(temperature) > len(task_list):
            temperature = temperature[:len(task_list)]

    processed_messages_list = []
    task_types = []
    for task in task_list:
        task_type = task.get("type")
        messages = task.get("messages")
        kwargs = task.get("kwargs", {})

        if task_type == "final":
            processed = inject_references_to_messages_final(messages, kwargs.get("references", []), model=model)
        elif task_type == "A":
            processed = inject_references_to_messages_A(messages, kwargs.get("q"), kwargs.get("k"))
        elif task_type == "A_self":
            processed = inject_references_to_messages_A_self(messages, kwargs.get("q"))
        elif task_type == "AKV":
            processed = inject_references_to_messages_AKV(messages, kwargs.get("ats", []), kwargs.get("q"), model=model)
        elif task_type == "res":
            processed = inject_references_to_messages_res(messages, kwargs.get("pre", []), kwargs.get("cur", []), model=model, is_es=False)
        else:
            processed = copy.deepcopy(messages)
        processed_messages_list.append(processed)
        task_types.append(task_type)

    # If stage_name is not specified, use the task_type of the first task
    if stage_name is None and task_types:
        stage_name = task_types[0]

    return inference_fn(
        model=model,
        messages_list=processed_messages_list,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        enforce_eager=enforce_eager,
        stage_name=stage_name,
        save_dir=save_dir,
    )


def generate_vllm_batch(
    model,
    messages_list,
    max_tokens=2048,
    max_model_len=None,  # changed to None as default value
    temperature=0.7,
    top_p=1.0,
    top_k=100,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=False,
    dtype="auto",
    enforce_eager=True,
    stage_name=None,
    save_dir="./batch_inference_results",
):
    """
    Function for batch inference using vllm
    
    Args:
        model: model name or path
        messages_list: list[list[dict]], batch conversation history
        max_tokens: maximum number of tokens to generate
        max_model_len: vllm model maximum sequence length, if None then use model default value
        temperature: temperature parameter
        top_p: top_p sampling parameter
        top_k: top_k sampling parameter
        tensor_parallel_size: tensor parallel size for multi-card inference
        gpu_memory_utilization: GPU memory utilization rate (0-1)
        trust_remote_code: whether to trust remote code
        dtype: model data type
        enforce_eager: whether to force eager mode (disable CUDA Graph)
        stage_name: stage name for saving file path
        save_dir: root directory for saving results
        
    Returns:
        list[str]: generation results for each sample
    """
    
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("Need to install vllm: pip install vllm")
        return None
    
    # Extract model name (used for file path)
    model_name = model.split('/')[-1] if '/' in model else model
    model_full_path = "/model_path/" + model

    llm_kwargs = {
        "model": model_full_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
        "enforce_eager": enforce_eager
    }


    
    # Only add to parameters when max_model_len is not None
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    
    llm = LLM(**llm_kwargs)
    
    # Set sampling parameters
    if isinstance(temperature, list):
        if len(temperature) != len(messages_list):
            logger.error(f"Temperature list length ({len(temperature)}) does not match messages list length ({len(messages_list)})")
            return None
    
        sampling_params = []
        for t in temperature:
            sampling_params.append(SamplingParams(
                temperature=t,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            ))
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

    # Batch inference
    outputs = llm.chat(
        messages=messages_list,
        sampling_params=sampling_params,
        use_tqdm=True,  # show progress bar
    )
    
    # Extract generated text
    results = []
    for output in outputs:
        if output.outputs:
            text = output.outputs[0].text.strip()
            if 'gpt-oss' in model_full_path.lower():
                text = re.sub(r'analysis.*?assistantfinal', '', text, flags=re.DOTALL)
            if 'qwen3' in model_full_path.lower():
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            results.append(text)
        else:
            results.append("")
            logger.warning("Generation result for one sample is empty")
            
    # Release llm object and GPU memory
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Released vllm model and GPU memory")
    
    return results

