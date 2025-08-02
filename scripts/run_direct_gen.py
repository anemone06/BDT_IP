# run_direct_gen.py
import csv
import json
import random
import torch
import re
import os, time
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluate import run_evaluation
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)
import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")
    
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True, 
        # 添加了新的数据集选项
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth', 'legal_citation', 'International', 'consumer_contracts_qa', 'abercrombie', 'function_of_decision_section', 'proa'],
        help="Name of the dataset to use."
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        required=True, 
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )
    
    parser.add_argument(
        '--subset_num', 
        type=int, 
        default=-1, 
        help="Number of examples to process. Defaults to all if not specified."
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Path to the pre-trained model."
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.7, 
        help="Sampling temperature."
    )
    
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.8, 
        help="Top-p sampling parameter."
    )
    
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=20, 
        help="Top-k sampling parameter."
    )
    
    parser.add_argument(
        '--repetition_penalty', 
        type=float, 
        default=None, 
        help="Repetition penalty. If not set, defaults based on the model."
    )
    
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        # 默认值与 run_search_o1.py 保持一致
        default=32768, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # 添加了 start_index 和 end_index 参数，与 run_search_o1.py 保持一致
    parser.add_argument(
        '--start_index',
        type=int,
        default=None,
        help="Start index for processing subset of data."
    )
    
    parser.add_argument(
        '--end_index',
        type=int,
        default=None,
        help="End index for processing subset of data."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 与 run_search_o1.py 一样设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    
    # 更新了 repetition_penalty 的默认值逻辑，与 run_search_o1.py 保持一致
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0
    
    # 更新了数据路径逻辑，以支持新数据集
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    elif dataset_name.startswith('legalbench') or dataset_name in ['legal_citation', 'International', 'consumer_contracts_qa', 'abercrombie', 'function_of_decision_section', 'proa']:
        data_path = f'./data/{dataset_name}_{split}.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'
    
    logger.info('-----------------------')
    logger.info(f'Using {dataset_name} {split} set from {data_path}.')
    logger.info('-----------------------')

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # 更新了 model_short_name 和 output_dir 的逻辑，与 run_search_o1.py 保持一致
    if 'qwq' in model_path.lower():
        model_short_name = 'qwq'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')

    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode'] or dataset_name.startswith('legal'):
            output_dir = f'./outputs/{dataset_name}.qwq.direct'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.direct'
    else:
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct'
    os.makedirs(output_dir, exist_ok=True)
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )
    
    # 加载数据
    with open(data_path, mode='r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
    
    # 更新了数据切片逻辑，优先使用 start/end index
    if args.start_index is not None and args.end_index is not None:
        filtered_data = filtered_data[args.start_index:args.end_index]
    elif subset_num != -1:
        filtered_data = filtered_data[:subset_num]

    # 准备输入
    input_list = []
    for item in filtered_data:
        question = item['Question']
        # 更新了 prompt 选择逻辑，以支持新数据集
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name in ['gpqa', 'medmcqa', 'pubhealth', 'International', 'consumer_contracts_qa', 'abercrombie', 'function_of_decision_section', 'proa']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
            
        elif dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        
        elif dataset_name == 'legal_citation':
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        else:
            # 为未匹配的数据集提供默认值
            logger.warning(f"No specific prompt instruction for dataset {dataset_name}. Using a generic prompt.")
            user_prompt = question
            
        prompt = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)
    
    # 更新了 max_tokens 的默认值设置逻辑，与 run_search_o1.py 保持一致
    if 'qwq' in model_path.lower():
        if dataset_name in ['aime', 'amc', 'livecode']:
            max_tokens = 32768
        else:
            max_tokens = 20480
    else:
        max_tokens = 8192
    
    t_start = time.time()
    # 生成模型输出
    output_list = llm.generate(
        input_list, 
        sampling_params=SamplingParams(
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            repetition_penalty=repetition_penalty,
        )
    )
    total_time = time.time() - t_start
    
    # 运行评估
    run_evaluation(
        filtered_data, 
        input_list, 
        output_list, 
        dataset_name, 
        output_dir, 
        total_time, 
        split,
    )
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()