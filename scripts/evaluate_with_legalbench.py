#!/usr/bin/env python3
"""
使用LegalBench官方评估方法重新评估模型输出结果
"""

import json
import sys
import os
import argparse
import re
from typing import List

# 添加legalbench路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'legalbench'))

from evaluation import evaluate, normalize

def extract_final_answer(output_text: str) -> str:
    """
    从模型输出中提取最终答案
    
    Args:
        output_text: 模型的原始输出文本
        
    Returns:
        提取的答案，如果没找到则返回空字符串
    """
    # 首先尝试查找 \boxed{} 格式的答案
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, output_text, re.IGNORECASE)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # 如果没有boxed格式，尝试查找"Relevant"或"Irrelevant"
    # 注意：privacy_policy_qa的答案应该是"Relevant"或"Irrelevant"
    relevant_pattern = r'\b(Relevant|Irrelevant)\b'
    relevant_matches = re.findall(relevant_pattern, output_text, re.IGNORECASE)
    if relevant_matches:
        # 返回最后一个匹配的答案（通常是最终答案）
        return relevant_matches[-1].capitalize()
    
    # 如果还是没找到，尝试其他可能的格式
    answer_patterns = [
        r'(?:final answer|answer|conclusion).*?is\s*[:\-]?\s*(Relevant|Irrelevant)',
        r'(?:therefore|thus|so),?\s*(Relevant|Irrelevant)',
        r'^(Relevant|Irrelevant)$'  # 单行答案
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, output_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].capitalize()
    
    # 如果都没找到，返回空字符串
    return ""

def load_results(file_path: str) -> tuple:
    """
    加载模型输出结果文件
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        tuple: (generations, answers) - 模型生成结果和正确答案的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    generations = []
    answers = []
    
    for item in data:
        # 提取模型的原始输出
        raw_output = item.get('Output', '')
        
        # 从输出中提取最终答案
        extracted_answer = extract_final_answer(raw_output)
        generations.append(extracted_answer)
        
        # 获取正确答案
        correct_answer = item.get('Correct Choice', '')
        answers.append(correct_answer)
    
    return generations, answers

def print_detailed_analysis(generations: List[str], answers: List[str]):
    """
    打印详细的分析结果
    """
    print("\n=== 详细分析 ===")
    print(f"总样本数: {len(generations)}")
    print(f"有效提取答案数: {sum(1 for g in generations if g)}")
    print(f"空答案数: {sum(1 for g in generations if not g)}")
    
    # 统计答案分布
    answer_counts = {}
    for ans in answers:
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    
    generation_counts = {}
    for gen in generations:
        if gen:
            generation_counts[gen] = generation_counts.get(gen, 0) + 1
    
    print(f"\n正确答案分布: {answer_counts}")
    print(f"模型预测分布: {generation_counts}")
    
    # 逐个样本分析（只显示前5个）
    print("\n前5个样本的预测结果:")
    for i in range(min(5, len(generations))):
        correct = "✓" if generations[i] == answers[i] else "✗"
        print(f"  样本 {i+1}: 预测='{generations[i]}' | 正确='{answers[i]}' | {correct}")

def main():
    parser = argparse.ArgumentParser(description="使用LegalBench官方方法评估privacy_policy_qa结果")
    parser.add_argument('--input_file', type=str, required=True, 
                       help='模型输出结果文件路径')
    parser.add_argument('--task', type=str, default='privacy_policy_qa',
                       help='任务名称 (默认: privacy_policy_qa)')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细分析')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 文件 {args.input_file} 不存在")
        return
    
    print(f"正在评估任务: {args.task}")
    print(f"输入文件: {args.input_file}")
    
    # 加载结果
    try:
        generations, answers = load_results(args.input_file)
    except Exception as e:
        print(f"错误: 无法加载结果文件: {e}")
        return
    
    # 使用LegalBench官方评估方法
    try:
        score = evaluate(args.task, generations, answers)
        print(f"\n=== LegalBench官方评估结果 ===")
        print(f"任务: {args.task}")
        print(f"评估指标: 平衡准确率 (Balanced Accuracy)")
        print(f"分数: {score:.4f}")
        print(f"百分比: {score * 100:.2f}%")
        
        # 显示详细分析
        if args.detailed:
            print_detailed_analysis(generations, answers)
            
    except Exception as e:
        print(f"错误: 评估过程中出现问题: {e}")
        return

if __name__ == "__main__":
    main() 