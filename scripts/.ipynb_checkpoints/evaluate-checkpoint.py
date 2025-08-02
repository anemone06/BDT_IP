import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict
from lcb_runner.evaluation import codegen_metrics
from utils.math_equivalence import is_equiv
from sklearn.metrics import balanced_accuracy_score



def normalize_for_legal(text: str) -> str:
    """
    Normalizes strings for legal citation evaluation, based on legalbench.
    - Removes punctuation
    - Removes extra spaces
    - Converts to lower case
    """
    # Remove punctuation
    text = str(text).translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = text.strip()
    # Make lower case
    text = text.lower()
    return text


def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "**Final Information**"
        pattern_step = "**Modified Reasoning Steps**"
        clean_output = output.strip() 
        if clean_output.startswith(pattern_info): # 使用 startswith 更精确
            extracted_text = clean_output.split(pattern_info, 1)[-1].replace("\n","").strip("```").strip()
        elif clean_output.startswith(pattern_step):
            extracted_text = clean_output.split(pattern_step, 1)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    else:
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
    return final_metric, pred_answer



def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False):
    all_ground_truths = []
    all_predictions = []
    
    if dataset_name == 'livecode':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []

        # Collect difficulty levels for per-domain metrics
        difficulties = []
        per_difficulty_count = {}
        num_valid_answer = 0


        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            difficulty = item.get("difficulty", "Unknown")
            difficulties.append(difficulty)
            # Track metrics per domain
            if difficulty not in per_difficulty_count.keys():
                per_difficulty_count[difficulty] = 0

            pred_code = extract_answer(item['Output'], mode='codegen')
            if pred_code != '':
                num_valid_answer += 1
                per_difficulty_count[difficulty] += 1
            # Assuming each item has 'input_output' with 'inputs' and 'outputs'
            public_test_cases = json.loads(item.get("public_test_cases", "{}"))

            inputs, outputs = [], []
            for case in public_test_cases:
                inputs.append(case["input"])
                outputs.append(case["output"])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append([pred_code])
            item['Pred_Answer'] = pred_code
            item['Question'] = input_prompt


        # Call codegen_metrics with pass@1
        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[1],  # Evaluate the top 1 generated result
            num_process_evaluate=2,   # Parallel evaluation
            timeout=10,  # Set timeout to 10 seconds
            debug=False,  # Enable debug mode
        )
        # print('samples_list', samples_list)
        # print('generations_list', generations_list)
        # print('metrics', metrics)

        # Extract pass@1
        pass_at_1 = metrics.get('pass@1', 0.0)
        detail_pass_at_1 = metrics['detail']['pass@1']

        for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
            item['Metrics'] = {'pass@1': pass1}
            item['Results'] = res
            item['Final_metadata'] = meta

        # Initialize per-difficulty metrics
        difficulty_metrics = defaultdict(list)
        for idx, difficulty in enumerate(difficulties):
            pass1 = detail_pass_at_1[idx]
            difficulty_metrics[difficulty].append(pass1)

        # Compute overall pass@1
        overall_metrics = {
            'pass@1': pass_at_1,  # / num_valid_answer * len(input_list),
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }

        # Compute per-difficulty pass@1
        per_difficulty_metrics = {}
        for difficulty, passes in difficulty_metrics.items():
            avg_pass = np.mean(passes) if len(passes) > 0 else 0.0
            num_valid_answer = per_difficulty_count[difficulty]
            per_difficulty_metrics[difficulty] = {
                'pass@1': avg_pass,
                'num_valid_answer': f'{num_valid_answer} of {len(passes)}'
            }

        # Save the metrics
        final_metrics = {
            'overall': overall_metrics,
            'per_domain': per_difficulty_metrics
        }

    else:
        # Existing evaluation for other datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0

        # If the dataset is GPQA, track metrics per domain
        domain_metrics = {}

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text

            # --- START OF RECOMMENDED MODIFICATION ---
            
            if dataset_name == 'legal_citation':
                # 1. 获取并清理标准答案
                labeled_answer = item.get("Answer", item.get("answer", "")).replace("Citation:", "").strip()
                
                # 2. 获取模型生成
                generation = item['Output']
            
                # 3. 规范化
                normalized_answer = normalize_for_legal(labeled_answer)
                normalized_generation = normalize_for_legal(generation)
            
                # 4. 评估核心逻辑 (来自 legalbench)
                citation_acc = 1.0 if normalized_answer in normalized_generation and normalized_answer != "" else 0.0
                
                # 5. 组装 metric 字典
                metric = {"is_valid_answer": True, "acc": citation_acc, "em": citation_acc, "f1": citation_acc, 'math_equal': 0}
                pred_answer = generation # 记录完整的生成作为预测答案
        
            else:
                # 原始的评估逻辑
                if dataset_name in ['gpqa', 'medmcqa', 'International', 'consumer_contracts_qa', 'abercrombie', 'function_of_decision_section', 'proa']:
                    labeled_answer = item["Correct Choice"]
                    mode = 'choose'
                elif dataset_name in ['math500', 'aime', 'amc']:
                    labeled_answer = item["answer"]
                    mode = 'gen'
                elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                    labeled_answer = item["answer"]
                    mode = 'qa'
                elif dataset_name in ['pubhealth']:
                    labeled_answer = item["answer"]
                    mode = 'choose'
                elif dataset_name.startswith('legalbench'):
                    labeled_answer = item.get("answer", item.get("Answer", item.get("Correct Choice")))
                    mode = 'choose'
                else:
                    raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
                metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer, mode=mode)
        
            # 2. 在循环中，收集规范化后的标签和预测
            # 我们只对需要计算平衡准确率的数据集执行此操作
            # legalbench 的许多分类任务（包括 international）都使用此指标
            if dataset_name in ('International', 'consumer_contracts_qa', 'abercrombie', 'function_of_decision_section', 'proa') or dataset_name.startswith('legalbench'):
                # 使用 legalbench 的规范化方法来处理标签和预测
                normalized_label = normalize_for_legal(str(labeled_answer))
                normalized_pred = normalize_for_legal(str(pred_answer))

                # 确保添加的不是空字符串
                if normalized_label and normalized_pred:
                    all_ground_truths.append(normalized_label)
                    all_predictions.append(normalized_pred)


            item['Pred_Answer'] = pred_answer
            item['Metrics'] = metric
            item['Question'] = input_prompt

            # 这部分代码保持不变，它会使用上面生成的 metric 字典
            my_method_valid = pred_answer != '' # 可以根据需要调整有效性判断
            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

            # If the dataset is GPQA, update domain-specific metrics
            if dataset_name == 'gpqa':
                domain = item.get("High-level domain", "Unknown")
                if domain not in domain_metrics:
                    domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['em'].append(metric['em'])
                domain_metrics[domain]['acc'].append(metric['acc'])
                domain_metrics[domain]['f1'].append(metric['f1'])
                domain_metrics[domain]['math_equal'].append(metric['math_equal'])
                if my_method_valid:
                    domain_metrics[domain]['num_valid_answer'] += 1

        t = time.localtime()
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

        # Compute overall metrics
        overall_results = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }

        # 3. 在循环外，计算并添加平衡准确率到最终结果中
        if all_ground_truths and all_predictions:
            try:
                # 调用 sklearn 函数计算平衡准确率
                balanced_acc_score = balanced_accuracy_score(all_ground_truths, all_predictions)
                overall_results['balanced_accuracy'] = balanced_acc_score
                print(f"Balanced Accuracy calculated successfully: {balanced_acc_score:.4f}")
            except Exception as e:
                # 如果计算出错（例如，预测中只出现一个类别），则记录错误
                overall_results['balanced_accuracy'] = "Error during calculation"
                print(f"Could not calculate balanced accuracy. Labels: {set(all_ground_truths)}, Preds: {set(all_predictions)}. Error: {e}")


        # If the dataset is GPQA, output average metrics per domain
        domain_avg_metrics = {}
        if dataset_name == 'gpqa':
            for dm, m in domain_metrics.items():
                domain_avg_metrics[dm] = {
                    'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                    'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                    'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                    'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                    'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
                }

        # 保存总体和分domain的指标
        final_metrics = {'overall': overall_results}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')

    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    import argparse

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Evaluate model outputs with optional backoff.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--output_metrics_path', type=str, help='Path to save the evaluation metrics.')
    parser.add_argument('--apply_backoff', action='store_true', help='Enable backoff to normal outputs if main output is invalid.')
    parser.add_argument("--backoff_path", type=str, default=None, help="Path to the backoff file.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset being evaluated.")
    args = parser.parse_args()

    # Determine the task type based on dataset name
    if args.dataset_name:
        if "math" in args.dataset_name.lower() or "aime" in args.dataset_name.lower():
            task_type = "math"
        elif "gpqa" in args.dataset_name.lower():
            task_type = "gpqa"
        elif "livecodebench" in args.dataset_name.lower():
            task_type = "livecodebench"
        elif args.dataset_name.lower().startswith("legalbench"):
            task_type = "legalbench"
        else:
            task_type = "qa"
    else:
        # Fallback for older calls without dataset_name
        if "math" in args.output_path.lower() or "aime" in args.output_path.lower():
            task_type = "math"
        elif "gpqa" in args.output_path.lower():
            task_type = "gpqa"
        elif "livecodebench" in args.output_path.lower():
            task_type = "livecodebench"
        elif "legalbench" in args.output_path.lower():
            task_type = "legalbench"
        else:
            task_type = "qa"

    if args.apply_backoff:
        if args.backoff_path:
            normal_output_path = args.backoff_path
        else:
            # Determine dataset name based on the output path
            # NOTE: To apply back off strategy for retrieval-augmented reasoning methods, please replace normal_output_path with your actual path for results with run_direct_gen.
            if 'gpqa' in args.output_path:
                dataset_name = 'gpqa'
                normal_output_path = './outputs/gpqa.qwq.direct/diamond.12.13,18:23.json'
                if 'extended' in args.output_path:
                    normal_output_path = './outputs/gpqa.qwq.direct/extended.12.28,15:44.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = './outputs/runs.baselines/gpqa.qwen2.5-32b-instruct.direct/diamond.12.14,20:34.json'
            elif 'math500' in args.output_path:
                dataset_name = 'math500'
                normal_output_path = './outputs/math500.qwq.direct/test.12.13,18:26.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = './outputs/runs.baselines/math500.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
            elif 'aime' in args.output_path:
                dataset_name = 'aime'
                normal_output_path = './outputs/aime.qwq.direct/2024.12.13,19:36.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = './outputs/runs.baselines/aime.qwen2.5-32b-instruct.direct/test.12.14,20:28.json'
            elif 'amc' in args.output_path:
                dataset_name = 'amc'
                normal_output_path = './outputs/amc.qwq.direct/test.12.14,14:31.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = './outputs/runs.baselines/amc.qwen2.5-32b-instruct.direct/test.12.14,20:26.json'
            elif 'livecode' in args.output_path:
                dataset_name = 'livecode'
                normal_output_path = './outputs/livecode.qwq.direct/test.12.13,21:24.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = './outputs/runs.baselines/livecode.qwen2.5-32b-instruct.direct/test.12.14,20:32.json'
            elif 'nq' in args.output_path:
                dataset_name = 'nq'
                normal_output_path = './outputs/runs.qa/nq.qwq.direct/test.12.15,14:50.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''
            elif 'triviaqa' in args.output_path:
                dataset_name = 'triviaqa'
                normal_output_path = './outputs/runs.qa/triviaqa.qwq.direct/test.12.15,15:35.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''
            elif 'hotpotqa' in args.output_path:
                dataset_name = 'hotpotqa'
                normal_output_path = './outputs/runs.qa/hotpotqa.qwq.direct/test.12.15,14:52.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''
            elif 'musique' in args.output_path:
                dataset_name = 'musique'
                normal_output_path = './outputs/runs.qa/musique.qwq.direct/test.12.27,16:44.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''
            elif 'bamboogle' in args.output_path:
                dataset_name = 'bamboogle'
                normal_output_path = './outputs/runs.qa/bamboogle.qwq.direct/test.12.28,9:51.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''
            elif '2wiki' in args.output_path:
                dataset_name = '2wiki'
                normal_output_path = './outputs/runs.qa/2wiki.qwq.direct/test.12.15,15:32.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''
            elif 'medmcqa' in args.output_path:
                dataset_name = 'medmcqa'
                normal_output_path = './outputs/runs.qa/medmcqa.qwq.direct/test.12.15,16:57.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''
            elif 'pubhealth' in args.output_path:
                dataset_name = 'pubhealth'
                normal_output_path = './outputs/runs.qa/pubhealth.qwq.direct/test.12.15,20:32.json'
                if 'qwq' not in args.output_path:
                    normal_output_path = ''

    # Load main output data
    with open(args.output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    # Load main metrics data
    if args.output_metrics_path:
        with open(args.output_metrics_path, mode='r', encoding='utf-8') as file:
            metrics = json.load(file)
    else:
        # If no metrics path provided, try to infer it from output_path
        output_metrics_path = args.output_path.replace('.json', '.metrics.json')
        if os.path.exists(output_metrics_path):
            with open(output_metrics_path, mode='r', encoding='utf-8') as file:
                metrics = json.load(file)
        else:
            metrics = {} # Initialize empty metrics if not found

    # Extract existing metrics
    if 'overall' in metrics:
        query_latency = metrics['overall']['query_latency']
        original_num_valid_answer = metrics['overall']['num_valid_answer']
    else:
        query_latency = metrics.get('query_latency', 'N/A')
        original_num_valid_answer = metrics.get('num_valid_answer', 'N/A')

    # Load normal output data if backoff is enabled
    normal_data = None
    if args.apply_backoff:
        if not os.path.exists(normal_output_path):
            raise FileNotFoundError(f"Normal output file not found at: {normal_output_path}")
        with open(normal_output_path, mode='r', encoding='utf-8') as file:
            normal_data = json.load(file)

    if task_type != "qa":
        # Existing evaluation for non-qa datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0

        # Initialize per-domain metrics
        domain_metrics = {}

        for i, item in enumerate(data):
            if task_type == "gpqa":
                labeled_answer = item["Correct Choice"]
                domain = item.get("High-level domain", "Unknown")
                mode = 'choose'
            elif task_type == "math":
                labeled_answer = item["answer"]
                domain = item.get("level", "Unknown")
                mode = 'gen'
            elif task_type == "livecodebench":
                labeled_answer = item["answer"]
                domain = item.get("difficulty", "Unknown")
                mode = 'gen'
            elif task_type == "legalbench":
                labeled_answer = item["answer"]
                domain = item.get("difficulty", "Unknown")
                mode = 'gen'
            else:
                raise ValueError(f"Unsupported dataset: {task_type}")

            output = item['Output']

            metric, pred_answer = evaluate_predictions(
                output=output, 
                labeled_answer=labeled_answer,
                mode=mode,
            )

            # Determine if the main method's answer is valid
            my_method_valid = (pred_answer != '' and not (mode == 'choose' and task_type == 'gpqa' and len(pred_answer) > 1))

            # If invalid and backoff is enabled, use normal method's output
            if args.apply_backoff and not my_method_valid and normal_data is not None:
                normal_item = normal_data[i]
                if task_type == "gpqa":
                    normal_labeled_answer = normal_item["Correct Choice"]
                    normal_mode = 'choose'
                elif task_type == "math":
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif task_type == "livecodebench":
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif task_type == "legalbench":
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                else:
                    raise ValueError(f"Unsupported dataset for backoff: {task_type}")

                normal_output = normal_item['Output']

                normal_metric, normal_pred_answer = evaluate_predictions(
                    output=normal_output, 
                    labeled_answer=normal_labeled_answer,
                    mode=normal_mode,
                )
                normal_valid = (normal_pred_answer != '' and not (normal_mode == 'choose' and task_type == 'gpqa' and len(normal_pred_answer) > 1))

                # Use normal method's result if valid
                if normal_valid:
                    metric = normal_metric
                    pred_answer = normal_pred_answer
                    my_method_valid = True

            # Track metrics per domain
            if domain not in domain_metrics:
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0}
            domain_metrics[domain]['total_num'] += 1
                
            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1
                domain_metrics[domain]['num_valid_answer'] += 1

        # Compute overall metrics
        overall_metrics = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0, 
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0, 
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0, 
            'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0, 
            'num_valid_answer': f'{num_valid_answer} of {data.__len__}',
            'query_latency': query_latency,
        }
        if args.apply_backoff:
            overall_metrics['original_num_valid_answer'] = original_num_valid_answer

        # Compute per-domain metrics
        domain_avg_metrics = {}
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
            }

        # Prepare final metrics
        final_metrics = {'overall': overall_metrics}
        if task_type == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    else:
        # Evaluation and backoff for qa dataset
        split = 'test'  # Modify as needed or extract from output_path

        if args.apply_backoff and normal_data is not None:
            # Apply backoff by replacing invalid outputs with normal outputs
            for i, item in enumerate(data):
                # Extract Pred_Answer from main output
                pred_answer = item['Pred_Answer']

                # Check if Pred_Answer is invalid
                if pred_answer == '':
                    # Replace Output with normal output
                    item['Output'] = normal_data[i]['Output']

        # Prepare input_list and output_list for run_evaluation
        input_list = [item['Question'] for item in data]
        output_list = [item['Output'] for item in data]

        # Estimate total_time (if available). Here, set to 0 as a placeholder.
        total_time = 0  # Modify if timing information is available

        # Run evaluation
        run_evaluation(
            filtered_data=data,
            input_list=input_list,
            output_list=output_list,
            dataset_name=task_type, # Pass task_type as dataset_name
            output_dir=args.output_path,
            total_time=total_time,
            split=split,
            apply_backoff=True,
        )
        # run_evaluation handles saving the metrics for livecode

    # Save metrics for non-livecode datasets
    if task_type != 'livecode' or not args.apply_backoff:
        # If dataset is livecode and backoff was applied, metrics are already saved by run_evaluation
        if args.apply_backoff:
            output_metrics_path = args.output_metrics_path.replace('.json', '.backoff.json')
        with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
