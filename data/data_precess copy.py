import pandas as pd
import json

def process_tsv_to_json(input_file, output_file):
    """
    将TSV文件处理成指定格式的JSON文件，将contract内容融合到Question中
    
    Args:
        input_file (str): 输入的TSV文件路径
        output_file (str): 输出的JSON文件路径
    """
    # 读取TSV文件
    df = pd.read_csv(input_file, sep='\t')
    
    # 创建结果列表
    result = []
    
    # 遍历每一行数据
    for _, row in df.iterrows():
        # 将contract内容作为上下文融合到question中
        question_text = f"Context: {row['contract']}\n\nQuestion: {row['question']}"
        
        # 创建字典格式（只包含Question和Correct Choice）
        item = {
            'Question': question_text,
            'Correct Choice': row['answer']
        }
        
        result.append(item)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理了 {len(result)} 条数据")
    return result

def process_tsv_alternative(input_file, output_file):
    """
    使用纯Python处理TSV文件的替代方法，将contract内容融合到Question中
    
    Args:
        input_file (str): 输入的TSV文件路径  
        output_file (str): 输出的JSON文件路径
    """
    result = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 获取标题行
        if lines:
            header = lines[0].strip().split('\t')
            print(f"检测到的列名: {header}")
        
        # 处理数据行
        for line_num, line in enumerate(lines[1:], 1):
            try:
                # 分割TSV行
                parts = line.strip().split('\t')
                
                if len(parts) >= 4:
                    index = parts[0]      # 第1列是index
                    contract = parts[1]   # 第2列是contract
                    question = parts[2]   # 第3列是question
                    answer = parts[3]     # 第4列是answer
                    
                    # 将contract内容融合到question中
                    full_question = f"Context: {contract}\n\nQuestion: {question}"
                    
                    # 创建字典（只包含Question和Correct Choice）
                    item = {
                        'Question': full_question,
                        'Correct Choice': answer
                    }
                    
                    result.append(item)
                else:
                    print(f"警告：第{line_num}行数据不完整，跳过处理")
                    
            except Exception as e:
                print(f"处理第{line_num}行时出错: {e}")
                continue
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理了 {len(result)} 条数据")
    return result

# 使用示例
if __name__ == "__main__":
    input_filename = 'test.tsv'  # 替换为你的TSV文件名
    output_filename = 'consumer_contracts_qa_test.json'   # 输出的JSON文件名
    
    # 处理TSV文件，将contract内容融合到Question中
    try:
        print("使用pandas处理TSV文件...")
        result = process_tsv_to_json(input_filename, output_filename)
        
        # 打印前2条数据作为示例
        print("\n前2条处理结果：")
        for i, item in enumerate(result[:2]):
            print(f"{i+1}. Question: {item['Question'][:200]}...")
            print(f"   Correct Choice: {item['Correct Choice']}")
            print("-" * 50)
            
    except ImportError:
        print("pandas未安装，使用纯Python方法...")
        result = process_tsv_alternative(input_filename, output_filename)
        
        # 打印前2条数据作为示例
        print("\n前2条处理结果：")
        for i, item in enumerate(result[:2]):
            print(f"{i+1}. Question: {item['Question'][:200]}...")
            print(f"   Correct Choice: {item['Correct Choice']}")
            print("-" * 50)
            
    except FileNotFoundError:
        print(f"文件 {input_filename} 未找到，请检查文件路径")
        print("请将TSV文件放在脚本同一目录下，并修改input_filename变量")