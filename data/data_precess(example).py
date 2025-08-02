import pandas as pd
import json

def process_tsv_to_json(input_file, output_file):
    """
    将TSV文件处理成指定格式的JSON文件
    
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
        # 构建问题文本，结合index和question列
        question_text = f"{row['question']}"
        
        # 创建字典格式
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
    使用纯Python处理TSV文件的替代方法
    
    Args:
        input_file (str): 输入的TSV文件路径  
        output_file (str): 输出的JSON文件路径
    """
    result = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 跳过标题行
        for line in lines[1:]:
            # 分割TSV行
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                index = parts[1]  # 第二列是index
                question = parts[2]  # 第三列是question
                answer = parts[3]   # 第四列是answer
                
                # 构建完整问题
                full_question = f"{index} {question}"
                
                # 创建字典
                item = {
                    'Question': full_question,
                    'Correct Choice': answer
                }
                
                result.append(item)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理了 {len(result)} 条数据")
    return result

# 使用示例
if __name__ == "__main__":
    # 方法1：使用pandas（推荐）
    try:
        result = process_tsv_to_json('test.tsv', 'International.json')
        
        # 打印前几条数据作为示例
        print("\n前3条处理结果：")
        for i, item in enumerate(result[:3]):
            print(f"{i+1}. {item}")
            
    except ImportError:
        print("pandas未安装，使用替代方法...")
        # 方法2：使用纯Python
        result = process_tsv_alternative('test.tsv', 'International.json')
        
        # 打印前几条数据作为示例
        print("\n前3条处理结果：")
        for i, item in enumerate(result[:3]):
            print(f"{i+1}. {item}")