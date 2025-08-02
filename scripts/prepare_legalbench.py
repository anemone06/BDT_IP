import json
import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Name of the legalbench task to process.")
    args = parser.parse_args()

    task_dir = os.path.join("legalbench", "tasks", args.task)
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"legalbench_{args.task}_test.json")

    # Try to find the data file, accommodating different extensions
    data_file = None
    for ext in ["jsonl", "tsv", "csv", "json"]:
        potential_file = os.path.join(task_dir, f"data.{ext}")
        if os.path.exists(potential_file):
            data_file = potential_file
            break
        # Fallback for other common names
        potential_file_train = os.path.join(task_dir, f"train.{ext}")
        if os.path.exists(potential_file_train):
            data_file = potential_file_train
            break

    if not data_file:
        print(f"Error: No data file (jsonl, tsv, csv, json) found in {task_dir}")
        return

    data = []
    if data_file.endswith('.tsv'):
        df = pd.read_csv(data_file, sep='\t')
        print("TSV Columns:", df.columns.tolist())
        print("TSV Head:\n", df.head())
        
        for _, row in df.iterrows():
            if args.task == "corporate_lobbying":
                # Corporate lobbying需要结合bill和company信息
                question = f"Bill Title: {row.get('bill_title', '')}\n\nBill Summary: {row.get('bill_summary', '')}\n\nCompany Name: {row.get('company_name', '')}\n\nCompany Description: {row.get('company_description', '')}\n\nQuestion: Is this bill relevant to the company?"
                answer = row.get('answer', '')
            elif args.task == "contract_qa":
                # Contract QA格式
                question = f"Contract Clause: {row.get('text', '')}\n\nQuestion: {row.get('question', '')}"
                answer = row.get('answer', '')
            elif args.task == "consumer_contracts_qa":
                # Consumer contracts QA格式
                question = f"Contract: {row.get('contract', '')}\n\nQuestion: {row.get('question', '')}"
                answer = row.get('answer', '')
            else:
                # 通用格式
                if "question" in df.columns:
                    text_col = "question"
                else:
                    text_col = "text"
                question = row.get(text_col, "")
                answer = row.get("answer", "")
            
            data.append({
                "Question": question,
                "Correct Choice": answer,
                "task": args.task
            })

    elif data_file.endswith('.jsonl'):
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
        if "question" in df.columns:
            text_col = "question"
        else:
            text_col = "text"
        for _, row in df.iterrows():
            data.append({"text": row.get(text_col, ""), "label": row.get("answer", "")})
    else: # Assuming json
        with open(data_file, 'r') as f:
            data = json.load(f)

    processed_data = []
    for item in data:
        if isinstance(item, dict) and "Question" in item:
            # 已经处理过的数据
            processed_data.append(item)
        else:
            # 未处理的数据，使用通用格式
            question = item.get("text", "")
            answer = item.get("label", "")
            processed_data.append({
                "Question": question,
                "Correct Choice": answer,
                "task": args.task
            })

    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print(f"Successfully preprocessed {len(processed_data)} items for task '{args.task}'.")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main() 