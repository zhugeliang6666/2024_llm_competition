import json
from collections import Counter
import os
import pandas as pd
import argparse
print("---------------------------开始投票---------------------------")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with PEFT and LoRA.")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--relu_id_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--jsol_output_name", type=str)
    parser.add_argument("--start_vllm", type=str, default="True")
    return parser.parse_args()
args = parse_args()
if args.data_name=="初赛":
    data_num=4501
if args.data_name=="复赛训练集":
    data_num=5001
if args.data_name=="复赛":
    data_num=2001

# 执行硬投票
def hard_voting(data_files):
    # 初始化一个空的列表来存储所有的答案
    all_answers = []
    
    # 遍历提供的文件列表
    for file in data_files:
        # 读取JSON文件中的数据
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取'answer'列，并将其添加到答案列表中
        answers = [item['answer'] for item in data]
        all_answers.append(answers)
    
    # 将所有答案转换为DataFrame
    df_answers = pd.DataFrame(all_answers).T
    
    # 对每一行的答案进行硬投票
    voted_results = []
    for index, row in df_answers.iterrows():
        # 计算每行中每个答案的出现次数
        counter = Counter(row)
        # 获取得票最多的答案
        # print(counter)
        most_common_answer = counter.most_common(1)[0][0]
        voted_results.append(most_common_answer)
    
    return voted_results

# 保存结果为JSONL
def save_jsonl(data):
    sub=pd.DataFrame(list(range(1,data_num)),columns=["question_id"])
    sub["answer"]=data["answer"]
    sub["rule_id"]=data["rule_id"]
    sub['rule_id'] = sub['rule_id'].apply(lambda x: [str(i) for i in x])
    # 构建输出列表
    output = []
    for index, row in sub.iterrows():
        output.append({
            "question_id": str(row['question_id']),
            "answer": row['answer'],
            "rule_id": row['rule_id']
        })

    # 将输出列表转换为JSON字符串
    json_output = json.dumps(output, indent=4, ensure_ascii=False)

    # 输出到控制台
    # print(json_output)

    # 如果需要保存到文件
    with open(args.jsol_output_name, 'w', encoding='utf-8') as f:
        f.write(json_output)
    print("程序运行完毕")

# 定义文件路径
path="./data/prediction_result/"
data1 = path+"model1.jsonl"
data2 = path+"model2.jsonl"
data3 = path+"model3.jsonl"
data4 = path+"model4.jsonl"
data5 = path+"model5.jsonl"
data6 = path+"model6.jsonl"
data7 = path+"model7.jsonl"
data8 = path+"model8.jsonl"
data9 = path+"model9.jsonl"
data10 = path+"model10.jsonl"
data11 = path+"model11.jsonl"
try:
    # 数据文件列表
    data_files = [data1, data2, data3,data4, data5, data6,data7, data8, data9,data10,data11]
    data=pd.read_json(data1)
    # 执行硬投票
    results = hard_voting(data_files)
    
    # 输出结果
    # print(results)
    data["answer"]=results
    print("---------------------------投票完成---------------------------")
    
    # 保存结果为JSONL文件
    save_jsonl(data)
    print("硬投票结果已保存")
    
except Exception as e:
    print(e)