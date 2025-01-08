from openai import OpenAI
from tqdm import tqdm
import argparse
import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="vllm")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--relu_id_tok10_path", type=str)
    parser.add_argument("--relu_data_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--jsol_output_name", type=str)
    parser.add_argument("--start_vllm", type=str, default="True")
    return parser.parse_args()

args = parse_args()
if args.start_vllm=="True":
    time.sleep(90)

if args.data_name=="初赛":
    data_num=4501
if args.data_name=="复赛训练集":
    data_num=5001
if args.data_name=="复赛":
    data_num=2001
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "sk-xxx"
openai_api_base = "http://localhost:8000/v1"
# openai_api_base='http://127.0.0.1:8000/v1'
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
print("----------------------开始查看日志----------------------")

try:
    with open("start_server.log","r",encoding="utf8") as f:
        data=f.read()
        # print(data)
except Exception as e:
    print(e)
    print("运行失败")
import pickle
import pandas as pd
with open(args.relu_id_tok10_path, 'rb') as file:
    rag_data_index = pickle.load(file)
# print(rag_data_index)
# model_dir = "/home/un/桌面/QC/qwen2_5/Qwen2.5-7B-Instruct"
rag_data=pd.read_json(args.relu_data_path)
test_data=pd.read_json(args.data_path)
test_data["rule_id"]=rag_data_index

pred_list=[]
try:
    for index, row in tqdm(test_data.iterrows()):
        instruction = instruction ="你是一位经验丰富的应急响应专家，擅长解决应急场景的问题。以下是一个逻辑推理的题目，形式为单项选择题。"
        input_value = row['question_text']
        rag_prompt="\n以下是相关上下文：\n"
        rag_data_index=row["rule_id"]
        rag_datas=[rag_data.loc[int(i)-1].values[1] for i in rag_data_index][:3]
        temp = '\n'.join(rag_datas)
        chat_response = client.chat.completions.create(
        model=args.model_name,
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value+rag_prompt+temp}"}
        ]
    )

        pred_list.append(chat_response.choices[0].message.content)

    import json
    sub=pd.DataFrame(list(range(1,data_num)),columns=["question_id"])
    sub["answer"]=pred_list
    sub["rule_id"]=test_data["rule_id"]
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
    with open("./data/prediction_result/"+args.jsol_output_name, 'w', encoding='utf-8') as f:
        f.write(json_output)
    print("程序运行完毕")
except:
    pass