import re
from collections import defaultdict

import vllm
import pandas as pd
import pickle
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest
import json
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser(description="vllm")
    parser.add_argument("--rag_data", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--relu_id_tok10_path", type=str)
    parser.add_argument("--model_name_lora", type=str)
    parser.add_argument("--jsol_output_name", type=str)
    return parser.parse_args()
args = parse_args()

model_path = "./data/user_data/Qwen2.5-7B-Instruct"
rag_data = pd.read_json(args.rag_data)
test_data = pd.read_json(args.data_path)
# test_data = pd.concat([dev_df], axis=0, ignore_index=True)
with open(args.relu_id_tok10_path, 'rb') as file:
    rag_data_index = pickle.load(file)
test_data["rule_id"]=rag_data_index
print(test_data)
# test_data = test_data

tokenizer = AutoTokenizer.from_pretrained(model_path)


def apply_template(row):
    question_text = row['question_text']
    rag_prompt = "\n以下是相关上下文：\n"
    rag_data_index = row["rule_id"]
    rag_datas = [rag_data.loc[int(i) - 1].values[1] for i in rag_data_index][:3]
    temp = question_text + rag_prompt + '\n'.join(rag_datas)
    messages = [
        {"role": "system",
         "content": f"你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为: 答案是：A|B|C|D"},
        {"role": "user", "content": f"{temp}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


test_data["messages"] = test_data.apply(apply_template, axis=1)

llm = vllm.LLM(
    model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    # dtype="half",
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True,
    enable_lora=True
)

test_pred_dict = defaultdict(list)
model_name_lora=os.path.join(args.model_name_lora,os.listdir(args.model_name_lora)[-1])
for _ in range(11):
    responses = llm.generate(
        test_data["messages"],
        use_tqdm=True,
        lora_request=LoRARequest("lora1", 1, model_name_lora)
    )
    for index, response in enumerate([x.outputs[0].text for x in responses]):
        test_pred_dict[index].append(response)


def most_frequent(lst):
    return max(set(lst), key=lst.count)


test_pred_list = [most_frequent(test_pred_dict[i]) for i in range(0, len(test_pred_dict))]
# print(test_pred_list)
test_data['answer'] = test_pred_list
test_data['rule_id'] = test_data['rule_id'].apply(lambda x: [str(i) for i in x])
output = []
for index, row in test_data.iterrows():
    output.append({
        "question_id": str(row['question_id']),
        # "question_text": row['question_text'],
        "answer": row['answer'],
        "rule_id":row['rule_id']
    })
print(output)
json_output = json.dumps(output, indent=4, ensure_ascii=False)

with open(args.jsol_output_name, 'w', encoding='utf-8') as f:
    f.write(json_output)
print("程序运行完毕")
