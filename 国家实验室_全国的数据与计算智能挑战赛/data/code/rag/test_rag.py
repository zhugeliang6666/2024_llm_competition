from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import datasets
import evaluate
import transformers
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="RAG")
    parser.add_argument("--root_dir", type=str, default="./data/raw_data/test.json", help="测试集")
    parser.add_argument("--rules_data", type=str, default="./data/raw_data/rule1.json", help="规则数据")
    parser.add_argument("--model_path", type=str, default="./data/user_data/MiniCPM-Embedding", help="MiniCPM_Embedding模型路径")
    parser.add_argument("--save_lora_steps", type=str, default="", help="lora权重")
    parser.add_argument("--output_dir", type=str, default="", help="输出路径")
    return parser.parse_args()
args = parse_args()




root_dir = args.root_dir #测试集
test_texts_data=pd.read_json(root_dir)
import faiss
import numpy as np
import torch
from peft import PeftModel
test_texts=pd.read_json(args.rules_data)
model_lora_path=os.path.join(args.save_lora_steps,os.listdir(args.save_lora_steps)[-1])
model = SentenceTransformer(args.model_path, trust_remote_code=True)
model=PeftModel.from_pretrained(model=model,model_id=model_lora_path)

# 存储所有向量和标签
all_vectors = []

for i in test_texts["rule_text"]:
    all_vectors.append(model.encode(i,normalize_embeddings=True))


# 将列表转换为NumPy数组
all_vectors = np.vstack(all_vectors)
import faiss
print(len(all_vectors))
# 确定向量的维度
d = all_vectors.shape[1]

# 创建一个索引 - 这里使用FlatL2索引
index = faiss.IndexFlatL2(d)

# 将数据添加到索引
index.add(all_vectors)
# 保存索引
faiss.write_index(index, './data/user_data/MiniCPM_result.index')

# 保存向量和标签（使用NumPy保存）
np.save('./data/user_data/MiniCPM_old_data.npy', all_vectors)

# data = np.load('/home/un/桌面/QC/2024_全国大数据智能大赛/new_复赛_code/my_data/MiniCPM_old_data.npy')
# index = faiss.read_index('./data/user_data/MiniCPM_result.index')

def get_RAG_result1(query_sentence, k):
    query_vector = model.encode(query_sentence,normalize_embeddings=True)
    query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
    D, I = index.search(query_vector, k)  # D是距离，I是索引
    old_data=test_texts
    l=[]
    for i, idx in enumerate(I[0]):
        l.append(idx+1)
    return l
from tqdm import tqdm
pred_labels=[]
for i, row in tqdm(test_texts_data.iterrows()):
    predicted_order = get_RAG_result1(row["question_text"],k=30)
    pred_labels.append(predicted_order)

test_texts_data["rule_ids"]=pred_labels
import pickle
with open(args.output_dir, 'wb') as file:
    pickle.dump(pred_labels, file)