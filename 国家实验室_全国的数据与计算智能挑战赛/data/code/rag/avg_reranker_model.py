import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with PEFT and LoRA.")
    parser.add_argument("--bge_reranker_scores", type=str, default="", help="bge_reranker_scores")
    parser.add_argument("--minicpm_reranker_scores", type=str, default="", help="minicpm_reranker_scores")
    parser.add_argument("--rag_data_index", type=str)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--bge_reranker_weight", type=float)
    parser.add_argument("--minicpm_reranker_weight", type=float)
    parser.add_argument("--ouput_dir", type=str)
    return parser.parse_args()
args = parse_args()

with open(args.bge_reranker_scores, 'rb') as file:
    bge_reranker_scores = pickle.load(file)
import pickle
with open(args.minicpm_reranker_scores, 'rb') as file:
    minicpm_reranker_scores = pickle.load(file)
import pickle
with open(args.rag_data_index, 'rb') as file:
    rag_data_index = pickle.load(file)
import numpy as np

sum_score=np.array(bge_reranker_scores)*args.bge_reranker_weight+np.array(minicpm_reranker_scores)*args.minicpm_reranker_weight
avg_score=sum_score/2
avg_score_list=avg_score.tolist()
# 对每组值进行排序
sorted_rag_data_index = []
for labels, indices in zip(avg_score_list, rag_data_index):
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)
    # print(sorted_indices)
    # print(indices)
    sorted_rag_data_index.append([indices[i] for i in sorted_indices][:10])

# 打印排序后的rag_data_index
for group in sorted_rag_data_index:
    print(group)
    break

import pandas as pd
root_dir = args.root_dir #测试集
test_texts_data=pd.read_json(root_dir)

pred_labels=[]
for i, row in test_texts_data.iterrows():
    predicted_order = sorted_rag_data_index[i]
    pred_labels.append(predicted_order)

import pickle
# ouput_dir="/home/un/桌面/QC/2024_全国大数据智能大赛/new_复赛_code/my_data/finetune_bge+minincpm_tok30_2_tok10.pkl"
with open(args.ouput_dir, 'wb') as file:
    pickle.dump(pred_labels, file)