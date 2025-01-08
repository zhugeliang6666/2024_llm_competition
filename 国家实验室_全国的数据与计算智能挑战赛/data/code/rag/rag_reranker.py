from transformers import AutoModel, LlamaTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from peft import PeftModel
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with PEFT and LoRA.")
    parser.add_argument("--MiniCPM_Reranker_path", type=str, default="./data/user_data/MiniCPM-Reranker", help="重排模型路径")
    parser.add_argument("--root_dir", type=str, default="./data/raw_data/test.json", help="测赛集路径")
    parser.add_argument("--rule_id_tok30", type=str, default="", help="tok30文件")
    parser.add_argument("--all_rule_path", type=str, default="", help="规则数据集")
    parser.add_argument("--MiniCPM_test_rules_reranker_tok30_scores", type=str, default="", help="MiniCPM_test_reranker_tok30_scores")
    parser.add_argument("--MiniCPM_test_rules_reranker_new_tok30_2_tok10", type=str, default="", help="MiniCPM_test_reranker_tok30_2_tok10")
    return parser.parse_args()
args = parse_args()




class MiniCPMRerankerLLamaTokenizer(LlamaTokenizer):
    def build_inputs_with_special_tokens(
            self, token_ids_0, token_ids_1 = None
        ):
            if token_ids_1 is None:
                return super().build_inputs_with_special_tokens(token_ids_0)
            bos = [self.bos_token_id]
            sep = [self.eos_token_id]
            return bos + token_ids_0 + sep + token_ids_1

model_name = args.MiniCPM_Reranker_path
tokenizer = MiniCPMRerankerLLamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "right"

model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to("cuda")
# model=PeftModel.from_pretrained(model,model_id="/home/un/桌面/QC/2024_全国大数据智能大赛/复赛_code/finetune_minincpm_reranker/miniCPMv2-rank/checkpoint-873")
model.eval()

@torch.no_grad()
def rerank(input_query, input_docs):
    tokenized_inputs = tokenizer([[input_query, input_doc] for input_doc in input_docs], return_tensors="pt", padding=True, truncation=True, max_length=4096) 

    for k in tokenized_inputs:
      tokenized_inputs [k] = tokenized_inputs[k].to("cuda")

    outputs = model(**tokenized_inputs)
    score = outputs.logits
    return score.float().detach().cpu().numpy()

import pandas as pd
root_dir = args.root_dir
train_texts=pd.read_json(root_dir)
import pickle
with open(args.rule_id_tok30, 'rb') as file:
    rag_data_index = pickle.load(file)
rules=pd.read_json(args.all_rule_path)
l=[]
l1=[]
for i in range(len(rag_data_index)):
    for j in rag_data_index[i]:
        # print(j)
        l.append(rules["rule_text"].loc[j-1])
    l1.append(l)
    l=[]
queries = [i for i in train_texts["question_text"]]
passages = l1
from tqdm import tqdm
INSTRUCTION = "Query: "
queries = [INSTRUCTION + query for query in queries]
# print(queries)
scores = []
# len(queries)
reranker_label=[]
reranker_labels=[]
for j in tqdm(range(len(queries))):
    for i in range(30):
        reranker_label.append(rerank(queries[j],[passages[j][i]])[0][0])
    reranker_labels.append(reranker_label)
    reranker_label=[]
import pickle
with open(args.MiniCPM_test_rules_reranker_tok30_scores, 'wb') as file:
    pickle.dump(reranker_labels, file)

# 对每组值进行排序
sorted_rag_data_index = []
for labels, indices in zip(reranker_labels, rag_data_index):
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)
    # print(sorted_indices)
    # print(indices)
    sorted_rag_data_index.append([indices[i] for i in sorted_indices][:10])

# 打印排序后的rag_data_index
for group in sorted_rag_data_index:
    print(group)
    break
pred_labels=[]
for i, row in train_texts.iterrows():
    predicted_order = sorted_rag_data_index[i]
    pred_labels.append(predicted_order)

import pickle
with open(args.MiniCPM_test_rules_reranker_new_tok30_2_tok10, 'wb') as file:
    pickle.dump(pred_labels, file)