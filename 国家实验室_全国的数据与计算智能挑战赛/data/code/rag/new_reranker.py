import pandas as pd
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with PEFT and LoRA.")
    parser.add_argument("--root_dir", type=str, default="./data/raw_data/dev.json", help="测试集")
    parser.add_argument("--rag_data_index", type=str, default="", help="tok30_rule_id")
    parser.add_argument("--all_rule_path", type=str, default="", help="规则数据集")
    parser.add_argument("--model_path", type=str, default="", help="模型路径")
    parser.add_argument("--output_reranker_tok30_scores_path", type=str, default="", help="tok30_scores_path")
    parser.add_argument("--output_reranker_tok30_2_tok10", type=str, default="", help="tok30_2_tok10")
    return parser.parse_args()
args = parse_args()


root_dir = args.root_dir
train_texts=pd.read_json(root_dir)
import pickle
with open(args.rag_data_index, 'rb') as file:
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "给定查询 A 和文章 B，通过提供“是”或“否”的预测来确定文章是否包含查询的答案。"
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to('cuda')
model.eval()

from tqdm import tqdm
reranker_label=[]
reranker_labels=[]
for j in tqdm(range(len(queries))):
    for i in range(30):
        with torch.no_grad():
            inputs = get_inputs([[queries[j], passages[j][i]]], tokenizer).to(model.device)
            all_scores = model(**inputs, return_dict=True, cutoff_layers=[28])
            all_scores = [scores[:, -1].view(-1, ).float() for scores in all_scores[0]]
            reranker_label.append(all_scores[0].cpu().numpy().tolist()[0])
    reranker_labels.append(reranker_label)
    reranker_label=[]
import pickle
with open(args.output_reranker_tok30_scores_path, 'wb') as file:
    pickle.dump(reranker_labels, file)
# 对每组值进行排序
sorted_rag_data_index = []
for labels, indices in zip(reranker_labels, rag_data_index):
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)
    sorted_rag_data_index.append([indices[i] for i in sorted_indices[:10]])

# 打印排序后的rag_data_index
for group in sorted_rag_data_index:
    print(group)
    break
pred_labels=[]
for i, row in train_texts.iterrows():
    predicted_order = sorted_rag_data_index[i]
    pred_labels.append(predicted_order)
import pickle
with open(args.output_reranker_tok30_2_tok10, 'wb') as file:
    pickle.dump(pred_labels, file)
