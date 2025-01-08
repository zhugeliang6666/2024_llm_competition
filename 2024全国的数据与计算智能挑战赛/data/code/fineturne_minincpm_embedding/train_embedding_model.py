import os
import json
import time
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

train_data = pd.read_json("./data//raw_data/dev.json")
rule_data=pd.read_json("./data/raw_data/rules1.json")

model = SentenceTransformer(r'./data/user_data/MiniCPM-Embedding', trust_remote_code=True)
model = model.to(torch.bfloat16)
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    TaskType.FEATURE_EXTRACTION,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, config)

def embedding_s(data):
    l=[]
    for i in data:
        l.append(model.encode(i))
    return l
positive_data=embedding_s(train_data["question_text"])
anchor_data=embedding_s(rule_data["rule_text"])

import numpy as np
train_cos_sim_arr = cosine_similarity(positive_data, anchor_data)
train_sorted_indices = np.argsort(-train_cos_sim_arr, axis=1)
train_sorted_indices=train_sorted_indices+1

# 转换成列表格式
top_indices = train_sorted_indices[:, :10]

# 转换成列表格式
top_indices_list = [list(row) for row in top_indices]

# 添加到 DataFrame
train_data['predictrule_id'] = top_indices_list

# 添加到 DataFrame
train_data['predictrule_id'] = top_indices_list
train_exploded = train_data.explode('predictrule_id')
predict_mapping = rule_data.add_prefix('Predict').rename(columns={'Predictrule_id': 'predictrule_id'})
predict_mapping["rule_id"]=list(range(len(predict_mapping)))
predict_mapping["rule_id"]=predict_mapping["rule_id"]+1
train_joined_once = train_exploded.merge(predict_mapping, left_on='predictrule_id', right_on='predictrule_id', how='left')
# 创建一个空列用于存放规则文本
train_joined_once['rule_data'] = None

# 遍历每一条记录，并填充rule_data列
for index, row in train_joined_once.iterrows():
    rule_ids = row['rule_id_x']
    rule_texts = []
    for rule_id in rule_ids:
        rule_texts.append(rule_data.loc[int(rule_id) - 1, 'rule_text'])
    train_joined_once.at[index, 'rule_data'] = "\n".join(rule_texts)
train_joined_once["label_id"]=[int(i[0]) for i in train_joined_once["rule_id_x"]]
NUM_PROC = os.cpu_count()
train = (
    Dataset.from_pandas(train_joined_once)
    .filter(  # To create an anchor, positive, and negative structure, delete rows where the positive and negative are identical.
        lambda example: example["label_id"] != example["predictrule_id"],
        num_proc=NUM_PROC,
    )
)

# model = SentenceTransformer("/home/un/桌面/QC/2024_全国大数据智能大赛/rag/MiniCPM-Embedding", trust_remote_code=True)

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="./data/user_data/fineturne_minincpm_embedding/miniCPMv3",
    # Optional training parameters:
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    # per_device_eval_batch_size=BS,
    # eval_accumulation_steps=GRAD_ACC_STEP,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    #bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    lr_scheduler_type="cosine_with_restarts",
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=50,
    # report_to=REPORT_TO,  # Will be used in W&B if `wandb` is installed
    # run_name=EXP_NAME,
    do_eval=False
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train.select_columns(
        ["question_text", "rule_data", "Predictrule_text"]
    ),
    loss=loss
)

trainer.train()
# model.save_pretrained("./data/user_data/fineturne_minincpm_embedding/miniCPMv3")