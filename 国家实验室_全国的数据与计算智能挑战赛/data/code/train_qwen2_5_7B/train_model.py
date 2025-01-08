from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoTokenizer
import torch
import re
from tqdm import tqdm
import json
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with PEFT and LoRA.")
    parser.add_argument("--train_data", type=str, default="./data/raw_data/dev.json", help="训练数据")
    parser.add_argument("--output_dir", type=str, default="./data/user_data/chusai_model", help="输出路径")
    parser.add_argument("--logging_steps", type=int, default=200, help="日志步数")
    parser.add_argument("--save_steps", type=int, default=200, help="保存步数")
    return parser.parse_args()
arg = parse_args()

model_dir = "./data/user_data/Qwen2.5-7B-Instruct"

### 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

train_dir=arg.train_data

def process_func(example):
    """
    将数据集进行预处理
    """
    # global i
    MAX_LENGTH = 512 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\n你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为:答案是：A。<|endoftext|>\n<|user|>\n{example['question_text']}<|endoftext|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['answer']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

model.enable_input_require_grads()
import pandas as pd
from datasets import Dataset
train_df = pd.read_json(train_dir)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, config)
args = TrainingArguments(
    output_dir=arg.output_dir, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=arg.logging_steps,
    num_train_epochs=2,
    save_steps=arg.save_steps,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    # fp16=True,
    save_total_limit=2,
    # seed=2024
)
from transformers import DataCollatorWithPadding
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()