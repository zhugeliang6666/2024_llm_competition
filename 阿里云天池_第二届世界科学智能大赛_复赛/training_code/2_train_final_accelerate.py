from lora_config import *
from util import *
import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig, GPTQConfig
from peft import LoraConfig, TaskType, get_peft_model
import gc
import sys


def main(task_type):
    torch.cuda.empty_cache()
    gc.collect()
    instruct_data: list = origin_file_instruct(train_file_list, True)
    # instruct_data = shuffle_keep(instruct_data, 0.8)
    print(f"data len: {len(instruct_data)}")
    ds = Dataset.from_list(instruct_data)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                              trust_remote_code=True)

    def process_func(example):
        MAX_LENGTH = 1024
        prompt = example["prompt"]
        # print(prompt)
        instruction = tokenizer(
            f"<|im_start|>system\n你是一个逻辑推理专家，擅长解决逻辑推理问题。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False)
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 quantization_config=GPTQConfig(bits=4, disable_exllama=True))

    model.enable_input_require_grads()
    print(next(model.parameters()).device)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1  # Dropout 比例
    )
    model = get_peft_model(model, config)
    args = TrainingArguments(
        output_dir=lora_out + task_type + "/",
        per_device_train_batch_size=per_device_train_batch_size,  # 2
        gradient_accumulation_steps=gradient_accumulation_steps,  # 4
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        save_steps=3000,
        learning_rate=1e-4,
        save_on_each_node=True,
        save_total_limit=1,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    task_type = sys.argv[1]
    print(task_type)
    main(task_type)
