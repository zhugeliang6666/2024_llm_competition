import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model


class Config:
    train_file = "../data/external_data/round1_train_data_instruction_v3.json"
    model_path = "./models/qwen1___5-32b-chat-gptq-int4/"
    lora_out = "./models/output/Qwen2_32b_int4_instruct_v3_lora_epoch_3/"


def main():
    df = pd.read_json(Config.train_file)
    ds = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_path, use_fast=False,
                                              trust_remote_code=True)

    def process_func(example):
        MAX_LENGTH = 1800  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性 512，784，1024
        instruction = tokenizer(
            f"<|im_start|>system\n你是一个逻辑推理专家，擅长解决逻辑推理问题。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    model = AutoModelForCausalLM.from_pretrained(Config.model_path, device_map="auto", torch_dtype=torch.float16)
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
        output_dir=Config.lora_out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=300,
        learning_rate=1e-4,
        save_on_each_node=True,
        save_total_limit=2,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()


if __name__ == '__main__':
    main()
