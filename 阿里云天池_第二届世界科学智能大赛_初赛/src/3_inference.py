import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import pandas as pd
from datasets import Dataset
import json


def generate_res(item, tokenizer, device, model):
    prompt = item['instruction'] + item['input']
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": "你是一个逻辑推理专家，擅长解决逻辑推理问题。"},
                                            {"role": "user", "content": prompt}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           ).to(device)

    gen_kwargs = {"max_length": 2000, "do_sample": True, "top_k": 1,"pad_token_id": tokenizer.pad_token_id}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return res


def most_frequent_char(char_list):
    # 创建一个字典来存储每个字符的出现次数
    frequency = {}
    # 遍历列表，增加每个字符的出现次数
    for char in char_list:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    # 找到出现次数最多的字符
    most_frequent = max(frequency, key=frequency.get)
    return most_frequent


def deal_process(input_file, output_file, res_dict):
    data_list = []
    # 按行读取数据
    with open(input_file) as reader:
        for line in reader:
            sample = json.loads(line)
            id = sample['id']
            print(id)
            res_list = res_dict[id]
            for index, question in enumerate(sample['questions']):
                question['answer'] = res_list[index]
            data_list.append(sample)
            # break

    with open(output_file, 'w') as writer:
        for sample in data_list:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')


def inference_deal(inference_file, num_vote, tokenizer, device, model) -> dict:
    df = pd.read_json(inference_file)
    cnt = 0

    res_dict = {}
    for item in Dataset.from_pandas(df):
        cnt += 1
        # print(item['id'])
        res_vote_list = []
        for i in range(num_vote):
            res_vote_list.append(generate_res(item, tokenizer, device, model))
        res = most_frequent_char(res_vote_list)
        if item['id'] in res_dict.keys():
            res_dict[item['id']].append(res)
        else:
            res_dict[item['id']] = [res]
        if cnt % 5 == 0:
            print(cnt)

    return res_dict


def main():
    model_path = './models/qwen1___5-32b-chat-gptq-int4/'
    lora_path = 'models/output/Qwen2_32b_int4_instruct_v3_lora_epoch_3/checkpoint-2476'
    device = "cuda:0"
    inference_file = "../data/external_data/round1_test_data_instruction.json"
    test_data = "../data/round1_test_data.jsonl"
    output_file = "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+".jsonl"
    num_vote = 3

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16, device_map=device)
    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    print(next(model.parameters()).device)

    res_dict: dict = inference_deal(inference_file, num_vote, tokenizer, device, model)

    deal_process(test_data, output_file, res_dict)


if __name__ == '__main__':
    main()
