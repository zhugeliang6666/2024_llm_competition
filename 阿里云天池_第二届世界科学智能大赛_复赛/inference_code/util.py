import re
from collections import Counter

import httpx
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import json
import time
from tqdm import tqdm

# logger.remove()  # 移除默认的控制台输出
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip")

from lora_config import inference_url, http_client, check_mode


def call_api_local_check(MODEL_NAME, data, temperature):
    max_checks = 5
    attempt_check = 0
    step1_res = ""
    while attempt_check < max_checks:
        step1_res = call_api_local(MODEL_NAME, data["prompt"], temperature)
        answer_str = data['options'][char_to_number(step1_res)]
        prompt_check = prompt_lora_step_check(data, answer_str)
        check = call_api_local(check_mode, prompt_check, temperature)
        if check == "F":
            # prompt = data["prompt"]
            # print(f"{prompt}")
            logger.info(f"answer_str = {answer_str},step1_res={step1_res},check={check}")
            attempt_check += 1
        else:
            return step1_res
    return step1_res


def call_api_remote_check(MODEL_NAME, data, temperature):
    max_checks = 5
    attempt_check = 0
    step1_res = ""
    while attempt_check < max_checks:
        step1_res = call_api_remote(MODEL_NAME, data["prompt"], temperature)

        answer_str = data['options'][char_to_number(step1_res)]
        prompt_check = prompt_lora_step_check(data, answer_str)
        check = call_api_remote(check_mode, prompt_check, temperature)
        if check == "F":
            # prompt = data["prompt"]
            # logger.info(f"{prompt}")
            logger.info(f"answer_str = {answer_str},step1_res={step1_res},check={check}")
            attempt_check += 1
        else:
            return step1_res
    return step1_res


def api_retry(MODEL_NAME, data, local, temperature, use_check=False):
    max_retries = 5
    retry_delay = 60  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            if use_check:
                # if local:
                #     return call_api_local_check(MODEL_NAME, data, temperature=temperature)
                # else:
                #     return call_api_remote_check(MODEL_NAME, data, temperature=temperature)
                return "A"
            else:
                if local:
                    return call_api_local(MODEL_NAME, data["prompt"], temperature=temperature)
                else:
                    return call_api_remote(MODEL_NAME, data["prompt"], temperature=temperature)
        except Exception as e:
            id = data["id"]
            attempts += 1
            if attempts < max_retries:
                # logger.warning(f"Attempt {attempts} failed for text: {prompt}.Error: {e}, Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed for text: {id}. Error: {e}")
                raise


def call_api_local(MODEL_NAME, query, temperature):
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-xxx",
        http_client=httpx.Client(timeout=10000000)
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": query}
        ],
        temperature=temperature
    )
    return completion.choices[0].message.content


def call_api_remote(MODEL_NAME, query, temperature):
    client = OpenAI(
        # base_url="http://localhost:8000/v1",
        base_url=inference_url,
        api_key="sk-xxx",
        http_client=http_client,
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": query}
        ],
        temperature=temperature
    )
    return completion.choices[0].message.content


def prompt_lora_step_choose(item):
    prompt = f"""以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题,并在最后一行直接输出答案，最后一行的格式为:"A"。题目如下：
### 题目:
{item['instruction']}

### 问题:
{item['input']}
"""
    # print(prompt)
    return prompt


def prompt_lora_step_choose_opt(item):
    prompt = f"""以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。\n## 在闭世界假设（CWA）下，我们只能使用题目中明确给出的信息来回答问题。任何未明确提及的信息都被认为是不成立的.请逐步分析问题,并在最后一行直接输出答案，最后一行的格式为:"答案:A|B|C|D|E|F|G"。题目如下：
### 题目:
{item['instruction']}

### 问题:
{item['input']}
"""
    # print(prompt)
    return prompt


def get_prompt_item(item):
    prompt = item['instruction'] + item['input']
    return prompt


# 这里使用extract抽取模获得抽取的结果
def extract(input_text):
    ans_pattern = re.compile(r"(.)", re.S)

    # ans_pattern = re.compile(r"答案是：(.)", re.S)
    problems = ans_pattern.findall(input_text)
    # print(problems)
    if (problems == []):
        return 'A'
    return problems[0]


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


def inference_detail(datas, MODEL_NAME, local, vote_num, use_check, temperature):
    inference_res = {}
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_data = {}
        # 送入多线程任务
        for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
            for i in range(0, vote_num):
                if isinstance(MODEL_NAME, str):
                    future1 = executor.submit(api_retry, MODEL_NAME, data, local=local, use_check=use_check,
                                              temperature=temperature)
                    future_data[future1] = (data, data['id'])
                elif isinstance(MODEL_NAME, list):
                    for MODEL_NAME_sub in MODEL_NAME:
                        future1 = executor.submit(api_retry, MODEL_NAME_sub, data, local=local, use_check=use_check,
                                                  temperature=temperature)
                        future_data[future1] = (data, data['id'])

        # print(len(future_data))
        result_dict = {}
        for future in tqdm(as_completed(future_data), total=len(future_data.keys()), desc="Processing tasks"):
            data = future_data[future][0]
            current_id = future_data[future][1]
            try:
                data["answer"] = future.result()
                # print(data["answer"])
                if current_id in result_dict.keys():
                    result_dict[current_id].append(data)
                else:
                    result_dict[current_id] = [data]
            except Exception as e:
                logger.error(f"Failed to process text: {data}. Error: {e}")
        # print(len(result_dict.keys()))
        # 解决投票问题
        for current_id, problem_list in result_dict.items():
            result_list = [problem['answer'] for problem in problem_list]
            vote_res = most_frequent_char(result_list)
            inference_res[current_id] = vote_res
    return inference_res


def char_to_number(char):
    return ord(char) - 65


def origin_file_instruct(input_file_list, has_out_put):
    data_list = []
    # 按行读取数据
    for input_file in input_file_list:
        with open(input_file) as reader:
            for line in reader:
                sample = json.loads(line)
                data_list.append(sample)
                # break

    result_list = []
    for item in data_list:
        problem = item['problem']
        if "questions" in item.keys():
            try:
                questions = item['questions']
                id = item['id']
                sub_id = 0
                for question in questions:
                    options_str = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(question['options']))
                    item = {
                        "instruction": f"{problem}",
                        "input": f"回答以下选择题:\n{question['question']}\n{options_str}",
                        "id": f"{id}-{sub_id}",
                        "question": question['question'],
                        "options": question['options'],
                        "problem": problem,
                    }
                    if has_out_put:
                        item["output"] = question["answer"]
                    # item["prompt"] = prompt_lora_step_choose(item)
                    item["prompt"] = prompt_lora_step_choose_opt(item)
                    # item["close_rule"] = prompt_lora_step_choose_cwd_rule(item)

                    sub_id += 1
                    result_list.append(item)
            except Exception as e:
                logger.error(f"Failed to process text: {item}. Error: {e}")

    return result_list


def prompt_lora_step_check(qustion, answer):
    prompt = f"""以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。\n## 在闭世界假设（CWA）下，我们只能使用题目中明确给出的信息来回答问题。任何未明确提及的信息都被认为是不成立的。\n## 在开放世界假设（OWA）下，未提及的信息既不被确认为真，也不被确认为假。这意味着在OWA下，我们可能会考虑题目中未明确提及的可能性。\n
首先总结 闭世界假设(close-world assumption)和 开发世界假设(Open-World Assumption)下的，当前题目的冲突点。逐步分析问题，判断答案是否正确，若正确返回：T，若错误返回:F。题目如下：
### 题目:
{qustion['problem']}

### 问题:
{qustion['question']}

### 答案：
{answer}
"""
    # print(prompt)
    return prompt


def origin_file_instruct_check(input_file_list, has_out_put):
    data_list = []
    # 按行读取数据
    for input_file in input_file_list:
        with open(input_file) as reader:
            for line in reader:
                sample = json.loads(line)
                data_list.append(sample)
                # break

    result_list = []
    for item in data_list:
        problem = item['problem']
        if "questions" in item.keys():

            questions = item['questions']
            for question in questions:
                answer = question['options'][char_to_number(question["answer"])]
                if has_out_put:
                    for option in question['options']:
                        local_item = {
                            "problem": problem,
                            "question": question["question"],
                            "answer": option
                        }

                        if answer == option:
                            local_item["output"] = "T"
                        else:
                            local_item["output"] = "F"
                        local_item["prompt"] = prompt_lora_step_check(local_item, answer)
                        result_list.append(local_item)

    return result_list


def inference_res_to_file(test_data, output_file, inference_res):
    data_list = []
    # 按行读取数据
    with open(test_data) as reader:
        for line in reader:
            sample = json.loads(line)
            id = sample['id']
            for sub_id, question in enumerate(sample['questions']):
                key = f"{id}-{sub_id}"
                if key in inference_res.keys():
                    question['answer'] = inference_res[key]
                else:
                    question['answer'] = 'A'
            data_list.append(sample)

    with open(output_file, 'w') as writer:
        for sample in data_list:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')


def evaluate_res_to_file(origin_file, inference_res):
    pse = 0
    cnt = 0
    tot = 0
    # 按行读取数据
    with open(origin_file) as reader:
        for line in reader:
            sample = json.loads(line)
            id = sample['id']
            for sub_id, question in enumerate(sample['questions']):
                real_answer = question['answer']
                key = f"{id}-{sub_id}"
                predict_answer = None
                if key in inference_res.keys():
                    predict_answer = inference_res[key]

                tot += 1
                cnt += real_answer == predict_answer
                if predict_answer is None:
                    pse += 1
    print(cnt, tot, cnt / tot, pse)


def down_model():
    # 模型下载
    from modelscope import snapshot_download
    # model_dir = snapshot_download('iic/gte_Qwen2-7B-instruct', cache_dir="models/")
    # model_dir = snapshot_download('iic/gte_Qwen2-7B-instruct', cache_dir="models/")
    # 模型下载
    from modelscope import snapshot_download

    # model_dir = snapshot_download('qwen/Qwen1.5-32B-Chat-GPTQ-Int4',
    #                               cache_dir="/Users/wuqing/PycharmProjects/202408-阿里云逻辑推理/models/")
    # model_dir = snapshot_download('qwen/qwen1.5-32b-chat-gptq-int4',
    #                               cache_dir="/Users/wuqing/PycharmProjects/202408-阿里云逻辑推理/models/")
    model_dir = snapshot_download('iic/nlp_gte_sentence-embedding_chinese-large',
                                  cache_dir="/Users/wuqing/PycharmProjects/202408-阿里云逻辑推理/models/")


import time


def tail(f):
    """Tail -f a file in python"""
    f.seek(0, 2)  # 移动到文件末尾
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.1)  # 等待一段时间，避免CPU占用过高
            continue
        yield line


def monitor_log(file_path, search_string):
    with open(file_path, 'r') as file:
        for line in tail(file):
            print(line)
            if search_string in line:
                print("推理服务启动完成，开始推理", end='')
                return


def wait_server_start():
    log_file_path = './start_server.log'
    search_string = 'Uvicorn running on'
    monitor_log(log_file_path, search_string)


def read_jsonl_file(input_files):
    data_list = []
    # 检查输入是否是列表，如果不是，则将其转换为列表
    if not isinstance(input_files, list):
        input_files = [input_files]

    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                sample = json.loads(line)
                data_list.append(sample)

    return data_list


def write_jsonl_file(data_list, output_file):
    with open(output_file, 'w', encoding="utf-8") as writer:
        for sample in data_list:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')


def hard_voting(data_files):
    all_data = [read_jsonl_file(file) for file in data_files]
    result = []

    num_rounds = len(all_data[0])

    for idx in range(num_rounds):
        current_problem_list = [data[idx] for data in all_data]
        round_result = {"id": current_problem_list[0]["id"], "questions": []}

        num_questions = len(current_problem_list[0]["questions"])

        for question_idx in range(num_questions):
            answers = [data["questions"][question_idx]["answer"] for data in current_problem_list]

            # 进行硬投票
            most_common_answer = Counter(answers).most_common(1)[0][0]
            round_result["questions"].append({"answer": most_common_answer})
        result.append(round_result)
    return result


import random


def shuffle_keep(original_list, keep_len=0.8):
    keep_count = int(len(original_list) * keep_len)
    random.shuffle(original_list)
    filtered_list = original_list[:keep_count]
    return filtered_list


def get_need_cal_id_list(res_dict_list: dict, num_throed):
    id_list = []
    for sub_id, answer_list in res_dict_list.items():
        most_common = Counter(answer_list).most_common(1)[0]
        most_common_answer = most_common[0]
        most_common_answer_time = most_common[1]
        if most_common_answer_time < (num_throed / 2):
            id_list.append(sub_id)
    return id_list