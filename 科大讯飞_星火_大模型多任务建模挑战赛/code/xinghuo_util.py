import time

import httpx
import pandas as pd
from loguru import logger
from openai import OpenAI

from xinghuo_config import inference_url, http_client
import re

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import json
import time
from tqdm import tqdm


# content	person	sentiment	en
def read_file(file):
    if isinstance(file, str):
        file = [file]
    json_dict_list = []
    for file_one in file:
        data = pd.read_csv(file_one, sep="\t", encoding='utf-8')
        json_str = data.to_json(orient='records', force_ascii=False)
        local_list = pd.read_json(json_str, orient='records', typ='list').tolist()
        for local in local_list:
            if "person" in local.keys() and local['person'] is None:
                local['person'] = ""
            json_dict_list.append(local)
    return json_dict_list


def write_file(data_list, output_file):
    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False, sep='\t', encoding='utf-8', mode='w')
    print(f"Data has been written to {output_file}")


def gen_prompt(item):
    prompt_person = f""""作为信息提取专家，你的工作是识别文本中提到的所有人物姓名。请阅读以下文本，并从中找出所有被提及的人物姓名。

{item['content']}

只提取所有的具体人物姓名，用逗号分隔。若无法提取出任何人物姓名，返回空字符串""即可。
### 符合中文的人物姓名的规则，或者符合英文的人物姓名的规则，比如：黄某,黄甫章,亚当·里奇,里奇,彼得斯,德克,史蒂文森,德克-诺维茨基,诺维茨基,阿尔德里奇,张海彦,泰森-钱德勒,拜纳姆,加索尔,周婆婆,李爹爹,彭宇,王琦,许云鹤
### 排除代词：不提取“我”、“你”等代词。
### 排除亲属称谓：不提取“父亲”、“孩子”、“妻子”、“儿子”等亲属称谓。
### 排除职业和职务：不提取“工程师”、“CEO”、“主席”、“警方”等职业和职务名称。
### 需要排除公司名称，品牌名称,组织名称，比如:湖北省招办,中国移动,WAC联盟,新浪体育讯,76人,华夏,Netflix,威孚高科,工行,央视,兴业基金,海尔,中国男篮,华商策略精选,微博,执政党,金螳螂,中恒集团,中环股份,掌趣科技,Hulu,A股 等等。
### 排除地名：不提取“纽约”、“新西兰”等地名。
"""

    prompt_sentiment = f"""你是一个情感分析专家，专门用于从文本中识别和分类情感倾向。你的任务是阅读以下文本，并确定它表达的正向或者负向是的情感。请仔细阅读文本，并准确判断其情感倾向。

{item["content"]}

请分析文本的情感倾向，并将其分类为正向或者负向。"""

    prompt_en = f"""作为一位资深翻译专家，你的职责是将以下中文文本翻译成英文。请确保你的翻译不仅准确传达原文的意义，而且语言流畅自然，尽可能保留原文的语气和风格。翻译完成后，我们将使用BLEU评分系统来评估翻译的质量。

原文：
{item['content']}

请开始翻译，将文本从中文翻译成英文。"""
    return prompt_person, prompt_sentiment, prompt_en


def origin_file_instruct(file):
    data = read_file(file)
    for item in data:
        prompt_person, prompt_sentiment, prompt_en = gen_prompt(item)
        item['prompt_person'] = prompt_person
        item['prompt_sentiment'] = prompt_sentiment
        item['prompt_en'] = prompt_en

    return data


def api_retry(MODEL_NAME, query, local: bool):
    max_retries = 5
    retry_delay = 10  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            if local:
                return call_api_local(MODEL_NAME, query)
            else:
                return call_api_remote(MODEL_NAME, query)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed for text:{query}. Error: {e}")
                raise


def call_api_local(MODEL_NAME, query):
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
        # temperature=temperature
    )
    return completion.choices[0].message.content


def call_api_remote(MODEL_NAME, query):
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
        # temperature=temperature
    )
    return completion.choices[0].message.content


def inference_detail(datas, model_name, output_file, local):
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_data = {}
        # 送入多线程任务
        for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
            content = data['content']
            field_name = f"prompt_{model_name}"
            future3 = executor.submit(api_retry, model_name, data[field_name], local)
            future_data[future3] = (content, model_name)

        res_has_dict = {}
        for future in tqdm(as_completed(future_data), total=len(future_data.keys()), desc="Processing tasks"):
            content = future_data[future][0]
            task_type = future_data[future][1]
            try:
                res = future.result()

                if content in res_has_dict.keys():
                    res_has_dict[content][task_type] = res
                else:
                    res_has_dict[content] = {}
                    res_has_dict[content]['content'] = content
                    res_has_dict[content][task_type] = res

                print(res_has_dict[content][task_type])
            except Exception as e:
                logger.error(f"Failed to process text: {data}. Error: {e}")

        res_list = []
        for item in res_has_dict.values():
            res_list.append(item)
    write_file(res_list, output_file)


def merge_result(test_file, person_file, sentiment_file, en_file, output_file):
    test = read_file(test_file)

    person = read_file(person_file)
    sentiment = read_file(sentiment_file)
    en = read_file(en_file)

    person_dict = {}
    sentiment_dict = {}
    en_dict = {}
    for person_item in person:
        res = None
        if person_item["person"]:
            local_list = list(set(person_item["person"].split(",")))
            if len(local_list) == 1:
                res = local_list[0]
            else:
                res = ",".join(local_list)
        if res:
            res = res.lstrip(",")
        if res == "无":
            res = None
        person_dict[person_item["content"]] = res

    # 处理sentiment数据
    for sentiment_item in sentiment:
        sentiment_dict[sentiment_item['content']] = sentiment_item['sentiment']

    en_dict = {item['content']: item['en'] for item in en}
    for test_iem in test:
        key = test_iem["content"]
        # person	sentiment	en
        test_iem["person"] = person_dict[key]
        test_iem["sentiment"] = sentiment_dict[key]
        test_iem["en"] = en_dict[key]

    write_file(test, output_file)
