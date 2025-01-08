import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
import dashscope
from http import HTTPStatus

from tqdm import tqdm


def api_retry(MODEL_NAME, query):
    max_retries = 5
    retry_delay = 60  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            return call_qwen_api(MODEL_NAME, query)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                logger.warning(f"Attempt {attempts} failed for text: {query}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed for text: {query}. Error: {e}")
                raise


def call_qwen_api(MODEL_NAME, query):
    # 这里采用dashscope的api调用模型推理，通过http传输的json封装返回结果
    messages = [
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        MODEL_NAME,
        messages=messages,
        result_format='message',  # set the result is message format.
    )
    if response.status_code == HTTPStatus.OK:
        # print(response)
        return response['output']['choices'][0]['message']['content']
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        raise Exception()


def process_datas(query_list, MODEL_NAME):
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_data = {}
        lens = 0
        # 送入多线程任务
        for query in tqdm(query_list, desc="Submitting tasks", total=len(query_list)):
            future = executor.submit(api_retry, MODEL_NAME, query)
            future_data[future] = query
            time.sleep(0.6)  # 控制每0.6秒提交一个任务
            lens += 1
        # 处理多线程任务
        for future in tqdm(as_completed(future_data), total=lens, desc="Processing tasks"):
            query = future_data[future]
            try:
                res = future.result()
                try:
                    json.loads(res)
                    results.append(res)
                except Exception as e:
                    logger.error(f"error query:{query} and error responce:{res}")
                    logger.error(f"error reason:{e}")
            except Exception as e:
                logger.error(f"Failed to process text: {query}. Error: {e}")

    return results


def deal_process_common(input_file, output_file, prompt, limit=1000000):
    query_list = []
    # 按行读取数据
    with open(input_file) as reader:
        cnt = 0
        for line in reader:
            cnt += 1
            query_list.append(prompt + line)
            if cnt == limit:
                break

    data_list = process_datas(query_list, MODEL_NAME)

    with open(output_file, 'w', encoding="utf-8") as writer:
        for sample in data_list:
            writer.write(sample)
            writer.write('\n')


def deal_process_similar_problem(input_file, output_file):
    instruction = "你是一个逻辑推理问题出题专家，参考下方json再生成类似的json问题，必须有一个problem，questions列表及answer."
    prompt = f"{instruction}{request}{end}"
    deal_process_common(input_file, output_file, prompt)


def deal_process_similar_sub_problem(input_file, output_file):
    instruction = "你是一个逻辑推理问题出题专家，以下json是一个problem，对应一些子问题。生成三个其他子问题。problem的内容必须完整包含在返回值中。"
    prompt = f"{instruction}{request}{end}"
    deal_process_common(input_file, output_file, prompt)


def deal_process(input_file_list, output_file, hasOutPut):
    data_list = []
    # 按行读取数据
    for input_file in input_file_list:
        with open(input_file) as reader:
            for line in reader:
                sample = json.loads(line)
                data_list.append(sample)
            # break
    # data_list = sorted(data_list, key=lambda x: int(str(x['id'])[-3:]))
    # data_list = sorted(data_list, key=lambda x: int(str(x['id'])
    #                                                 .replace("round1_train_data_", "")
    #                                                 .replace("round1_flight_data_", "")))
    common_instruction = "你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为\"答案是：A\"。"

    # 你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，包含了 problem，questions和answer。你需要给出类型的题目，按照题目格式返回 ：{"problem": "考虑一个计算累加到给定正整数的和的逻辑问题。累加从1开始，一直到这个正整数为止。比如，如果给定数字2，那么累加就是1+2；如果给定数字3，则累加是1+2+3，以此类推。\n\n根据上述逻辑，回答以下选择题：", "questions": [{"question": "选择题 1：\n将累加和计算到8时，得到的结果是多少？", "options": ["28", "36", "45", "55"], "answer": "B"}, {"question": "选择题 2：\n如果一个计算累加和到8得到的结果是7，这种说法正确吗？", "options": ["正确", "错误"], "answer": "B"}, {"question": "选择题 3：\n如果一个计算累加和到8得到的结果是5，这种说法正确吗？", "options": ["正确", "错误"], "answer": "B"}, {"question": "选择题 4：\n如果一个计算累加和到8得到的结果是1，这种说法正确吗？", "options": ["正确", "错误"], "answer": "B"}], "id": "round1_train_data_388"}
    # {"problem": "考虑一个计算累加到给定正整数的和的逻辑问题。累加从1开始，一直到这个正整数为止。比如，如果给定数字2，那么累加就是1+2；如果给定数字3，则累加是1+2+3，以此类推。\n\n根据上述逻辑，回答以下选择题：", "questions": [{"question": "选择题 1：\n将累加和计算到8时，得到的结果是多少？", "options": ["28", "36", "45", "55"], "answer": "B"}, {"question": "选择题 2：\n如果一个计算累加和到8得到的结果是7，这种说法正确吗？", "options": ["正确", "错误"], "answer": "B"}, {"question": "选择题 3：\n如果一个计算累加和到8得到的结果是5，这种说法正确吗？", "options": ["正确", "错误"], "answer": "B"}, {"question": "选择题 4：\n如果一个计算累加和到8得到的结果是1，这种说法正确吗？", "options": ["正确", "错误"], "answer": "B"}], "id": "round1_train_data_388"}
    result_list = []
    for item in data_list:
        problem = item['problem']
        if "questions" in item.keys():
            try:
                questions = item['questions']
                id = item['id']
                for question in questions:
                    options_str = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(question['options']))
                    item = {
                        "instruction": f"{common_instruction}题目如下:\n题目:{problem}",
                        "input": f"回答以下选择题:\n{question['question']}\n{options_str}",
                    }
                    if hasOutPut:
                        item["output"] = question["answer"]

                        # answer等于G的直接排除
                        if question["answer"] == "E" and len(question["options"]) < 5:
                            logger.error(f"skip instruct: {item}")
                            continue

                        if question["answer"] == "F" and len(question["options"]) < 6:
                            logger.error(f"skip instruct: {item}")
                            continue

                        if question["answer"] == "G" and len(question["options"]) < 7:
                            logger.error(f"skip instruct: {item}")
                            continue

                    item["id"] = id
                    result_list.append(item)
            except Exception as e:
                logger.error(f"Failed to process text: {item}. Error: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

    return result_list


def api_gen_problem(input_file, external_data_base_path, is_gen_new_problem: bool):
    # 生成相似的子问题
    similar_sub_problem = external_data_base_path + os.path.basename(
        input_file.replace(".jsonl", f"_similar_sub_problem-{MODEL_NAME}.jsonl"))

    # 相似的问题
    similar_problem = external_data_base_path + os.path.basename(
        input_file.replace(".jsonl", f"_similar_problem-{MODEL_NAME}.jsonl"))

    if is_gen_new_problem:
        deal_process_similar_problem(input_file, similar_problem)
        deal_process_similar_sub_problem(input_file, similar_sub_problem)

    return input_file, similar_problem, similar_sub_problem


MODEL_NAME = "qwen-max"
dashscope.api_key = "xxxxxx"
request = "返回值满足所有要求:返回值必须包含一个problem，questions列表及answer；返回值必须不包含```json 和 ```；返回值必须保持格式与参考json一致；返回值必须是一行json数据；返回值中的questions列表中的必须是单选题；返回值的answer必须是对应question的正确answer；返回值的answer的值必须是ABCDEFG的单个字母,A代表当前options的第一个选项，B代表当前options的第二个选项，C代表当前options的第三个选项，D代表当前options的第四个选项，E代表当前options的第五个选项，F代表当前options的第六个选项，G代表当前options的第七个选项，answer的值必须代表当前options中的一个存在的选项，answer的值必须代表当前question的正确解答"
end = "你做的非常好。参考json:"

if __name__ == '__main__':
    # step1 train_data + 相似的问题+相似的子问题
    input_file = "../data/round1_train_data.jsonl"
    external_data_base_path = "../data/external_data/"
    is_gen_new_problem = False
    input_file, similar_problem, similar_sub_problem = api_gen_problem(input_file,
                                                                       external_data_base_path,
                                                                       is_gen_new_problem)
    input_file = [
        input_file, similar_problem, similar_sub_problem
    ]

    # step2  转换成 instruction+input+output+id
    instruction_train_file = external_data_base_path + os.path.basename(
        input_file[0].replace(".jsonl", "_instruction_v3.json"))
    deal_process(input_file, instruction_train_file, hasOutPut=True)

    # step4  转换成 instruction+input+id
    input_file = ["../data/round1_test_data.jsonl"]
    output_file = external_data_base_path + os.path.basename(input_file[0].replace(".jsonl", "_instruction.json"))
    deal_process(input_file, output_file, hasOutPut=False)
