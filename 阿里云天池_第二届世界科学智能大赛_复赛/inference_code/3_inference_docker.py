import time

from loguru import logger

from util import most_frequent_char, get_need_cal_id_list

logger.remove()  # 移除默认的控制台输出
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip")

from util import origin_file_instruct, inference_detail, inference_res_to_file, wait_server_start, hard_voting, \
    write_jsonl_file


def inference_file_docker(inference_file, output_file, MODEL_NAME, vote_num, temperature, use_check=False):
    instruct_data = origin_file_instruct([inference_file], False)
    inference_res = inference_detail(instruct_data, MODEL_NAME=MODEL_NAME, local=True, vote_num=vote_num,
                                     temperature=temperature,
                                     use_check=use_check)
    inference_res_to_file(inference_file, output_file, inference_res)


def inference_file_docker_foreach_model(inference_file, output_file, MODEL_NAME_ALL, vote_num, temperature,
                                        use_check=False):
    instruct_data = origin_file_instruct([inference_file], False)

    result_file_list = []
    for model_name_sub in MODEL_NAME_ALL:
        start_time = time.time()
        inference_res = inference_detail(instruct_data, model_name_sub,
                                         local=True,
                                         vote_num=vote_num,
                                         use_check=use_check,
                                         temperature=temperature)
        result_file_local = f"./results_{model_name_sub}.jsonl"
        inference_res_to_file(inference_file, result_file_local, inference_res)
        result_file_list.append(result_file_local)
        end_time = time.time()  # 获取当前时间
        print(f"推理 {model_name_sub} 耗时：{end_time - start_time}秒")
    result_list = hard_voting(result_file_list)
    write_jsonl_file(result_list, output_file)


def inference_file_docker_foreach_model_double(inference_file, output_file, MODEL_NAME_ALL, vote_num, temperature,
                                               use_check=False):
    instruct_data = origin_file_instruct([inference_file], False)
    res_dict_list = {item["id"]: [] for item in instruct_data}

    for model_name_sub in MODEL_NAME_ALL:
        start_time = time.time()
        inference_res = inference_detail(instruct_data, model_name_sub,
                                         local=True,
                                         vote_num=vote_num,
                                         use_check=use_check,
                                         temperature=temperature)  # {current_id:answer}
        for sub_id, answer in inference_res.items():
            res_dict_list[sub_id].append(answer)
        end_time = time.time()  # 获取当前时间
        print(f"一次推理 {len(instruct_data)} : {model_name_sub} 耗时：{end_time - start_time}秒")

    need_id_list = get_need_cal_id_list(res_dict_list, len(MODEL_NAME_ALL))
    print(f"二次推理{len(need_id_list)}个问题")
    print(need_id_list)
    for model_name_sub in sorted(MODEL_NAME_ALL * 10):
        temperature = 0.9
        need_instruct_data = [item for item in instruct_data if str(item["id"]) in need_id_list]
        start_time = time.time()
        inference_res = inference_detail(need_instruct_data, model_name_sub,
                                         local=True,
                                         vote_num=vote_num,
                                         use_check=use_check,
                                         temperature=temperature)  # {current_id:answer}
        for sub_id, answer in inference_res.items():
            res_dict_list[sub_id].append(answer)
        end_time = time.time()  # 获取当前时间
        print(f"二次推理 {len(instruct_data)} ： {model_name_sub} 耗时：{end_time - start_time}秒")

    res_dict_item = {}
    for sub_id, answer_list in res_dict_list.items():
        res_dict_item[sub_id] = most_frequent_char(answer_list)

    inference_res_to_file(inference_file, output_file, res_dict_item)


def main():
    start_time = time.time()
    wait_server_start()
    print(f"推理开始")
    test_data = "/tcdata/round2_test_data.jsonl"
    result_file = "/app/results.jsonl"

    MODEL_NAME_ALL = [
        "lora1", "lora2", "lora3", "lora4",
        "lora11", "lora12", "lora13", "lora14",
        "lora24", "lora25", "lora26", "lora27", "lora28",
        "lora81", "lora82",
        "lora91", "lora92",
    ]

    vote_num = 1
    use_check = False
    temperature = 0.7
    inference_file_docker_foreach_model_double(test_data, result_file, MODEL_NAME_ALL, vote_num, temperature, use_check)

    end_time = time.time()  # 获取当前时间
    print("----------end---------")
    print(f"推理耗时：{end_time - start_time}秒")


if __name__ == '__main__':
    main()
