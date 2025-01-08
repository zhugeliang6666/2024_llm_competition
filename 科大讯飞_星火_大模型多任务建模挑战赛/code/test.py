import time

from xinghuo_util import origin_file_instruct, inference_detail, merge_result


def inference_file_remote(inference_file, model_name, output_file):
    instruct_data = origin_file_instruct([inference_file])
    inference_res = inference_detail(instruct_data, model_name, output_file, local=True)


def main():
    time.sleep(90)
    start_time = time.time()
    print(f"推理开始")
    test_data = "../xfdata/test_submit.csv"

    model_name = "sentiment"
    sentiment_result_file = f"../user_data/results_{model_name}.csv"
    inference_file_remote(test_data, model_name, sentiment_result_file)

    model_name = "person"
    person_result_file = f"../user_data/results_{model_name}.csv"
    inference_file_remote(test_data, model_name, person_result_file)

    model_name = "en"
    en_result_file = f"../user_data/results_{model_name}.csv"
    inference_file_remote(test_data, model_name, en_result_file)

    end_time = time.time()  # 获取当前时间
    output_file = "../prediction_result/results.csv"

    # 合并结果
    merge_result(test_data, person_file=person_result_file, sentiment_file=sentiment_result_file,
                 en_file=en_result_file, output_file=output_file)

    print(f"推理耗时：{end_time - start_time}秒")


if __name__ == '__main__':
    main()
