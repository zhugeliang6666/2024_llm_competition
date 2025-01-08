from http.cookies import SimpleCookie

import httpx

inference_url = ''
cookie_str = ""
cookie = SimpleCookie(cookie_str)
cookies_dict = {key: morsel.value for key, morsel in cookie.items()}
http_client = httpx.Client(cookies=cookies_dict)

MODEL_NAME = "sql-lora"
vote_num = 1

train_file = "data/round1_train_data.jsonl"
test_file_with_answer_gpt = "data/chusai_gpt4_finish.jsonl"
add_train_100 = "./data/round1_train_100.jsonl"

train_file_list = [train_file,test_file_with_answer_gpt,add_train_100]

model_path = "./models/qwen/Qwen2___5-32B-Instruct-GPTQ-Int4"
lora_out = "./lora_data_shuffle_qwen_2_origin/"

per_device_train_batch_size = 1
gradient_accumulation_steps = 1
num_train_epochs = 2

check_mode = "check2"
