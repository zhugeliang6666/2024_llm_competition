from http.cookies import SimpleCookie

import httpx

inference_url = 'https://ctr.oceanus.qihoo.net/MTEuNy40OC4xMTY6ODA3MQ/user/j-hewuqing-jk/vscode/proxy/8000/v1'
cookie_str = "_xsrf=2|8a2a9ed1|8e3fbbb6c94c67721ce226c0ab46c55f|1725013383; jupyterhub-user-j-hewuqing-jk=2|1:0|10:1726464708|29:jupyterhub-user-j-hewuqing-jk|40:Q1l4eDMzQWdETURYc1BKeWVwdGxkWEJDYmpvdlNN|8a696e157768f4455e7c62ba4a8a70011c2d6973afc68269c40776b287a48720; jupyterhub-session-id=828e6ce5a49f4ed0bdd5d2e83ac87adb; __guid=122821401.771803812403085600.1721893236936.395; yiduoyun_hide_resource=JTdCJTIyai1oZXd1cWluZy1qayUyMiUzQWZhbHNlJTdE; _gcl_au=1.1.1858230484.1724055741; HULK_TD=t%3D1725353522%26s%3Dd8e9b576945fa938ff4445f679f64c1a%26a%3D1; HULK_QD=u%3Dw-urjhdvat-wx%26m%3Durjhdvat-wx%2540360fuhxr.pbz%26d%3D%25E4%25BD%2595%25E6%25AD%25A6%25E5%25BA%2586%26s%3D; zyun_ticket=OT-8a6728cb-fb34-48ce-8b21-e1751fe5360f; yiduoyun_resource=ai1oZXd1cWluZy1qayUzQjczNDQ3NTAxOTczNTc5MzY2NCUyQ3VuZGVmaW5lZCUyQzAlM0IxMDQxNTQwNTg4NTUwNjk2OTYxJTJDMzcyNiUyQzY1YTc0YjIwNzVhNTk5MTdjYzg2ODMxMCUyQ2FpX2Rldl9wbGF0Zm9ybQ==; apt.uid=AP-YFGMCGUNNIFB-2-1726219737449-79337382.0.2.6b933aba-006b-4c12-b4b5-a0ddc2adfcf4"
cookie = SimpleCookie(cookie_str)
cookies_dict = {key: morsel.value for key, morsel in cookie.items()}
http_client = httpx.Client(cookies=cookies_dict)

train_file = "../data/round1_train_data.jsonl"
test_file_with_answer_gpt = "../data/chusai_gpt4_finish.jsonl"
test_file_with_answer_mannul = "../data/new_data.jsonl"
round1_train_250 = "../data/round1_train_250.jsonl"
add_train_10 = "../data/round1_train_10.jsonl"
add_train_100 = "../data/round1_train_100.jsonl"
add_train_1000 = "../data/round1_train_10000.jsonl"
add_train_25000 = "../data/LogiQuest_25000.jsonl"

train_file_list = [train_file, test_file_with_answer_gpt, add_train_100]
# train_file_list = [add_train_10]

model_path = "./models/qwen/Qwen1___5-32B-Chat-GPTQ-Int4/"
lora_out = "./lora/qwen_32b_int4_epoch_2/"
lora_out_checkpoint = "./models/lora/qwen_32b_int4_epoch_2_add_data/checkpoint-6372"

per_device_train_batch_size = 2
gradient_accumulation_steps = 4
num_train_epochs = 2

check_mode = "check2"
# top_p = 0.9
