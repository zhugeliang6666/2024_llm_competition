from http.cookies import SimpleCookie

import httpx

inference_url = 'https://ctr.oceanus.qihoo.net/MTEuNy40OC4xNDI6ODA3MQ/user/j-hewuqing-jk/vscode/proxy/8000/v1'
cookie_str = "_xsrf=2|b7c4d96a|0ac9b9630d62a9b08c406c04ee016e34|1725861352; jupyterhub-user-j-hewuqing-jk=2|1:0|10:1726131426|29:jupyterhub-user-j-hewuqing-jk|40:UmtvZ3g5bmZCUmhSMWZBRG1LSzZrRWRCUGR3cW4z|3df7fa5d6fcbb0a8fd039596e7e472d64be9488d41a449c56ed69924f96d33b8; jupyterhub-session-id=63a67a4fe6d64c46b7a5470238ae08c8; __guid=122821401.771803812403085600.1721893236936.395; yiduoyun_hide_resource=JTdCJTIyai1oZXd1cWluZy1qayUyMiUzQWZhbHNlJTdE; _gcl_au=1.1.1858230484.1724055741; HULK_TD=t%3D1725353522%26s%3Dd8e9b576945fa938ff4445f679f64c1a%26a%3D1; HULK_QD=u%3Dw-urjhdvat-wx%26m%3Durjhdvat-wx%2540360fuhxr.pbz%26d%3D%25E4%25BD%2595%25E6%25AD%25A6%25E5%25BA%2586%26s%3D; apt.uid=AP-YFGMCGUNNIFB-2-1725354942109-69723971.0.2.a2e3917a-4f22-4c24-8671-436b74b600a2; zyun_ticket=OT-8a6728cb-fb34-48ce-8b21-e1751fe5360f; yiduoyun_resource=ai1oZXd1cWluZy1qayUzQjczNDQ3NTAxOTczNTc5MzY2NCUyQ3VuZGVmaW5lZCUyQzAlM0IxMDQxNTQwNTg4NTUwNjk2OTYxJTJDMzcyNiUyQzY1YTc0YjIwNzVhNTk5MTdjYzg2ODMxMCUyQ2FpX2Rldl9wbGF0Zm9ybQ==; mailMd5=dab37127c3a8929759db33022eb011a343b7766b"
cookie = SimpleCookie(cookie_str)
cookies_dict = {key: morsel.value for key, morsel in cookie.items()}
http_client = httpx.Client(cookies=cookies_dict, timeout=10000000)

train_file = "../xfdata/train.csv"
train_file_list = [train_file]

model_path = "./models/qwen2-72b-instruct-gptq-int4/"
lora_out = "./xinghuo_lora/"

per_device_train_batch_size = 2
gradient_accumulation_steps = 1
num_train_epochs = 2
