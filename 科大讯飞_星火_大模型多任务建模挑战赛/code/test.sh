# model_path = "./models/iic/gte_Qwen2-7B-instruct/"
# lora_out = "./models/output/gate_qwen_7b_epoch_2_add_data/"
# lora_out_checkpoint = "./models/output/gate_qwen_7b_epoch_2_add_data/checkpoint-88"

#!/bin/bash
# 启动服务并使用nohup确保即使终端关闭服务也会继续运行
nohup python -m vllm.entrypoints.openai.api_server \
    --model ./qwen2-72b-instruct-gptq-int4/  \
    --tensor-parallel-size 4 \
    --served-model-name qwen \
    --max-model-len=4096 \
    --enable-lora \
    --enforce-eager \
    --lora-modules person=./xinghuo_lora/person/checkpoint-48 en=./xinghuo_lora/en/checkpoint-100 sentiment=./xinghuo_lora/sentiment/checkpoint-100 > ./start_server.log 2>&1 &

python ./test.py