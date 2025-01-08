cd ./inference_code
#!/bin/bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model /lib/model/  \
    --served-model-name qwen \
    --max-model-len=4096 \
    --enable-lora \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --lora-modules lora1=/lib/lora1/ lora2=/lib/lora2/ lora3=/lib/lora3/ > ./start_server.log 2>&1 &

python 3_inference_docker.py