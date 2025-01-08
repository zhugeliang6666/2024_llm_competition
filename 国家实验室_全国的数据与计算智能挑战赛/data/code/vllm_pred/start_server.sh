nohup python -m vllm.entrypoints.openai.api_server \
    --model ./data/user_data/qwen2_5/Qwen2.5-7B-Instruct  \
    --served-model-name qwen \
    --max-model-len=3840 \
    --enable-lora \
    --enforce_eager \
    --lora-modules lora1=/home/un/桌面/QC/2024_全国大数据智能大赛/new_复赛_code/train_model/72b+o1_model2/qwen2_5_7b_lora/checkpoint-20000 > ./start_server.log 2>&1 &

python vllm_api.py --model_name lora1 --jsol_output_name model1.jsonl --start_vllm True
python vllm_api.py --model_name lora1 --jsol_output_name model2.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model3.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model4.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model5.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model6.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model7.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model8.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model9.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model10.jsonl --start_vllm False
python vllm_api.py --model_name lora1 --jsol_output_name model11.jsonl --start_vllm False