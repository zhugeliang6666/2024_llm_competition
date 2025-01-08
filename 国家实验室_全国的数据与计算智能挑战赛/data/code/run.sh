# 1.训练rag embedding部分，训练minincpm_embedding
python ./data/code/fineturne_minincpm_embedding/train_embedding_model.py
#拼接所有规则rule1.json和rule2.json
python ./data/code/p_data/p_data_rules.py
#2.训练rag reranker部分，训练bge_reranker
python ./data/code/finetune_bge_reranker/finetune_reranker_model.py

python ./data/code/finetune_bge_reranker/FlagEmbedding/scripts/hn_mine.py \
    --model_name_or_path "./data/user_data/bge-large-zh-v1.5" \
    --input_file ./data/user_data/finetune_bge_reranker/finetune_data.jsonl \
    --candidate_pool ./data/user_data/finetune_bge_rerankear/pretrain_data.jsonl \
    --output_file ./data/user_data/finetune_bge_reranker/finetune_data_minedHN.jsonl \
    --range_for_sampling 10-110 \
    --negative_number 25 

torchrun --nproc_per_node 1 -m FlagEmbedding.finetune.reranker.decoder_only.layerwise \
    --model_name_or_path ./data/user_data/bge-reranker-v2-minicpm-layerwise \
    --use_lora True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --save_merged_lora_model True \
    --model_type decoder \
    --model_type from_finetuned_model \
    --start_layer 8 \
    --head_multi True \
    --head_type simple \
    --trust_remote_code True \
    --train_data ./data/user_data/finetune_bge_reranker/finetune_data_minedHN.jsonl \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --knowledge_distillation False \
    --query_instruction_for_rerank 'A: ' \
    --query_instruction_format '{}{}' \
    --passage_instruction_for_rerank 'B: ' \
    --passage_instruction_format '{}{}' \
	--output_dir ./data/user_data/finetune_bge_reranker/test_decoder_only_base_bge-reranker-v2-minicpm-layerwise \
    --overwrite_output_dir \
    --learning_rate 3e-4 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ./data/code/finetune_bge_reranker/ds_stage0.json \
    --logging_steps 200 \
    --save_steps 200

cp ./data/user_data/bge-reranker-v2-minicpm-layerwise/modeling_minicpm_reranker.py ./data/user_data/finetune_bge_reranker/test_decoder_only_base_bge-reranker-v2-minicpm-layerwise/merged_model






#3.训练初赛模型，构建初赛测试集rag，得到初赛测试集的伪标签
#训练模型
python ./data/code/train_qwen2_5_7B/train_model.py --train_data ./data/raw_data/dev.json --output_dir ./data/user_data/chusai_model --logging_steps 200 --save_steps 200
#对初赛测试集进行tok30 embedding
python ./data/code/rag/test_rag.py --root_dir ./data/raw_data/test1.json --rules_data ./data/raw_data/rules1.json --model_path ./data/user_data/MiniCPM-Embedding --save_lora_steps ./data/user_data/fineturne_minincpm_embedding/miniCPMv3 --output_dir ./data/user_data/chusai_test_rules_tok30.pkl
#对初赛测试集进行minincpm和bge reanker排序,然后进行融合
python ./data/code/rag/rag_reranker.py --MiniCPM_Reranker_path ./data/user_data/MiniCPM-Reranker --root_dir ./data/raw_data/test1.json --rule_id_tok30 ./data/user_data/chusai_test_rules_tok30.pkl --all_rule_path ./data/raw_data/rules1.json --MiniCPM_test_rules_reranker_tok30_scores ./data/user_data/chusai_MiniCPM_test_rules_reranker_tok30_scores.pkl --MiniCPM_test_rules_reranker_new_tok30_2_tok10 ./data/user_data/chusai_MiniCPM_test_rules_reranker_new_tok30_2_tok10.pkl
python ./data/code/rag/new_reranker.py --root_dir ./data/raw_data/test1.json --rag_data_index ./data/user_data/chusai_test_rules_tok30.pkl --all_rule_path ./data/raw_data/rules1.json --model_path ./data/user_data/finetune_bge_reranker/test_decoder_only_base_bge-reranker-v2-minicpm-layerwise/merged_model --output_reranker_tok30_scores_path ./data/user_data/chusai_bge_test_rules_reranker_tok30_scores.pkl --output_reranker_tok30_2_tok10 ./data/user_data/chusai_bge_test_rules_reranker_new_tok30_2_tok10.pkl
python ./data/code/rag/avg_reranker_model.py --bge_reranker_scores ./data/user_data/chusai_bge_test_rules_reranker_tok30_scores.pkl --minicpm_reranker_scores ./data/user_data/chusai_MiniCPM_test_rules_reranker_tok30_scores.pkl --rag_data_index ./data/user_data/chusai_test_rules_tok30.pkl --root_dir ./data/raw_data/test1.json --bge_reranker_weight 0.7 --minicpm_reranker_weight 0.3 --ouput_dir ./data/user_data/chusai_finetune_bge+minincpm_tok30_2_tok10.pkl
#4.对初赛测试集进行预测
python ./data/code/vllm_pred/test_vllm.py --rag_data ./data/raw_data/rules1.json --data_path ./data/raw_data/test1.json --relu_id_tok10_path ./data/user_data/chusai_finetune_bge+minincpm_tok30_2_tok10.pkl --model_name_lora ./data/user_data/chusai_model --jsol_output_name ./data/user_data/chusai.json
#5.初赛训练集合并初赛测试集伪标签
python ./data/code/p_data/p_data.py --data1 ./data/raw_data/dev.json --data2 ./data/raw_data/test1.json --data2_sub ./data/user_data/chusai.json --output_data ./data/user_data/dev_fusai_train.json
# 6.训练训练集+初赛测试集，构建复赛训练集rag，得到复赛训练集伪标签
# 训练模型
python ./data/code/train_qwen2_5_7B/train_model.py --train_data ./data/user_data/dev_fusai_train.json --output_dir ./data/user_data/chusai_fusai_model --logging_steps 2000 --save_steps 2000
# #对复赛训练集集进行tok30 embedding
python ./data/code/rag/test_rag.py --root_dir ./data/raw_data/复赛新增训练参考集.json --rules_data ./data/user_data/all_rules_data.json --model_path ./data/user_data/MiniCPM-Embedding --save_lora_steps ./data/user_data/fineturne_minincpm_embedding/miniCPMv3 --output_dir ./data/user_data/chusai_fusai_test_rules_tok30.pkl
# #对复赛训练集集进行minincpm和bge reanker排序,然后进行融合
python ./data/code/rag/rag_reranker.py --MiniCPM_Reranker_path ./data/user_data/MiniCPM-Reranker --root_dir ./data/raw_data/复赛新增训练参考集.json --rule_id_tok30 ./data/user_data/chusai_fusai_test_rules_tok30.pkl --all_rule_path ./data/user_data/all_rules_data.json --MiniCPM_test_rules_reranker_tok30_scores ./data/user_data/chusai_fusai_MiniCPM_test_rules_reranker_tok30_scores.pkl --MiniCPM_test_rules_reranker_new_tok30_2_tok10 ./data/user_data/chusai_fusai_MiniCPM_test_rules_reranker_new_tok30_2_tok10.pkl
python ./data/code/rag/new_reranker.py --root_dir ./data/raw_data/复赛新增训练参考集.json --rag_data_index ./data/user_data/chusai_fusai_test_rules_tok30.pkl --all_rule_path ./data/user_data/all_rules_data.json --model_path ./data/user_data/finetune_bge_reranker/test_decoder_only_base_bge-reranker-v2-minicpm-layerwise/merged_model --output_reranker_tok30_scores_path ./data/user_data/chusai_fusai_bge_test_rules_reranker_tok30_scores.pkl --output_reranker_tok30_2_tok10 ./data/user_data/chusai_fusai_bge_test_rules_reranker_new_tok30_2_tok10.pkl
python ./data/code/rag/avg_reranker_model.py --bge_reranker_scores ./data/user_data/chusai_fusai_MiniCPM_test_rules_reranker_tok30_scores.pkl --rag_data_index ./data/user_data/chusai_fusai_test_rules_tok30.pkl --root_dir ./data/raw_data/复赛新增训练参考集.json --bge_reranker_weight 0.7 --minicpm_reranker_weight 0.3 --ouput_dir ./data/user_data/chusai_fusai_finetune_bge+minincpm_tok30_2_tok10.pkl
# #7.对复赛训练集进行预测
python ./data/code/vllm_pred/test_vllm.py --rag_data ./data/user_data/all_rules_data.json --data_path ./data/raw_data/复赛新增训练参考集.json --relu_id_tok10_path ./data/user_data/chusai_fusai_finetune_bge+minincpm_tok30_2_tok10.pkl --model_name_lora ./data/user_data/chusai_fusai_model --jsol_output_name ./data/user_data/fusai.json
#8.拼接训练集+初赛测试集+复赛训练集伪标签
python ./data/code/p_data/p_data.py --data1 ./data/user_data/dev_fusai_train.json --data2 ./data/raw_data/复赛新增训练参考集.json --data2_sub ./data/user_data/fusai.json --output_data ./data/user_data/all_train.json
#9.训练训练集+初赛测试集+复赛训练集伪标签，构建复赛测赛集rag，预测复赛测试集
#训练模型
python ./data/code/train_qwen2_5_7B/train_model.py --train_data ./data/user_data/all_train.json --output_dir ./data/user_data/final_model --logging_steps 2000 --save_steps 2000
#对复赛测试集进行tok30 embedding
python ./data/code/rag/test_rag.py --root_dir ./data/raw_data/test.json --rules_data ./data/user_data/all_rules_data.json --model_path ./data/user_data/MiniCPM-Embedding --save_lora_steps ./data/user_data/fineturne_minincpm_embedding/miniCPMv3 --output_dir ./data/user_data/final_test_rules_tok30.pkl
# 对复赛测试集进行minincpm和bge reanker排序,然后进行融合
python ./data/code/rag/rag_reranker.py --MiniCPM_Reranker_path ./data/user_data/MiniCPM-Reranker --root_dir ./data/raw_data/复赛新增训练参考集.json --rule_id_tok30 ./data/user_data/final_test_rules_tok30.pkl --all_rule_path ./data/user_data/all_rules_data.json --MiniCPM_test_rules_reranker_tok30_scores ./data/user_data/final_MiniCPM_test_rules_reranker_tok30_scores.pkl --MiniCPM_test_rules_reranker_new_tok30_2_tok10 ./data/user_data/final_MiniCPM_test_rules_reranker_new_tok30_2_tok10.pkl
python ./data/code/rag/new_reranker.py --root_dir ./data/raw_data/test.json --rag_data_index ./data/user_data/final_test_rules_tok30.pkl --all_rule_path ./data/user_data/all_rules_data.json --model_path ./data/user_data/finetune_bge_reranker/test_decoder_only_base_bge-reranker-v2-minicpm-layerwise/merged_model --output_reranker_tok30_scores_path ./data/user_data/final_bge_test_rules_reranker_tok30_scores.pkl --output_reranker_tok30_2_tok10 ./data/user_data/final_bge_test_rules_reranker_new_tok30_2_tok10.pkl
python ./data/code/rag/avg_reranker_model.py --bge_reranker_scores ./data/user_data/final_MiniCPM_test_rules_reranker_tok30_scores.pkl --minicpm_reranker_scores ./data/user_data/final_MiniCPM_test_rules_reranker_tok30_scores.pkl --rag_data_index ./data/user_data/final_test_rules_tok30.pkl --root_dir ./data/raw_data/test.json --bge_reranker_weight 0.8 --minicpm_reranker_weight 0.2 --ouput_dir ./data/user_data/final_finetune_bge+minincpm_tok30_2_tok10.pkl
#10.对复赛测试集进行预测
python ./data/code/vllm_pred/test_vllm.py --rag_data ./data/user_data/all_rules_data.json --data_path ./data/raw_data/test.json --relu_id_tok10_path ./data/user_data/final_finetune_bge+minincpm_tok30_2_tok10.pkl --model_name_lora ./data/user_data/final_model --jsol_output_name ./data/prediction_result/result.json