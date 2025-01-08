python finetune_reranker_model.py

python FlagEmbedding/scripts/hn_mine.py \
    --model_name_or_path "./data/user_data/bge-large-zh-v1.5" \
    --input_file ./data/user_data/finetune_bge_reranker/finetune_data.jsonl \
    --candidate_pool ./data/user_data/finetune_bge_reranker/pretrain_data.jsonl \
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