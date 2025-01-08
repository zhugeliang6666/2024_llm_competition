# nohup accelerate launch --config_file ./config/2_gpu.ymal xinghuo_train_accelerate.py en > ./train.log 2>&1 &
# accelerate launch --config_file ./4_gpu.ymal xinghuo_train_accelerate.py en > ./train.log
# accelerate launch --config_file ./4_gpu.ymal xinghuo_train_accelerate.py sentiment > ./train.log
# accelerate launch --config_file ./4_gpu.ymal xinghuo_train_accelerate.py person > ./train.log
# sh -x ./start_vllm.sh
# accelerate launch --config_file ./config/2_gpu.ymal xinghuo_train_accelerate.py en > ./train.log

export TORCH_USE_CUDA_DSA=1
python ./train.py sentiment > ./train.log
python ./train.py person > ./train.log
python ./train.py en > ./train.log