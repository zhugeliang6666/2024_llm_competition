# 项目说明
## 1.硬件配置
- 系统：Ubuntu22
- python版本：3.10.14
- cuda版本：12.1
- torch版本：2.3.1+cuda12.1
- transformers版本： 4.43.4
- auto_gptq :0.7.1
## 2.环境依赖
### 完整环境见requirements.txt
## 3.文件和代码结构
- data 
  - external_data
- src  
    - models 存放大模型预训练权重
        - qwen1___5-32b-chat-gptq-int4
        - outputs 存放Lora微调后的检查点
    - 1_data_process.py
    - 2_train_lora.py 
    - 3_inference.py
- submit
- run.sh
## 4.方法介绍
### 4.1.数据预处理
数据预处理由一下3步构成：
- 所以我们使用api调用qwen-max，生成了新的数据，使用“相似的问题”，“相似的子问题”两种prompt生成数据。 文件名是：round1_train_data_similar_problem-qwen-max.jsonl，round1_train_data_similar_sub_problem-qwen-max.jsonl
- 但是生成的数据问然存在明显的逻辑问题，所以我们去掉存在明显问题的数据。
- 每条记录包含多个子问题，我们对数据集进行重新处理，以便将每个子问题分割成独立的数据条目。
### 4.2.模型选择与训练
模型选择方面，我们使用了：wen1.5-32b-chat-gptq-int4
```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/qwen1.5-32b-chat-gptq-int4')
```
使用了比赛方的500条训练数据，外加拓展得到的数据496+496条数据。
使用Lora微调的方式对qwen1.5-32b-chat-gptq-int4进行微调训练，对qwen1.5-32b-chat-gptq-int4的"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"进行微调，
max_len为1800，学习率1e-4,epoch为2，训练显存超过了32G，我们使用2张v100 进行训练。

### 4.3.推理
由于使用auto_gptq，我们没有找到lora权重和开源模型merge的方式。故使用：model.generate(**inputs, **gen_kwargs) 进行推理，推理速度较慢，没有使用多路投票的方式。
- 单张v100，vote_num = 1 ，推理耗时在1小时50分钟左右。得分0.8040左右。
- 单张v100，vote_num = 3 ，推理耗时在5小时左右。得分0.8100左右。
## 5.项目运行步骤
使用sh脚本运行整个项目
```python
sh run.sh
```
运行流程为：
1. 运行src/1_data_process.py 生成新的数据（生成的flag是false，不进行新的数据生成了），直接进行对训练集和测试集进行子问题拆分，得到新的数据集。round1_train_data_instruction_v3.json，round1_test_data_instruction.json
2. 运行src/2_train_lora.py 对qwen1.5-32b-chat-gptq-int4进行lora微调，得到lora后的权重在src/models/outputs/Qwen2_32b_int4_instruct_v3_lora_epoch_3
3. 运行src/3_inference.py 加载qwen1.5-32b-chat-gptq-int4和lora文件，推理结果，保存结果文件在submit中。