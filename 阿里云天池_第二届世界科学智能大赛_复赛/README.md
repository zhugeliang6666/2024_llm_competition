# 项目说明
## 1.硬件配置
- 系统：Ubuntu22
- python版本：3.10.14
- cuda版本：12.1
- torch版本：2.3.1+cuda12.1
- transformers版本： 4.43.4
- auto_gptq :0.7.1
- vllm: 0.5.5
- accelerate==0.33.0
## 2.环境依赖
### 完整环境见requirements.txt
## 3.training_code文件结构
- config
  - 2_gpu_config -- accelerate 配置文件
- data 
  - external_data -- 训练数据
- src  
    - models
        - Qwen2.5-32B-Instruct-GPTQ-Int4
    - 2_train_final_accelerate.py -- 训练代码
    - lora_config.py
    - util.py 
    - run_train.sh -- 训练脚本

## 4.方法介绍
整体方案沿用我们初赛的方案（初赛方案：复杂推理能力评估赛道_DT_Juice_81_18565770177.zip），存在4个改进点：
1. 模型更强大：qwen1___5-32b-chat-gptq-int4 修改为了 Qwen2.5-32B-Instruct-GPTQ-Int4 
2. 数据修改：调用qwen-max 衍生的数据 修改为了 2份开源的数据。
3. 数据shuffle：对于数据进行了shuffle 并shuffle 之后去80%的数据，进行一次微调
3. 推理方式改进：推理model.generate 的方式 替换为了 推理服务vllm ，并进行多次投票，
4. 二次投票：对第一轮投票结果，没有过半数的问题进行了二次投票。
### 4.1.数据预处理
最后训练数据数据由3部分组成：(见training_code/data/)，
- round1_train_data.jsonl
- chusai_gpt4_finish.jsonl ，来自开源数据（https://modelscope.cn/datasets/q2386908104/Test_Set_Answers/files） 和
- round1_train_100.jsonl，来自开源数据（https://modelscope.cn/datasets/chenr1209/Logical_reasoning_choice/files） 中LogiQuest_25000.jsonl 的前100条
每条记录包含多个子问题，我们对数据集进行重新处理，以便将每个子问题分割成独立的数据条目。
- 然后 数据进行了shuffle ，并shuffle之后取80%的数据，进行一次微调。
### 4.2.模型选择与训练
模型选择方面，我们使用了：Qwen2.5-32B-Instruct-GPTQ-Int4
```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4')
```
使用了以上3份数据进行了抽取 其中80% 的数据进行训练。每次训练透随机抽取。
使用Lora微调的方式对Qwen2.5-32B-Instruct-GPTQ-Int4进行微调训练，对qwen1.5-32b-chat-gptq-int4的"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"进行微调，
max_len为1800，学习率1e-4,epoch为2，训练显存超过了32G，我们使用2张v100 进行训练。
- 注意：我们尝试了batch size = 1，2，，4，8，但是 batch size = 1 单模效果最好。 
- 注意：round1_train_data 中存在16道题错误，但是我们并没有修复。修复版本vs 没有修复版本：效果反而不好，怀疑数据质量。

### 4.3.推理
我们使用vllm，--lora-modules 的方式进行部署推理服务。
- 优化了推理数据，使用http client 进行多线程访问 推理服务：推理线程数量可以达到64。
- 一次预测，约在6～8min左右。
- 反复随机取数据，训练同一个模型。 进行15，17，19，21，25 次投票。 最后结果是21投票最好。

## 5.项目运行步骤
使用sh 进行训练。
```python
run_train.sh
```

## 6.尝试过程
| 模型       | 参数                                      | 数据                                          | 测试    | 复赛单模 |
|------------|------------------------------------------|----------------------------------------------|---------|--------|
| qwen32_int4| epoch=2                                 | train+额外top1000                             | 0.8132  | 0.79   |
| qwen32_int4| epoch=2, loss到了0.02                    | train+test(gpt标签)                          | 0.8592  | 0.8264 |
| qwen32_int4| epoch=3                                 | train+test(gpt标签)+额外top250                | 0.8614  | 0.8212 |
| qwen32_int4| epoch=3                                 | train(qwen解题思路)+test(qwen解题思路)           | 0.8095  | 0.7993 |
| qwen32_int4| epoch=3                                 | train(juice的解题思路)                        | 0.8015  |        |
| qwen32_int4| epoch=2                                 | train+test(gpt标签)+额外top100                |   |0.8380  |
| qwen32_int4| epoch=2                                 | train(去掉16题)+test(gpt标签)+额外top100        | 0.8614  | 0.8344 |
| qwen32_int4| epoch=2                                 | check_v1 判断闭世界(train data) train+test(gpt标签)+额外top100 | 0.8569  | 0.8249 |
| qwen32_int4| epoch=2                                 | check_v2 判断错误(train data) train+test(gpt标签)+额外top100 | 0.8330  |        |

1. 尝试过“qwen解题思路”:逐步给出解题思路，在32bint4 的模型下推理速度太慢，且结果不利于投票放弃。
2. 尝试过“初赛测试集合人工标签”：发现,复赛表现并不好，放弃放弃。
3. 尝试过：对于LogiQuest_25000.jsonl 对RAG，然后检索相似问题，发现 LogiQuest_25000数据 数据质量较差放弃。
4. 尝试过：额外数据使用 LogiQuest_25000.jsonl 中随机 抽取 5000，10000 条数据，结果并不理想。放弃。
5. 尝试过：额外数据使用 LogiQuest_25000.jsonl 中前抽取 100，250，500 条数据，结果并不理想。放弃。
6. 尝试过：微调第二个模型，对第一次预测结果进行判断，没有提升放弃 。
最后发现：train+test(gpt标签)+额外top100  是最优单模型成绩。0.8380
之后模型qwen2.5 发布得到单模型切换到 Qwen2.5-32B-Instruct-GPTQ-Int4 。
之前数据并不理想，为了增加随机性，所以对以上数据进行随机抽样80%，进行投票。 最后21投票效果最佳。