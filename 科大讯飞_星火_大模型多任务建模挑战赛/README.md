# 项目说明
## 1.硬件配置
- 系统：Ubuntu22
- python版本：3.10.14
- cuda版本：12.1
- torch版本：2.3.1+cuda12.1
- transformers版本： 4.43.4
- auto_gptq :0.7.1
- vllm ： 0.5.5
## 2.环境依赖
### 完整环境见requirements.txt
## 3.文件和代码结构
- code 
  - models 
    - qwen2-72b-instruct-gptq-int4 模型文件(未下载原始模型文件)
    - xinghuo_lora 存放Lora权重
      - en 
      - person
      - sentiment
  - test.sh 预测脚本
  - test.py 
  - train.sh 训练脚步
  - train.py 
  - xinghuo_config.py
  - xinghuo_util.py
- prediction_result  
  - results.csv 提交结果
- user_data
  - results_en.csv en预测结果
  - results_person.csv
  - results_sentiment.csv
- xfdata 原始数据
- README.md
- requirements.txt
## 4.方法介绍
多任务拆解为3个任务，person，en，sentiment三个任务，给予基础模型，微调训练3个Lora权重，然后分别预测，合并成为结果：
### 4.1.数据预处理
多任务拆解为3个任务，person，en，sentiment三个任务：
- person任务
  - 分析所有label，和context，确定prompt，需要满足：
  - 符合中文的人物姓名的规则，或者符合英文的人物姓名的规则，
  - 排除代词：不提取“我”、“你”等代词。 
  - 排除亲属称谓：不提取“父亲”、“孩子”、“妻子”、“儿子”等亲属称谓。 
  - 排除职业和职务：不提取“工程师”、“CEO”、“主席”、“警方”等职业和职务名称。 
  - 排除公司名称，品牌名称,组织名称，排除地名
- en 任务比较简单
- sentiment 比较简单
### 4.2.模型选择与训练
模型选择方面，我们最后使用了：qwen2-72b-instruct-gptq-int4 （需要下载一下，下载模型文件）
```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2-72B-Instruct-GPTQ-Int4')
```
1. 使用了比赛方的50条训练数据，未拓展其他数据。
2. 使用Lora微调的方式对qwen2-72b-instruct-gptq-int4进行微调训练，对qwen2-72b-instruct-gptq-int4的"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"进行微调，
3. MAX_LENGTH = 2048，学习率1e-4,epoch为2，在一台4张A100上进行微调。

### 4.3.推理
我们使用vllm，--lora-modules 的方式进行部署推理服务。
- person任务：人名提取后，对结果进行了去重。
- en：推理时间较长，设置了非常唱的http client 超时时间。
- sentiment 比较简单

## 5.项目运行步骤
使用 train.sh 进行了训练项目
```python
sh run.sh
```
运行流程为：
1. train.sh中调用了train.py,加载模型，训练，并把权重保存在 xinghuo_lora 下面

使用 test.sh 脚本运行整个项目
```python
sh test.sh
```
运行流程为：
1. 启动vllm sever
2. 然后进行 3个任务的推理。
3. 之后合并结果