# 项目说明

## 1. 硬件配置及环境依赖

- 系统：Ubuntu22
- python版本：3.12.3
- cuda版本：12.1
- transformers版本：4.44.2
- 详细依赖都在docker文件中

### 因为使用到了Flagembedding框架进行训练reranker模型，所以需要安装Flagembedding

```python
pip install FlagEmbedding==1.3.2
```

### 项目需要安装的库

```python
pip install torch==2.4.0 torchvision==0.19.0 torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install peft==0.11.1
pip install datasets==2.19.0
pip install pandas==2.2.2
pip install transformers==4.44.2
pip install tqdm
pip install vllm==0.5.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install FlagEmbedding==1.3.2
pip install openai loguru httpx -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sentence-transformers==3.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets==2.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
详细依赖都在docker文件中
```

## 2.文件和代码结构

- data/code （存放代码）
  - finetune_bge_reranker(训练bge-reranker-v2-minicpm-layerwise重排序模型)
  - fineturne_minincpm_embedding  （训练词嵌入模型MiniCPM-Embedding）
  - p_data   （拼接rule_id和预测后的伪标签和之前的数据拼接）
  - rag(词嵌入部分）

    - test_rag.py(对测试集进行tok30的检索)
    - rag_reranker.py（minincpm_reranker对tok30进行重排序得到tok30各自分数和tok10）
    - new_reranker.py（bge-reranker-v2-minicpm-layerwise对tok30进行重排序得到tok30各自分数和tok10）
    - avg_reranker_model.py （对minincpm_reranker的tok30分数和bge-reranker-v2-minicpm-layerwise的tok30分数进行融合得到最终的tok10）
  - train_qwen2_5_7B (训练qwen2.5-7b)
  - vllm_pred（使用vllm框架进行推理）

    - test_vllm.py（单模型推理11次进行硬投票）
- data/user_data
  - bge-large-zh-v1.5 (词嵌入模型，作用：查找负样本，参数量0.4B)
  - bge-reranker-v2-minicpm-layerwise(重排序模型，作用：对tok30进行排序，参数量2.4B)
  - finetune_bge_reranker(训练bge-reranker-v2-minicpm-layerwise得到的模型输出文件)
  - fineturne_minincpm_embedding(训练MiniCPM-Embedding得到的模型输出文件)
  - MiniCPM-Embedding(词嵌入模型，作用：对数据集查找相似度tok30，参数量2.4B)
  - MiniCPM-Reranker(重排序模型，作用：对tok30进行排序，参数量2.4B)
  - Qwen2.5-7B-Instruct(LLM基座模型，作用：对数据集进行预测，参数量7B)

## 3. 方法介绍

### 1.数据预处理

设计prompt为指令+问题+选项+答案，其中答案为一个选项，因为本次比赛不能标注数据，所以不能使用cot模式。

### 2. 模型的选择和下载

#### 初赛

因为比赛限制模型参数大小，所以我们最终使用qwen2.5-7b

分数为0.441

### 复赛

也是使用qwen2.5-7b

### 3. 模型训练

训练比赛方的训练数据+初赛测试集（伪标签）+复赛训练集（伪标签）

使用Lora微调的方式对Qwen2.5-7B-Instruct进行微调训练，对Qwen2.5-7B-Instruct的"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"进行微调，
max_len为512，学习率1e-4,epoch为2，经过数据分析，大部分训练数据的长度都不超过512.

### 4.RAG部分

首先训练MiniCPM-Embedding，再训练bge-reranker-v2-minicpm-layerwise，之后使用MiniCPM-Embedding对测试集进行tok30的相似度，之后使用bge-reranker-v2-minicpm-layerwise对tok30进行重排序得到tok30分数,然后使用MiniCPM-Reranker得到tok30分数，最后使用加权融合（bge-reranker-v2-minicpm-layerwise*0.8+MiniCPM-Reranker**0.2）/2，得到最终的tok10

### 4. 模型预测

我们使用模型权重投票进行预测，预测过程中我们会加入hit3作为相关上下文，让模型能够参考规则来进行回答，提高模型的回答准确率，模型推理11次降低随机性的问题，然后还使用了vllm框架作为推理框架，加快模型推理的速度

## 4.所需要的显存和时间

```python
训练推理显存: 24G
项目全流程 14小时左右
```

## 5.项目运行方法和流程

使用sh脚本运行整个项目

```python
sh data/code/run.sh
```

运行流程为：

在run.sh中有注释
