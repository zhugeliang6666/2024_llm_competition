# 2024_llm_competition
2024年参加的一些比赛汇总


### 2024年10月
### 阿里云天池_第二届世界科学智能大赛
- **官网**：[阿里云天池](http://competition.sais.com.cn/competitionDetail/532231/format)
- **名次**：8/2815
- **问题**：逻辑推理，闭世界，
- **方案**：qwen,lora,vllm,quantization

### 科大讯飞_星火_大模型多任务建模挑战赛
- **官网**：[科大讯飞](https://challenge.xfyun.cn/topic/info?type=multi-task-modeling-challenge&ch=dw_dmx)
- **名次**：3/100
- **问题**：信息提取,多任务
- **方案**：qwen,lora,vllm,quantization

### 2024年11月
### 国家实验室_全国的数据与计算智能挑战赛
- **官网**：[DataFountain](https://www.datafountain.cn/competitions/1021/ranking?isRedance=0&sch=2378)
- **名次**：4/376
- **问题**：RAG,逻辑推理
- **RAG方案**：
  - 训练：首先训练MiniCPM-Embedding，再训练bge-reranker-v2-minicpm-layerwise，
  - 检索：MiniCPM-Embedding对测试集进行tok30的相似度，
  - 排序1：bge-reranker-v2-minicpm-layerwise对tok30进行重排序得到tok30分数,
  - 排序2: MiniCPM-Reranker得到tok30分数，
  - 加权融合：（bge-reranker-v2-minicpm-layerwise*0.8+MiniCPM-Reranker**0.2）/2，得到最终的tok10
- **逻辑推理**：
  - 模型权重投票，预测过程中我们会加入hit3作为相关上下文，让模型能够参考规则来进行回答，提高模型的回答准确率，模型推理11次降低随机性的问题，然后还使用了vllm框架作为推理框架，加快模型推理的速度

### 2024年12月
### Kaggle_Eedi - Mining Misconceptions in Mathematics
- **官网**：[Kaggle](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview)
- **名次**：铜牌,134/1446
- **问题**：RAG,逻辑推理
- **RAG方案**：
  - 向量1：个人微调的qwen2.5-14b，计算向量
  - 向量2：开源(微调后)的qwen2.5-14b，计算向量
  - 加权融合：doc=doc_embeddings*0.7+q_misconception_mapping*0.33  embeddings_data=query_embeddings*0.6+q_embeddings*0.33
  - 检索：最近邻检索top25
  - 排序：qwen-32b-instruct-awq，依次在25个中，
    - 序号为17-24,和last1中选取最优
    - 序号为9-16,和last1选取最优，
    - 序号为1-8,和last1选取最优，
    - 得到25个最优的1个放在首位。