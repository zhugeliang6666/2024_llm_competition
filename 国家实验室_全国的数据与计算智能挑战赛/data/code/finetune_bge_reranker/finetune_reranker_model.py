import pandas as pd
import json
train_data = pd.read_json("./data/raw_data/dev.json")
rule_data=pd.read_json("./data/user_data/all_rules_data.json")
train_data1 = pd.read_json("./data/raw_data/复赛新增训练参考集.json")
train_data1=train_data1[~train_data1["refered_rules"].isna()]
train_data1['rule_id'] = train_data1['refered_rules'].apply(lambda x: [int(y.split('.')[0]) for y in x])
train_data=pd.concat([train_data,train_data1])
# 创建一个空列用于存放规则文本
train_data['rule_data'] = None

# 遍历每一条记录，并填充rule_data列
for index, row in train_data.iterrows():
    rule_ids = row['rule_id']
    rule_texts = []
    for rule_id in rule_ids:
        rule_texts.append(rule_data.loc[int(rule_id) - 1, 'rule_text'])
    train_data.at[index, 'rule_data'] = "\n".join(rule_texts)

pretrain_data = [{'text': (misconception)} for misconception in list(train_data["rule_data"].values)]

finetune_data = [
    {
        'query': query.strip(),
        'pos': [misconception.strip()],
        'neg': []    # 负样本
    } for query, misconception in train_data[['question_text', 'rule_data']].values
]
with open('./data/user_data/finetune_bge_reranker/pretrain_data.jsonl', 'w') as f:
    for entry in pretrain_data:
        json.dump(entry, f)
        f.write('\n')
        
with open('./data/user_data/finetune_bge_reranker/finetune_data.jsonl', 'w') as f:
    for entry in finetune_data:
        json.dump(entry, f)
        f.write('\n')