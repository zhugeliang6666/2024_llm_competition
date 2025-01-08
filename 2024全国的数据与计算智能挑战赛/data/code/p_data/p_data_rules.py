import pandas as pd

rule_1=pd.read_json("./data/raw_data/rules1.json")
rule_2=pd.read_json("./data/raw_data/rules2.json")
all_rules_data=pd.concat([rule_1,rule_2]).reset_index()
all_rules_data=all_rules_data.drop("index",axis=1)
all_rules_data.to_json("./data/user_data/all_rules_data.json")