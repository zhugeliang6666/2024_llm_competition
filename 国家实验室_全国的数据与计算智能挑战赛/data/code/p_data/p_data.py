import pandas as pd
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="vllm")
    parser.add_argument("--data1", type=str)
    parser.add_argument("--data2", type=str)
    parser.add_argument("--data2_sub", type=str)
    parser.add_argument("--output_data", type=str)
    return parser.parse_args()

args = parse_args()

data_1=pd.read_json(args.data1)
data_2=pd.read_json(args.data2)
data_2_answer=pd.read_json(args.data2_sub)
data_2["answer"]=data_2_answer["answer"]
all_data=pd.concat([data_1,data_2]).reset_index()
all_data=all_data.drop("index",axis=1)
all_data.to_json(args.output_data)