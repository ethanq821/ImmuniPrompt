import pandas as pd
import json

print("The following results are from the SmoothLLM with different target models:")

# 读取pickle文件
print("GPT4o results: ")
df5 = pd.read_pickle('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/gpt4o_summary.pd')
df5["Target"] = "GPT4o"
# 检查数据
print(df5.head(),'\n')

print("llama2 results: ")
df0 = pd.read_pickle('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/llama2_summary.pd')
df0["Target"] = "llama2"
# 检查数据
print(df0.head(),'\n')

print("qwenmax results: ")
df = pd.read_pickle('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/qwenmax_summary.pd')
df["Target"] = "qwenmax"
# 检查数据
print(df.head(),'\n')

print("qwenplus results: ")
df1 = pd.read_pickle('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/qwenp_summary.pd')
df1["Target"] = "qwenplus"
# 检查数据
print(df1.head(),'\n')

print("qwenturbe results: ")
df2 = pd.read_pickle('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/qwent_summary.pd')
df2["Target"] = "qwenturbe"
# 检查数据
print(df2.head(),'\n')

print("baichuan2 results: ")
df3 = pd.read_pickle('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/baichuan2_summary.pd')
df3["Target"] = "baichuan2"
# 检查数据
print(df3.head(),'\n')

print("mistral results: ")
df4 = pd.read_pickle('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/mistral_summary.pd')
df4["Target"] = "mistral"
# 检查数据
print(df4.head(),'\n')

with open('/data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/results/new/all_summary.jsonl', 'w') as f:
    for index, row in df5.iterrows():
        f.write(f"{row.to_json()}\n")
    for index, row in df0.iterrows():
        f.write(f"{row.to_json()}\n")
    for index, row in df.iterrows():
        f.write(f"{row.to_json()}\n")
    for index, row in df1.iterrows():
        f.write(f"{row.to_json()}\n")
    for index, row in df2.iterrows():
        f.write(f"{row.to_json()}\n")
    for index, row in df3.iterrows():
        f.write(f"{row.to_json()}\n")
    for index, row in df4.iterrows():
        f.write(f"{row.to_json()}\n")

print("All results are saved to all_summary.jsonl")