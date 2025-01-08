import json

datas = []
with open('/data/jiani/prompt_new/test_llamaguard/llama_guard2_query_concise.jsonl', 'r') as f:
    for line in f:
        datas.append(json.loads(line))

with open('/data/jiani/prompt_new/test_llamaguard/llama_guard2_query_concise_safe.jsonl', 'w') as f:
    for data in datas:
        if data["judgment"] == "safe":
            f.write(json.dumps(data) + "\n")
