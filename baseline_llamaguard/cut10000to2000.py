import json
id2000 = []
with open('/data/jiani/prompt_new/llamaguard_judge/attack_benign/clean_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        id2000.append(data['id'])

with open('/data/jiani/prompt_new/test_llamaguard/llama_guard2_baseline_result_benign.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['id'] in id2000:
            with open('/data/jiani/prompt_new/test_llamaguard/llama_guard2_baseline_result_benign_2000.jsonl', 'a') as f1:
                f1.write(json.dumps(data) + '\n')