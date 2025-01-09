import json
import os

folders = ['qwenplus']

# 读取json文件
for folder in folders:
    total_runtime = 0
    total_loadtime = 0
    with open('/data/root/prompt_new/llamaguard_judge/attack_jailbreak/'+folder+'/our_overhead.jsonl', 'r') as f:
        datas = [json.loads(line) for line in f]
        folder_runtime = 0
        folder_loadtime = 0
        for data in datas:
            folder_runtime += data['runtime']
        print(folder, "total runtime", folder_runtime, 'average runtime', folder_runtime/len(datas))
        total_runtime += folder_runtime
        total_loadtime += folder_loadtime
print("total runtime", total_runtime)
print("total loadtime", total_loadtime)
print("total_time", total_runtime+total_loadtime)
print("average total runtime", total_runtime/len(folders))
print('average single runtime', total_runtime/len(folders)/len(datas))
print('average loadtime', total_loadtime/len(folders)/len(datas))