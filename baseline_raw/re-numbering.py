import json, os
from collections import OrderedDict

def get_subdirectories(directory):
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return subdirectories

def load_jsonl(file_path):
    datas = []
    with open(file_path, 'r') as f:
        for line in f:
            datas.append(json.loads(line))
    return datas

def main():
    direct = '/data/jiani/prompt_new/test_origin/result'
    model_names = get_subdirectories(direct)
    model_list = ['mistral', 'mistralv03']

    for model in model_names:
        tot_counter = 1
        suff_counter = 1
        if model in model_list:
            file_path = os.path.join(direct, model, '1.jsonl')
            datas = load_jsonl(file_path)
            for data in datas:
                if(suff_counter>14):
                    suff_counter = 1
                data['prefix_id'] = data['id']
                data['id'] = tot_counter
                data['suff_id'] = suff_counter
                tot_counter += 1
                suff_counter += 1
                
            with open(os.path.join(direct, model, 're-number.jsonl'), 'w') as f:
                for data in datas:
                    ordered_data = OrderedDict([
                        ('id', data['id']),
                        ('prefix_id', data['prefix_id']),
                        ('suffix_id', data['suff_id']),
                        ('prompt', data['prompt']),
                        ('response', data['response']),
                        ('jailbreak', data['jailbreak']),
                    ])
                    f.write(json.dumps(ordered_data) + '\n')
            print(f"Results of Model: {model} has been re-numbered!")
    

if __name__ == "__main__":
    main()