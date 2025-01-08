import csv
import os
import numpy as np

def evaluate(input_csv):
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        correct = 0
        total = 0
        for row in reader:
            if row[0] == 'question':
                continue
            ans = f'Option {row[1]}'
            ans2 = f'Answer {row[1]}'
            if(row[3] == 'True'):
                correct += 1
            elif ans in row[2]:
                correct += 1
            elif ans2 in row[2]:
                correct += 1
            total += 1
            
    result = {}
    result['datasets'] = input_csv.split('/')[-1].split('.')[0]
    result['correct'] = correct
    result['total'] = total
    result['accuracy'] = correct / total
    
    return result

if __name__ == '__main__':
    input_dir = '/data/jiani/prompt_new/nomal_usage/MMLU/after'
    output_dir = '/data/jiani/prompt_new/nomal_usage/MMLU/result'
    model_names = ['gpt4oraw']
    summary = []
    total = 0
    correct = 0
    accuracy = []
    for model_name in model_names:
        files = os.listdir(os.path.join(input_dir, model_name))
        output_file = os.path.join(output_dir, f'{model_name}_mmlu.csv')
        with open(output_file, 'w') as f:
            f.write('datasets,correct,total,accuracy\n')
        for file in files:
            input_csv = os.path.join(input_dir, model_name, file)
            result = evaluate(input_csv)
            total += result['total']
            correct += result['correct']
            accuracy.append(result['accuracy'])
            with open(output_file, 'a') as f:
                f.write(f'{result["datasets"]},{result["correct"]},{result["total"]},{result["accuracy"]}\n')
        with open(output_file, 'a') as f:
            f.write(f'average,{correct},{total},{correct / total}\n')
            f.write(f'weight average,{correct},{total},{np.mean(accuracy)}\n')