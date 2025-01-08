import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import csv
import time
import requests
import string
from http import HTTPStatus
import dashscope
from io import BytesIO
import json
import sys
sys.path.append('/data/jiani/prompt_new/utils')
import sys, torch
parent_dir = '/data/jiani/prompt_new'
sys.path.append(parent_dir)
import utils.models as model

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from transformers.generation.utils import GenerationConfig
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role

from prompt_format_utils import LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_2, SafetyCategory
from prompt_format_utils import build_custom_prompt, build_default_prompt, create_conversation, LlamaGuardVersion, AgentType
from typing import List, Tuple
import time 

sys_prompt = 'From now on you need to answer the following questions. Note!I do not want any explanations. You can only output the answer options(A/B/C/D) for the questions! Attention! You can only output the answer to the question(A/B/C/D)  !## Here is the question:\n'

system_prompt = sys_prompt

gpu_id = 0

# 检查是否有可用的CUDA设备
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)  # 设置当前设备为指定的GPU
    print(f"Using CUDA device: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")



import time  # 引入 time 模块

def mmlu_gpt35(input_csv, output_csv):
    """
    处理MMLU数据集并评估LLM的性能，同时记录运行时间。
    
    :param input_csv: 输入的CSV文件路径
    :param output_csv: 输出的CSV文件路径
    """
    results = []
    total_time = 0  # 用于记录总运行时间
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 获取表头
        
        # 处理每一行数据
        for row in reader:
            question_text = row[0]  # 问题文本
            options = row[1:-1]  # 从第2列到倒数第2列是选项
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  # 构建选项
            full_question = f"{question_text}\n{options_text}"  # 拼接问题和选项
            
            correct_answer = row[-1]  # 最后一列为标准答案
            
            # 开始计时
            start_time = time.time()
            
            # 调用本地LLM生成答案
            generated_answer = model.chatgpt(system_prompt, full_question)
            generated_answer = generated_answer[0]
            
            # 结束计时
            end_time = time.time()
            total_time += (end_time - start_time)  # 累加运行时间
            
            # 比较生成答案与标准答案
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            # 保存结果
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    # 写入评估结果到新的CSV文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    
    # 打印运行时间
    num_questions = len(results)  # 处理的问题数量
    avg_time = total_time / num_questions if num_questions > 0 else 0  # 计算平均运行时间
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"gpt35处理完成！评估结果已保存至：{output_csv}")



def mmlu_gpt4o(input_csv, output_csv):
    """
    处理MMLU数据集并评估LLM的性能。
    
    :param input_csv: 输入的CSV文件路径
    :param output_csv: 输出的CSV文件路径
    """
    results = []
    total_time = 0  # 用于记录总运行时间
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 获取表头
        
        # 处理每一行数据
        for row in reader:
            question_text = row[0]  # 问题文本
            options = row[1:-1]  # 从第2列到倒数第2列是选项
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  # 构建选项
            full_question = f"{question_text}\n{options_text}"  # 拼接问题和选项
            
            correct_answer = row[-1]  # 最后一列为标准答案
            # 开始计时
            start_time = time.time()
            
            # 调用本地LLM生成答案
            generated_answer = model.gpt4o(system_prompt, full_question)
            generated_answer = generated_answer[0]
            
            # 结束计时
            end_time = time.time()
            total_time += (end_time - start_time)  # 累加运行时间
            
            # 比较生成答案与标准答案
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            # 保存结果
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    # 写入评估结果到新的CSV文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    # 打印运行时间
    num_questions = len(results)  # 处理的问题数量
    avg_time = total_time / num_questions if num_questions > 0 else 0  # 计算平均运行时间
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")

    print(f"gpt4处理完成！评估结果已保存至：{output_csv}")



def mmlu_qwent(input_csv, output_csv):
    """
    处理MMLU数据集并评估LLM的性能。
    
    :param input_csv: 输入的CSV文件路径
    :param output_csv: 输出的CSV文件路径
    """
    results = []
    total_time = 0  # 用于记录总运行时间
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 获取表头
        
        # 处理每一行数据
        for row in reader:
            question_text = row[0]  # 问题文本
            options = row[1:-1]  # 从第2列到倒数第2列是选项
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  # 构建选项
            full_question = f"{question_text}\n{options_text}"  # 拼接问题和选项
            
            correct_answer = row[-1]  # 最后一列为标准答案
            # 开始计时
            start_time = time.time()
            
            # 调用本地LLM生成答案
            generated_answer = model.qwent(system_prompt, full_question)
            # 结束计时
            end_time = time.time()
            total_time += (end_time - start_time)  # 累加运行时间
            
            # 比较生成答案与标准答案
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            # 保存结果
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    # 写入评估结果到新的CSV文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
        # 打印运行时间
    num_questions = len(results)  # 处理的问题数量
    avg_time = total_time / num_questions if num_questions > 0 else 0  # 计算平均运行时间
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"qwenturbo处理完成！评估结果已保存至：{output_csv}")

def mmlu_qwenmax(input_csv, output_csv):
    """
    处理MMLU数据集并评估LLM的性能。
    
    :param input_csv: 输入的CSV文件路径
    :param output_csv: 输出的CSV文件路径
    """
    results = []
    total_time = 0  # 用于记录总运行时间
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 获取表头
        
        # 处理每一行数据
        for row in reader:
            question_text = row[0]  # 问题文本
            options = row[1:-1]  # 从第2列到倒数第2列是选项
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  # 构建选项
            full_question = f"{question_text}\n{options_text}"  # 拼接问题和选项
            
            correct_answer = row[-1]  # 最后一列为标准答案
            # 开始计时
            start_time = time.time()
            
            # 调用本地LLM生成答案
            generated_answer = model.qwenmax(system_prompt, full_question)
            # 结束计时
            end_time = time.time()
            total_time += (end_time - start_time)  # 累加运行时间
            
            # 比较生成答案与标准答案
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            # 保存结果
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    # 写入评估结果到新的CSV文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
        # 打印运行时间
    num_questions = len(results)  # 处理的问题数量
    avg_time = total_time / num_questions if num_questions > 0 else 0  # 计算平均运行时间
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"qwenmax处理完成！评估结果已保存至：{output_csv}")


def llama2( system_prompt,input_csv, output_csv):
    
    contral_system_prompt = "You are a useful assistant. Please answer the questions concisely and informatively. Don't repeat questions! You just need to output the answer.You can only complete one conversation "

    model_id = "/data/jiani/prompt/Foundation_Model/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_config = model.config
    max_position_embeddings = model_config.max_position_embeddings

    results = []
    total_time = 0  # 用于记录总运行时间
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 获取表头
        
        # 处理每一行数据
        for row in reader:
            question_text = row[0]  # 问题文本
            options = row[1:-1]  # 从第2列到倒数第2列是选项
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  # 构建选项
            full_question = f"{question_text}\n{options_text}"  # 拼接问题和选项
            
            correct_answer = row[-1]  # 最后一列为标准答案
            # 开始计时
            start_time = time.time()
            

            input_text = f"System: {contral_system_prompt+system_prompt}\nUser: {full_question}\nAssistant:"
            inputs = tokenizer(input_text, return_tensors="pt")

            input_length = inputs['input_ids'].shape[1]

            if input_length > max_position_embeddings:
                print(f"Skipping input with length {input_length} as it exceeds max_length of {max_position_embeddings}.")
                return "Input too long, skipping."
            
            inputs.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=max_position_embeddings,
                    temperature=0.7,  # 控制生成的随机性
                    top_p=0.2,  # 核采样
                    do_sample=True,
                    num_return_sequences=1
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split('Assistant:')[-1].strip()

            # 调用本地LLM生成答案
            generated_answer = response

            # 结束计时
            end_time = time.time()
            total_time += (end_time - start_time)  # 累加运行时间

            
            # 比较生成答案与标准答案
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            # 保存结果
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    # 写入评估结果到新的CSV文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    # 打印运行时间
    num_questions = len(results)  # 处理的问题数量
    avg_time = total_time / num_questions if num_questions > 0 else 0  # 计算平均运行时间
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"llama2处理完成！评估结果已保存至：{output_csv}")


def llama3( system_prompt,input_csv, output_csv):
    
    contral_system_prompt = "You are a useful assistant. Please answer the questions concisely and informatively. Don't repeat questions! You just need to output the answer.You can only complete one conversation "

    model_id = "/data/jiani/prompt/Foundation_Model/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_config = model.config
    max_position_embeddings = model_config.max_position_embeddings

    results = []
    total_time = 0  # 用于记录总运行时间
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 获取表头
        
        # 处理每一行数据
        for row in reader:
            question_text = row[0]  # 问题文本
            options = row[1:-1]  # 从第2列到倒数第2列是选项
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  # 构建选项
            full_question = f"{question_text}\n{options_text}"  # 拼接问题和选项
            
            correct_answer = row[-1]  # 最后一列为标准答案
            # 开始计时
            start_time = time.time()
            

            input_text = f"System: {contral_system_prompt+system_prompt}\nUser: {full_question}\nAssistant:"
            inputs = tokenizer(input_text, return_tensors="pt")

            input_length = inputs['input_ids'].shape[1]

            if input_length > max_position_embeddings:
                print(f"Skipping input with length {input_length} as it exceeds max_length of {max_position_embeddings}.")
                return "Input too long, skipping."
            
            inputs.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=max_position_embeddings,
                    temperature=0.7,  # 控制生成的随机性
                    top_p=0.2,  # 核采样
                    do_sample=True,
                    num_return_sequences=1
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split('Assistant:')[-1].strip()

            # 调用本地LLM生成答案
            generated_answer = response

            # 结束计时
            end_time = time.time()
            total_time += (end_time - start_time)  # 累加运行时间

            
            # 比较生成答案与标准答案
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            # 保存结果
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    # 写入评估结果到新的CSV文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    # 打印运行时间
    num_questions = len(results)  # 处理的问题数量
    avg_time = total_time / num_questions if num_questions > 0 else 0  # 计算平均运行时间
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"llama3处理完成！评估结果已保存至：{output_csv}")

def mistral_chat(sys_prompt, input_csv, output_csv):
    model_id = '/data/jiani/prompt/Foundation_Model/Mistral-7B-Instruct-v0.2'

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    system_prompt = "You are a useful assistant. Please answer the questions concisely and informatively. Don't repeat questions! You just need to output the answer."

    results = []
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 获取表头
        
        # 处理每一行数据
        for row in reader:
            question_text = row[0]  # 问题文本
            options = row[1:-1]  # 从第2列到倒数第2列是选项
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  # 构建选项
            full_question = f"{question_text}\n{options_text}"  # 拼接问题和选项
            
            correct_answer = row[-1]  # 最后一列为标准答案

            messages = [
                {"role": "user", "content": system_prompt},
                {"role": "assistant", "content": "Well, I will answer your question as best as I can. Please ask me any questions you may have."},
                {"role": "user", "content": system_prompt + full_question}
            ]

            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(device)
            model.to(device)

            generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)


            #print(decoded[0])
            for response in decoded:
                # 提取 "<s> [INST]" 之后的内容，并移除结尾的 "</s>"
                response_content = response.split("[INST]")[-1].split("[/INST]")[-1].replace("</s>", "").strip()
    
            generated_answer = response_content
            
            # 比较生成答案与标准答案
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            # 保存结果
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    # 写入评估结果到新的CSV文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"mistral处理完成！评估结果已保存至：{output_csv}")





# mmlu_math = '/data/jiani/prompt_new/nomal_usage/MMLU/elementary_mathematics_test.csv'
# mmlu_us = '/data/jiani/prompt_new/nomal_usage/MMLU/us_foreign_policy_test.csv'

# mmlu_time = '/data/jiani/prompt_new/nomal_usage/MMLU/time.csv'
# # 示例调用

# qwen_math_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/qwen/math_result.csv'  # 输出文件路径
# qwen_us_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/qwen/us_result.csv'

# mmlu_time = '/data/jiani/prompt_new/nomal_usage/MMLU/time.csv'
# # 示例调用

# mmlu_qwen(mmlu_math, qwen_math_result_path)

# mmlu_qwen(mmlu_us, qwen_us_result_path)

# gpt_math_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/gpt35/math_result.csv'  # 输出文件路径
# gpt_us_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/gpt35/us_result.csv'

# mmlu_gpt35(mmlu_math, gpt_math_result_path)

# mmlu_gpt35(mmlu_us, gpt_us_result_path)

# gpt4_math_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/gpt4o/math_result.csv'  # 输出文件路径
# gpt4_us_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/gpt4o/us_result.csv'

# mmlu_gpt4o(mmlu_math, gpt4_math_result_path)

# mmlu_gpt4o(mmlu_us, gpt4_us_result_path)


# ll_math_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/llama2/math_result.csv'  # 输出文件路径
# ll_us_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/llama2/us_result.csv'

# llama2(sys_prompt, mmlu_math, ll_math_result_path)
# llama2(sys_prompt, mmlu_us, ll_us_result_path)

# mis_math_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/mistral2/math_result.csv'
# mis_us_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/before/mistral2/us_result.csv'

# mistral_chat(sys_prompt, mmlu_math, mis_math_result_path)
# mistral_chat(sys_prompt, mmlu_us, mis_us_result_path)

mmlu_time = '/data/jiani/prompt_new/nomal_usage/MMLU/time.csv'
time_result_path = '/data/jiani/prompt_new/nomal_usage/MMLU/time_result.csv'


# mmlu_gpt4o(mmlu_time, time_result_path)
# mmlu_gpt35(mmlu_time, time_result_path)
# mmlu_qwent(mmlu_time, time_result_path)
# mmlu_qwenmax(mmlu_time, time_result_path)
# llama2(sys_prompt, mmlu_time, time_result_path)
llama3(sys_prompt, mmlu_time, time_result_path)
