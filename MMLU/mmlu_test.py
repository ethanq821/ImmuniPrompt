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
sys.path.append('/data/root/prompt_new/utils')
import sys, torch
parent_dir = '/data/root/prompt_new'
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

sys_prompt = 'From now on you need to answer the following questions. Note!I do not want any explanations. You can only output the answer options(A/B/C/D) for the questions! Attention! You can only output the answer to the question(A/B/C/D)  !'

system_prompt = sys_prompt

gpu_id = 0


if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)  
    print(f"Using CUDA device: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")



import time  

def mmlu_gpt35(input_csv, output_csv):
    """
    处理MMLU数据集并评估LLM的性能，同时记录运行时间。
    
    :param input_csv: 输入的CSV文件路径
    :param output_csv: 输出的CSV文件路径
    """
    results = []
    total_time = 0  
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        
        
        for row in reader:
            question_text = row[0]  
            options = row[1:-1]  
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  
            full_question = f"{question_text}\n{options_text}"  
            
            correct_answer = row[-1]  
            
            
            start_time = time.time()
            
            
            generated_answer = model.chatgpt(system_prompt, full_question)
            generated_answer = generated_answer[0]
            
            
            end_time = time.time()
            total_time += (end_time - start_time)  
            
            
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    
    
    num_questions = len(results)  
    avg_time = total_time / num_questions if num_questions > 0 else 0  
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
    total_time = 0  
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        
        
        for row in reader:
            question_text = row[0]  
            options = row[1:-1]  
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  
            full_question = f"{question_text}\n{options_text}"  
            
            correct_answer = row[-1]  
            
            start_time = time.time()
            
            
            generated_answer = model.gpt4o(system_prompt, full_question)
            generated_answer = generated_answer[0]
            
            
            end_time = time.time()
            total_time += (end_time - start_time)  
            
            
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    
    num_questions = len(results)  
    avg_time = total_time / num_questions if num_questions > 0 else 0  
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
    total_time = 0  
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        
        
        for row in reader:
            question_text = row[0]  
            options = row[1:-1]  
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  
            full_question = f"{question_text}\n{options_text}"  
            
            correct_answer = row[-1]  
            
            start_time = time.time()
            
            
            generated_answer = model.qwent(system_prompt, full_question)
            
            end_time = time.time()
            total_time += (end_time - start_time)  
            
            
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
        
    num_questions = len(results)  
    avg_time = total_time / num_questions if num_questions > 0 else 0  
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
    total_time = 0  
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        
        
        for row in reader:
            question_text = row[0]  
            options = row[1:-1]  
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  
            full_question = f"{question_text}\n{options_text}"  
            
            correct_answer = row[-1]  
            
            start_time = time.time()
            
            
            generated_answer = model.qwenmax(system_prompt, full_question)
            
            end_time = time.time()
            total_time += (end_time - start_time)  
            
            
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
        
    num_questions = len(results)  
    avg_time = total_time / num_questions if num_questions > 0 else 0  
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"qwenmax处理完成！评估结果已保存至：{output_csv}")


def llama2( system_prompt,input_csv, output_csv):
    
    contral_system_prompt = "You are a useful assistant. Please answer the questions concisely and informatively. Don't repeat questions! You just need to output the answer.You can only complete one conversation "

    model_id = "/data/root/prompt/Foundation_Model/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_config = model.config
    max_position_embeddings = model_config.max_position_embeddings

    results = []
    total_time = 0  
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        
        
        for row in reader:
            question_text = row[0]  
            options = row[1:-1]  
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  
            full_question = f"{question_text}\n{options_text}"  
            
            correct_answer = row[-1]  
            
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
                    temperature=0.7,  
                    top_p=0.2,  
                    do_sample=True,
                    num_return_sequences=1
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split('Assistant:')[-1].strip()

            
            generated_answer = response

            
            end_time = time.time()
            total_time += (end_time - start_time)  

            
            
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    
    num_questions = len(results)  
    avg_time = total_time / num_questions if num_questions > 0 else 0  
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"llama2处理完成！评估结果已保存至：{output_csv}")


def llama3( system_prompt,input_csv, output_csv):
    
    contral_system_prompt = "You are a useful assistant. Please answer the questions concisely and informatively. Don't repeat questions! You just need to output the answer.You can only complete one conversation "

    model_id = "/data/root/prompt/Foundation_Model/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_config = model.config
    max_position_embeddings = model_config.max_position_embeddings

    results = []
    total_time = 0  
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        
        
        for row in reader:
            question_text = row[0]  
            options = row[1:-1]  
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  
            full_question = f"{question_text}\n{options_text}"  
            
            correct_answer = row[-1]  
            
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
                    temperature=0.7,  
                    top_p=0.2,  
                    do_sample=True,
                    num_return_sequences=1
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split('Assistant:')[-1].strip()

            
            generated_answer = response

            
            end_time = time.time()
            total_time += (end_time - start_time)  

            
            
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    
    num_questions = len(results)  
    avg_time = total_time / num_questions if num_questions > 0 else 0  
    print(f"总运行时间：{total_time:.2f} 秒")
    print(f"平均每个问题运行时间：{avg_time:.2f} 秒")
    
    print(f"llama3处理完成！评估结果已保存至：{output_csv}")

def mistral_chat(sys_prompt, input_csv, output_csv):
    model_id = '/data/root/prompt/Foundation_Model/Mistral-7B-Instruct-v0.2'

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    system_prompt = "You are a useful assistant. Please answer the questions concisely and informatively. Don't repeat questions! You just need to output the answer."

    results = []
    
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        
        
        for row in reader:
            question_text = row[0]  
            options = row[1:-1]  
            options_text = "\n".join([f"Option {chr(65+i)}: {option}" for i, option in enumerate(options)])  
            full_question = f"{question_text}\n{options_text}"  
            
            correct_answer = row[-1]  

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

            
            for response in decoded:
                
                response_content = response.split("[INST]")[-1].split("[/INST]")[-1].replace("</s>", "").strip()
    
            generated_answer = response_content
            
            
            is_correct = (generated_answer.strip().upper() == correct_answer.strip().upper())
            
            
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
    
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["question", "correct_answer", "generated_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"mistral处理完成！评估结果已保存至：{output_csv}")
















































mmlu_time = '/data/root/prompt_new/nomal_usage/MMLU/time.csv'
time_result_path = '/data/root/prompt_new/nomal_usage/MMLU/time_result.csv'







llama3(sys_prompt, mmlu_time, time_result_path)
