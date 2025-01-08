#!/bin/bash

python main.py \
    --results_dir ./results/new \
    --target_model vicuna \
    --attack_model qwenp\
    --attack OUR \
    --attack_logfile /data/jiani/prompt_new/dataset/jailbreak/jailbreak_prompts_question_1_small.jsonl \
    --smoothllm_pert_type 'RandomSwapPerturbation' \
    --smoothllm_pert_pct 10 \
    --smoothllm_num_copies 10