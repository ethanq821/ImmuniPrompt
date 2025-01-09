#!/bin/bash

python main.py \
    --results_dir ./results/baseline \
    --target_model llama2 \
    --attack OUR \
    --attack_logfile /data/prompt_new/dataset/new/output_q6.jsonl \
    --smoothllm_pert_type 'RandomSwapPerturbation' \
    --smoothllm_pert_pct 10 \
    --smoothllm_num_copies 10