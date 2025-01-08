#!/bin/bash

python main.py \
    --results_dir ./results \
    --target_model llama2\
    --attack OUR \
    --attack_logfile /data/jiani/prompt_new/test_smoothllm_defense/smooth-llm-main/data/GCG/test.jsonl \
    --smoothllm_pert_type 'RandomSwapPerturbation' \
    --smoothllm_pert_pct 10 \
    --smoothllm_num_copies 10