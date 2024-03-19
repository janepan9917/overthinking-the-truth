#!/bin/bash
source /ext3/miniconda3/bin/activate overthinking

# for CoT early exit we want to NOT use --cot (because we want logit bias)
# hotpot_qa

# Steps:
# 1. Run commands to get direct prompting and CoT prompting
# 2. Run analysis doc to seed 
CRP="model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_25-verify_False-cot_True"
# python main.py -d $data --n_examples 25 --early_exit_trajectory --cot_result_prefix $CRP --log INFO
    
MODEL_STR="llama-2-7b-chat"
# for data in rte gsm8k commonsense_qa strategy_qa hotpot_qa; do
for data in rte; do
    # python main.py -d $data --model $MODEL_STR --n_samples 20 --n_examples 2000 --log INFO
    python main.py -d $data --model $MODEL_STR --n_samples 20 --n_examples 2000 --cot --log INFO
done;