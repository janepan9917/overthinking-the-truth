#!/usr/bin/env bash

# python logit_lens.py --models all --datasets all --settings all --num_inputs 1000 --num_demos "max"
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets sst2 --settings permuted_incorrect_labels --num_inputs 1000 --num_demos 30
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_start --settings true_labels --num_inputs 34 --num_demos 1
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_before --settings true_labels --num_inputs 34 --num_demos 1
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_two_before --settings true_labels --num_inputs 34 --num_demos 1
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_step --settings true_labels --num_inputs 34 --num_demos 1
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_end --settings true_labels --num_inputs 34 --num_demos 1


# NUM_DEMOS=3000
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_rte_overthinking_q0 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_rte_overthinking_q1 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_rte_overthinking_q2 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_rte_overthinking_q3 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_rte_overthinking_q4 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1

# NUM_DEMOS=2000
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_strategy_qa_overthinking_q0 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_strategy_qa_overthinking_q1 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_strategy_qa_overthinking_q2 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_strategy_qa_overthinking_q3 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
# CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_strategy_qa_overthinking_q4 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1

NUM_DEMOS=1000
CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_commonsense_qa_overthinking_q0 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_commonsense_qa_overthinking_q1 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_commonsense_qa_overthinking_q2 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_commonsense_qa_overthinking_q3 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1
CUDA_LAUNCH_BLOCKING=1 python logit_lens.py --models llama_2_13b_chat --datasets llama_commonsense_qa_overthinking_q4 --settings true_labels --num_inputs $NUM_DEMOS --num_demos 1



#flan_t5_large
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_start --settings true_labels --num_inputs 34 --num_demos 1

# jupyuter notebook
# jupyter lab --ip 0.0.0.0 --port 8965 --no-browser
