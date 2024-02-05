#!/usr/bin/env bash

# python logit_lens.py --models all --datasets all --settings all --num_inputs 1000 --num_demos "max"
# python logit_lens.py --models gpt_j --datasets sst2 --settings permuted_incorrect_labels --num_inputs 1 --num_demos 2
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_start --settings true_labels --num_inputs 34 --num_demos 1
python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_before --settings true_labels --num_inputs 34 --num_demos 1
python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_two_before --settings true_labels --num_inputs 34 --num_demos 1
python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_step --settings true_labels --num_inputs 34 --num_demos 1
python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_end --settings true_labels --num_inputs 34 --num_demos 1

#flan_t5_large
# python logit_lens.py --models llama_2_13b_chat --datasets rte_overthinking_start --settings true_labels --num_inputs 34 --num_demos 1

# jupyuter notebook
# jupyter lab --ip 0.0.0.0 --port 8965 --no-browser
