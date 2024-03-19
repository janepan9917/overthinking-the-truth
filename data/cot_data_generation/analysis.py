from utils import *
import numpy as np
import os
import pandas as pd
import string

direct_prefix = "model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_25-verify_False-cot_False-early_exit_trajectory_False"
# cot_prefix = "model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_25-verify_False-cot_True-early_exit_trajectory_False"

ee_prefix = "model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_25-verify_False-cot_False-early_exit_trajectory_True"
# direct_prefix = "model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_25-verify_False-cot_False-early_exit_trajectory_False"
# cot_prefix = "model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_25-verify_False-cot_True"

results_dir = "output/"

def write_to_file(fn, prompt, gt_solution):
    with open(fn, "a+") as f:
        prompt = prompt.replace("\n", "\\n")
        f.write(f"\"{prompt}\"|{gt_solution}\n")

def get_overthinking_step(answer_trajectory, gt_solution):
    # if initial direct prompt was wrong
    if answer_trajectory[0] != gt_solution:
        return None
    
    # if the answer trajectory was correct to start with...
    correct = True
    step = None
    for i, answer in enumerate(answer_trajectory[1:]):
        # if the answer switches to wrong answer
        if answer != gt_solution and correct:
            correct = False
            step = i+1

        # if the answer swaps back to the correct answer from having been incorrect
        elif answer == gt_solution and not correct:
            return None
            
    return step


def get_overthinking_files(dataset, results_dir, ee_prefix):
    # 4 files: before overthinking, right before overthinking, at overthinking, final
    for dataset in ["rte"]:
        # start_fn = f"rte_overthinking_data/{dataset}_overthinking_start.csv"
        # two_before_overthinking_fn = f"rte_overthinking_data/{dataset}_overthinking_2_before.csv"
        # before_overthinking_fn = f"rte_overthinking_data/{dataset}_overthinking_before.csv"
        # overthinking_fn = f"rte_overthinking_data/{dataset}_overthinking_step.csv"
        # end_fn = f"rte_overthinking_data/{dataset}_overthinking_end.csv"

        result_fn = get_other_results_fn(ee_prefix, dataset, results_dir)
    
        # get file header
        # for fn in [start_fn, before_overthinking_fn, overthinking_fn, end_fn]:
        for fn in [two_before_overthinking_fn]:
            print(fn)
            with open(fn, "a+") as f:
                f.write("text|label\n")

        for example in open_json(result_fn):
            prompt = example["prompt"]
            gt_solution = example["gt_solution"]

            early_exit_prompts = example["early_exit_prompts"]
            answer_trajectory = example["early_exit_answers"]

            # get overthinking step
            overthinking_step = get_overthinking_step(answer_trajectory, gt_solution)

            # write the 4 prompts to file
            if overthinking_step is not None:
                # write_to_file(start_fn, prompt, gt_solution)

                # if overthinking_step != 1:
                #     step_before_overthinking_prompt = early_exit_prompts[overthinking_step-1]
                #     write_to_file(before_overthinking_fn, step_before_overthinking_prompt, gt_solution)

                if overthinking_step > 2:
                    two_before_overthinking_prompt = early_exit_prompts[overthinking_step-2]
                    write_to_file(two_before_overthinking_fn, two_before_overthinking_prompt, gt_solution)

                # overthinking_prompt = early_exit_prompts[overthinking_step]
                # write_to_file(overthinking_fn, overthinking_prompt, gt_solution)

                # final_prompt = early_exit_prompts[-1]
                # write_to_file(end_fn, final_prompt, gt_solution)



    
direct_prefix = "model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_2000-verify_False-cot_False-early_exit_trajectory_False-1707342697.jsonl"
cot_prefix = "model_gpt-3.5-turbo-dataset_<ds>-length_2-n_samples_10-n_examples_2000-verify_False-cot_True-early_exit_trajectory_False-1707343202.jsonl"

# cot_prefix = 'model_llama-2-13b-chat-dataset_<ds>-length_2-n_samples_20-n_examples_2000-verify_False-cot_True-early_exit_trajectory_False-1707839037.jsonl'
# direct_prefix = "model_llama-2-13b-chat-dataset_<ds>-length_2-n_samples_20-n_examples_2000-verify_False-cot_False-early_exit_trajectory_False-1707837577.jsonl"

direct_prefix = "model_llama-2-13b-chat-dataset_commonsense_qa-length_2-n_samples_20-n_examples_2000-verify_False-cot_False-early_exit_trajectory_False-1707948363.jsonl"
cot_prefix = "model_llama-2-13b-chat-dataset_commonsense_qa-length_2-n_samples_20-n_examples_2000-verify_False-cot_True-early_exit_trajectory_False-1707875660.jsonl"

# direct_prefix = "model_llama-2-13b-chat-dataset_strategy_qa-length_2-n_samples_20-n_examples_2000-verify_False-cot_False-early_exit_trajectory_False-1707875625.jsonl"
# cot_prefix = "model_llama-2-13b-chat-dataset_strategy_qa-length_2-n_samples_20-n_examples_2000-verify_False-cot_True-early_exit_trajectory_False-1707875818.jsonl"


DATASETS = [
    # "strategy_qa",
    "commonsense_qa"
]
OPENAI=False
results_dir = "output/"
for dataset in DATASETS:
    print(f"Dataset: {dataset}")
    # get filenames
    data_fn_prefix = f"../llama_overthinking_data/llama_{dataset}_overthinking"
    result_fn = get_other_results_fn(direct_prefix, dataset, results_dir)
    cot_result_fn = get_other_results_fn(cot_prefix, dataset, results_dir)
    
    # get cot performance
    
    cot_acc = []
    for example in open_json(cot_result_fn):
        # import pdb; pdb.set_trace()
        answers = []
        for r in example["gpt_response"]:
            if "Answer:" in r:
                answers.append(r.split("Answer:")[-1].strip())
            
        # import pdb; pdb.set_trace()
        correct_answers = list(filter(lambda x: check_answer(x, example["gt_solution"]), answers))
        cot_acc.append(len(correct_answers)/len(answers))

    # get direct performance
    acc = []
    for example in open_json(result_fn):
        # import pdb; pdb.set_trace()
        correct_answers = list(filter(lambda x: check_answer(x, example["gt_solution"]), example["gpt_response"]))
        acc.append(len(correct_answers)/len(example["gpt_response"]))

    # get questions for which cot_acc < acc
    idx = [i for i in range(min(len(cot_acc), len(acc))) if cot_acc[i] < acc[i]]
    correct_idx = [i for i in range(len(acc)) if i not in idx]
    # import pdb; pdb.set_trace()

    # prepare data files
    for j in range(5):
        data_fn = f"{data_fn_prefix}_q{j}.csv"
        with open(data_fn, "w") as f:
            f.write("text|label\n")

    # get the CoT
    examples = list(open_json(cot_result_fn))
    for i in idx:
        example = examples[i]
        gt_solution = example["gt_solution"]
        prompt = example["prompt"]


        for raw_r in example["gpt_response"]: # for each CoT
            r = filter_cot(raw_r)
            if r is not None: # get CoT answer
                cot_ans = r.split("Answer:")[-1].strip()
                r = r.replace("Answer: Yes", "")
                r = r.replace("Answer: No", "")
                cot_steps, step_sep_str = split_cot_steps(r)

                # check to make sure the CoT is long enough and that CoT answer is wrong
                if len(cot_steps) >= 4 and not check_answer(cot_ans, gt_solution):
                    quarters = [
                        prompt + "\n\n" + step_sep_str.join(cot_steps[:int(len(cot_steps)/4*i)]) for i in range(5)
                    ]
 
                    # print data
                    for j in range(5):
                        data_fn = f"{data_fn_prefix}_q{j}.csv"
                        rte_overthinking_data = write_to_file(data_fn, quarters[j], gt_solution)
            

# CODE FOR GETTING OVERTHINKING DATA SPREADSHEET
    # for dataset in ["rte"]:
    #     cot_traj_fn = f"{dataset}_cot_trajectory2.tsv"
    #     result_fn = get_other_results_fn(ee_prefix, dataset, results_dir)
    #     with open(cot_traj_fn, "w") as f:
    #         f.write("Question/Step\tAnswer\tType\n")
    #         for example in open_json(result_fn):
    #             # p()
    #             prompt = example["prompt"]
    #             answer_trajectory = example["early_exit_answers"]
    #             f.write(f"{prompt}\t{answer_trajectory[0]}\tDirect Prompting Answer\n")
    #             cot_steps = split_cot_steps(example["gpt_response"][example["response_idx"]])[0]
    #             for i, step in enumerate(cot_steps[:-1]):
    #                 # p()
    #                 if i < len(answer_trajectory)-2:
    #                     print(cot_steps[i])
    #                     f.write(f"{cot_steps[i]}\t{answer_trajectory[i+1]}\tEarly Exit Answer {i}\n")
    #                 else:
    #                     f.write(f"{cot_steps[i]}\t{answer_trajectory[i+1]}\tFinal CoT Answer\n")
    #             f.write("\n\n")
            