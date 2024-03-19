from example import *
from utils import *
from prompt import *
from dataset import * 

import argparse
import openai
import os
import numpy as np
import logging
import json
from datetime import datetime

def get_results_fn(args: argparse.Namespace) -> str:
    """ Gets the results filename. """
    fn_list = []
    for arg in vars(args):
        if arg not in [
            "csv_fn",
            "oai_keyname",
            "log",
            "output_dir",
            "cot_result_prefix"
        ]:
            fn_list.append(f"{arg}_{str(getattr(args, arg))}")
    fn_list.append(str(int(datetime.now().timestamp())))
    fn = "-".join(fn_list)
    return args.output_dir + "/" + fn + ".jsonl"



def run(args: argparse.Namespace):
    np.random.seed(0)

    # set up utilities
    if args.log == "DEBUG":
        logging.basicConfig(level= logging.DEBUG)
    elif args.log == "INFO":
        logging.basicConfig(level= logging.INFO)
    else:
        logging.basicConfig(level= logging.WARNING)

    openai.api_key = os.getenv(args.oai_keyname)
    dataset = get_dataset(args)
    
    if args.early_exit_trajectory:
        assert args.cot_result_prefix is not None
        orig_cot_results_fn = get_other_results_fn(args.cot_result_prefix, args.dataset, args.output_dir)
        lines = open_json(orig_cot_results_fn)
        prompter = get_prompter(args)
        results_fn = get_results_fn(args)  
        logging.info(f"Saving results to {results_fn}")

        for i, r in enumerate(lines):
            if r["gt_solution"] == "Yes":
                prompt = r["prompt"]
                gt_solution = r["gt_solution"]
                
                for j, response in enumerate(r["gpt_response"]):  
                    r["response_idx"] = j
                    
                    # split cot response into steps
                    cot_steps, sep = split_cot_steps(response)
                    
                    # add original prompt
                    cot_steps = [prompt] + cot_steps
                    
                    early_exit_answers = []
                    early_exit_prompts = []
                    for step_idx in range(1, len(cot_steps)):
                        ee_prompt = sep.join(cot_steps[:step_idx])
                        if step_idx != 1:
                            ee_prompt += "\nAnswer: "
                        ee_ans = get_response(
                            args.model, 
                            ee_prompt, 
                            logit_bias=prompter.logit_bias,
                            n_samples=1,
                        )[0]
                        early_exit_answers.append(ee_ans)
                        early_exit_prompts.append(ee_prompt)

                        logging.info(f"Example {i}:\n_________________\n")
                        logging.info(f"Prompt:\n\n{ee_prompt}")
                        logging.info(f"Response Trajectory: {early_exit_answers}")
                        logging.info(f"True Solution: {gt_solution}\n_________________")

                        if args.log == "DEBUG": 
                            import pdb; pdb.set_trace()
                    r["early_exit_answers"] = early_exit_answers
                    r["early_exit_prompts"] = early_exit_prompts

                    # Save results to JSON file
                    with open(results_fn, "a+") as f:
                        print(i)
                        json.dump(r, f)
                        f.write("\n")

    else:
        results_fn = get_results_fn(args)  
        logging.info(f"Saving results to {results_fn}")

        prompter = get_prompter(args)
        examples = dataset.generate_examples()
        np.random.shuffle(examples)

        # load model if not using openai
        model = args.model
        tokenizer = None
        if "gpt" not in args.model:
            model, tokenizer = load_model_and_tokenizer(args.model)
        

        for i, ex in enumerate(examples[:args.n_examples]):
            r = vars(ex)
            p = prompter.build_prompt(ex.question)
            r["prompt"] = p
            r["gpt_response"] = get_response(
                model,
                p, 
                logit_bias=prompter.logit_bias,
                n_samples=args.n_samples,
                tokenizer=tokenizer,
            )

            logging.info(f"Queried example {i}")

            # Save results to JSON file
            with open(results_fn, "a+") as f:
                print(i)
                json.dump(r, f)
                f.write("\n")

            logging.info(f"Example {i}:\n_________________\n")
            logging.info(f"Prompt:\n\n{p}")
            logging.info(f"Response: {r['gpt_response']}")
            logging.info(f"True Solution: {ex.gt_solution}\n_________________")

            if args.log == "DEBUG": 
                p()

        logging.info(f"Results saved at {results_fn}")

    return

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-k", "--oai_keyname", default="HE_OPENAI_KEY", type=str)
    parser.add_argument("--log", type=str, default="DEBUG")
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--dataset", "-d", type=str, default="algebra",
                        help="algebra/manual")
    parser.add_argument("--csv_fn", type=str, default="data/manual_data.csv", 
                        help="Path to data file")
    parser.add_argument("--length", "-l", type=int, default=2)
    parser.add_argument("--n_samples", "-n", type=int, default=10)
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--cot", action="store_true")

    # early exit experiments
    parser.add_argument("--early_exit_trajectory", action="store_true")
    parser.add_argument("--cot_result_prefix", type=str, 
                        help="Prefix to cot results. Don't include dataset (replace with <ds>)")
    args = parser.parse_args()

    return run(args)

if __name__ == "__main__":
    main()