import argparse
import openai
import os
import logging
from typing import List, Dict
from datetime import datetime
import json
import tiktoken
import string
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch

def get_logit_bias(logit_bias: dict, model: str) -> dict:
    encoding = tiktoken.encoding_for_model(model)

    new_logit_bias = {}
    for k, v in logit_bias.items():
        new_logit_bias[encoding.encode(k)[0]] = v
    return new_logit_bias


def get_response(model, prompt: str, logit_bias: dict = None, n_samples: int = 1, tokenizer=None):
    if type(model) == str:
        return get_openai_response(model, prompt, logit_bias=logit_bias, n_samples=n_samples)
    else:
        return get_model_response(model, prompt, logit_bias=logit_bias, n_samples=n_samples, tokenizer=tokenizer, device=0)


def load_model_and_tokenizer(model_str, device=0):
    """
    Get the model and tokenizer corresponding to |model_name|.

    Parameters
    ----------
    model_str : required, str
        File path to the model.
    device : optional, int
        Device to put model on.

    Returns
    ------
    model : AutoModelForCausalLM or AutoModelForSequence2SequenceLM
        Model corresponding to |model_str|.
    tokenizer : AutoTokenizer
        Tokenizer corresponding to |model_str|.
    """

        # tokenizer = AutoTokenizer.from_pretrained(file_path, add_bos_token=False)

    model_str_dict = {
        "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
        "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    }
    # if model_str == "llama-2-13b-chat":

    full_model_str = model_str_dict[model_str]
    full_model_str = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        full_model_str, torch_dtype=torch.float16
    ).to(f"cuda:{device}")
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(full_model_str).to(f"cuda:{device}")
    
    tokenizer = AutoTokenizer.from_pretrained(full_model_str)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_cot_data(output, tokenizer):
    import pdb; pdb.set_trace()
    cot_str = tokenizer.decode(output, skip_special_tokens=True)

def get_model_response(model: str, prompt: str, logit_bias: dict = None, n_samples: int = 1, tokenizer = None, device=0) -> str:
    """ Gets a response from the model. """
    
    # doing CoT
    if logit_bias is None:
        toks = tokenizer(prompt, return_tensors='pt').to(device)
  
        outputs = model.generate(
            **toks,
            num_return_sequences=n_samples,
            temperature=0.7,
            do_sample=True,
            max_new_tokens=375,
            output_scores=True,
            return_dict_in_generate=True, 
        )
    
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        output_length = inputs.input_ids.shape[1] + np.sum(transition_scores.numpy() < 0, axis=1)
        length_penalty = model.generation_config.length_penalty
        reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)


        import pdb; pdb.set_trace()
        output_strs = [get_cot_data(output, tokenizer) for scores, output in outputs]

    # get scores over label words
    else:
        label_toks = {
            k: tokenizer.encode(k)[1] # remove the bos token
            for k in logit_bias.keys()
        }
        toks = tokenizer(prompt, return_tensors='pt').to(device)
        logits = model(
            **toks,
        ).logits
        probs = torch.nn.functional.softmax(logits[0][-1], dim=-1)
        label_probs = {
            k: probs[label_toks[k]].item()
            for k in logit_bias.keys()
        }
        norm_label_probs = {
            k: v / sum(label_probs.values())
            for k, v in label_probs.items()
        }

        output_strs = []
        for k, v in norm_label_probs.items():
            num_instances = round(v * n_samples)
            output_strs += [k] * num_instances


    return output_strs

        

    # return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_openai_response(model: str, prompt: str, logit_bias: dict = None, n_samples: int = 1) -> str:
    """ Gets a response from the OAI model. """

    if logit_bias is not None:
        logit_bias = get_logit_bias(logit_bias, model)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            logit_bias=logit_bias,
            max_tokens=1,
            n=n_samples,
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            n=n_samples,
        )

    return [response.choices[i]["message"]["content"] for i in range(len(response.choices))]


def filter_cot(full_cot):

    # remove prompt
    prompt_sep = "Let's think step-by-step.\n"
    if prompt_sep not in full_cot:
        prompt_sep = "Let\'s think step-by-step."
    split_cot = full_cot.split(prompt_sep)


    # check to see if the prompt got repeated by llama
    if len(split_cot) > 2 and split_cot[0].strip() == split_cot[1].strip():
        prompt = cot_prompt[0]
        cot = cot_prompt[-1]
    elif len(split_cot) > 2:
        return None
    else:
        prompt = split_cot[0]
        cot = split_cot[1]

    # make sure that these phrases aren't in the CoT
    # else the CoT is probably poor quality
    bad_phrases = [
        "Please provide your answer",
        "Let's begin!",
    ]
    for bp in bad_phrases:
        if bp in cot:
            return None

    # make sure these phrases don't end the CoT
    # else the CoT is probably poor quality
    bad_end_phrases = [
        "Thank you!",
        "Please go ahead and provide your answer.",
        "[Your Name]",
    ]

    for bep in bad_end_phrases:
        if bep in cot and cot.split(bep)[-1] == "":
            return None

    if "|" in cot:
        cot = cot.replace("|", "")
    
    return cot
    

def check_answer(answer, solution):
    words = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in answer.split()]
    for s in solution.split():
        if s.lower().strip() not in words:
            return False
    return True

def get_other_results_fn(prefix, dataset, results_dir):
    dp = prefix.replace("<ds>", dataset)

    files = sorted(os.listdir(results_dir))
    
    for f in files:
        if dp in f:
            direct = f
    return os.path.join(results_dir, direct)

def split_cot_steps(response: str) -> List[str]:
    # split cot response into steps
    sep = "\n"
    cot_steps = [x for x in response.split(sep) if x.strip() != ""]
    if len(cot_steps) < 2:
        sep = ". "
        cot_steps = response.split(sep)
    return cot_steps, sep
                    

def p():
    import pdb; pdb.set_trace()
    
def open_json(json_fn):
    with open(json_fn, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        yield json.loads(json_str)

def write_json(json_file: str, d: Dict):
    with open(json_file, "a+") as f:
        json.dump(d, f)
        f.write("\n")