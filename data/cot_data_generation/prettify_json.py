"""
Prettify a json.
"""

import json

# json_fn = "output/model_gpt-3.5-turbo-1695301225.jsonl"
# pretty_output_fn = "cleaned_output/model_gpt-3.5-turbo-1695301225.tsv"
# with open(json_fn, 'r') as json_file:
#     json_list = list(json_file)

# for json_str in json_list:
#     result = json.loads(json_str)
#     print(result.keys())
    
#     with open(pretty_output_fn, 'a+') as f:
#         qid = result["qid"]
#         prompt = result["prompt"]
#         gpt_response = result["gpt_response"]
#         mistake_type = result["mistake_type"]
#         mistake_step = result["mistake_step"]   
#         gt_sol = result["gt_solution"]

#         s = []
#         s.append(str(qid))
#         s.append(mistake_type)
#         s.append(str(mistake_step))
#         s.append(prompt)
#         s.append(gpt_response)
#         s.append(gt_sol)
#         s = "\t".join(s)
#         f.write(s)
# print(f"Saved to {pretty_output_fn}")
# json_fn = "data/prontoqa/verified_3hop_1shot_random_nodistractor_testdistractor.jsonl"

json_fn = "data/algebra/data_6.jsonl"
with open(json_fn, 'r') as json_file:
    json_list = list(json_file)

pretty_fn = "test.txt"
for json_str in json_list:
    ex = json.loads(json_str)

    for k, v in ex.items():
        with open(pretty_fn, 'a+') as f:
            f.write(f"{k}\n{v}\n\n")
            f.write("____________________\n\n")