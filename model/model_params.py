gpt2_xl = {
    "max_token_len": 1024,
    "max_demos": 20,
    "num_layers": 48,
    "num_heads": 25,
    "head_dim": 4800,
    "mlp_dim": 6400,
    "file_path": "gpt2-xl",
    "model_name": "gpt2_xl",
}

gpt_j = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 28,
    "num_heads": 16,
    "head_dim": 4096,
    "mlp_dim": 16384,
    "file_path": "EleutherAI/gpt-j-6B",
    "model_name": "gpt_j",
}

gpt_neox = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 44,
    "num_heads": 65,
    "head_dim": 18432,
    "mlp_dim": 24576,
    "file_path": "EleutherAI/gpt-neox-20b",
    "model_name": "gpt_neox",
}

# TODO: fill out information
llama_2_7b_chat = {
    "max_token_len": 4096,
    "max_demos": 40,
    "num_layers": 32,
    "num_heads": 32,
    "head_dim": None,
    "mlp_dim": None,
    "file_path": "meta-llama/Llama-2-7b-chat-hf",
    "model_name": "llama_2_7b_chat",
}

llama_2_13b_chat = {
    "max_token_len": 4096,
    "max_demos": 40,
    "num_layers": 40,
    "num_heads": 40,
    "head_dim": None,
    "mlp_dim": None,
    "file_path": "meta-llama/Llama-2-13b-chat-hf",
    "model_name": "llama_2_13b_chat",
}

flan_t5_large = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 24,
    "num_heads": 16,
    "head_dim": None,
    "mlp_dim": None,
    "file_path": "google/flan-t5-large",
    "model_name": "flan-t5-large",
}



MODEL_PARAMS = {
    "gpt2_xl": gpt2_xl,
    "gpt_j": gpt_j,
    "gpt_neox": gpt_neox,
    "llama_2_7b_chat": llama_2_7b_chat,
    "llama_2_13b_chat": llama_2_13b_chat,
    "flan_t5_large": flan_t5_large,
}