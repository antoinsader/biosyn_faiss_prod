


import numpy as np
import json
import torch

from transformers import AutoTokenizer
from tqdm import tqdm

from config import GlobalConfig
from helpers.Data import MyDataset, TokensPaths, load_queries, load_dictionary
from main_classes.MyEncoder import MyEncoder
from tokenizer import tokenize_names


cfg = GlobalConfig()
tokenize_batch_size = cfg.tokenize.tokenize_batch_size
dictionary_max_length = cfg.tokenize.dictionary_max_length
queries_max_length = cfg.tokenize.queries_max_length
dictionary_max_chars_length = cfg.tokenize.dictionary_max_chars_length

mention_start_special_token = cfg.tokenize.special_tokens_dict["mention_start"]
mention_end_special_token = cfg.tokenize.special_tokens_dict["mention_end"]

use_cuda = torch.cuda.is_available()
device = "cuda"    if use_cuda else "cpu"

tokens_paths  = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')
tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
tokenizer.add_special_tokens(cfg.tokenize.special_tokens)

mention_start_token_id  = tokenizer.convert_tokens_to_ids(cfg.tokenize.special_tokens_dict["mention_start"])
mention_end_token_id  = tokenizer.convert_tokens_to_ids(cfg.tokenize.special_tokens_dict["mention_end"])

meta = {
    "len_tokenizer": len(tokenizer), 
    "mention_start_token_id": mention_start_token_id,
    "mention_end_token_id": mention_end_token_id
}
with open(cfg.paths.tokenizer_meta_path, "w") as f:
    json.dump(meta, f)


my_encoder = MyEncoder(cfg)
dataset = MyDataset(tokens_paths, cfg)



(tokens_size, max_length ) = tokens_paths.queries_shape
N = tokens_size

query_inputs = dataset.queries_input_ids
query_att = dataset.queries_attention_mask

batch_size = 1024
for start in range(0, N,batch_size):
    end = min(start + batch_size, N)
    inp  = torch.as_tensor(query_inputs[start:end], device=device)
    att = torch.as_tensor(query_att[start:end],device=device)
    print(f"start:end: {start}:{end}")
    embs = my_encoder.get_emb(inp, att, use_amp=True, use_no_grad=True)
    del inp, att
del embs
torch.cuda.empty_cache()
print(f"all queries were embeded successfully")

# N = tokens_paths.dictionary_shape[0]
# dictionary_entries_n  = N

# dictionary_inputs = dataset.dictionary_input_ids
# dictionary_att = dataset.dictionary_attention_masks


# for start in tqdm(range(0, N, batch_size), desc="Building faiss index"):
#     end = min(start + batch_size, N)
#     inp  = torch.as_tensor(dictionary_inputs[start:end], device=device)
#     att = torch.as_tensor(dictionary_att[start:end],device=device)
#     embs = my_encoder.get_emb(inp, att, use_amp=True, use_no_grad=True)
#     del inp, att

# del dictionary_inputs, dictionary_att
# torch.cuda.empty_cache()
# print(f"all dictionary entries were embeded successfully")
