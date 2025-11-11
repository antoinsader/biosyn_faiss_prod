


import numpy as np
import json
import torch

from transformers import AutoTokenizer


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

train_queries = load_queries(cfg.paths.queries_raw_dir, 
                                    queries_max_length=queries_max_length,
                special_token_start=mention_start_special_token, 
                special_token_end=mention_end_special_token, 
                )
train_queries = train_queries[:10]
queries_names = [q[0] for q in train_queries]
queries_cuis = [q[1] for q in train_queries]
queries_sentences = [q[2] for q in train_queries]
np.save(tokens_paths.queries_cuis_path, queries_cuis)

tokenize_names(queries_sentences, 
                tokens_paths.queries_input_ids_path, 
                tokens_paths.queries_attention_mask_path, 
                max_length=queries_max_length,
                batch_size=tokenize_batch_size, tokenizer = tokenizer)


meta = {"shape": (len(queries_cuis), queries_max_length)}
with open(tokens_paths.queries_meta  , "w") as f:
    json.dump(meta, f)


dictionary = load_dictionary(cfg.paths.dictionary_raw_path, 
                                special_token_start=mention_start_special_token, 
                                special_token_end=mention_end_special_token,
                                dictionary_max_chars_length=dictionary_max_chars_length,
                                )
dictionary = dictionary[:10]
dictionary_cuis = [q[1] for q in dictionary]
# dictionary_names = [q[0] for q in dictionary]
dictionary_names_annotated = [q[2] for q in dictionary]
np.save(tokens_paths.dictionary_cuis_path, dictionary_cuis)

tokenize_names(dictionary_names_annotated, tokens_paths.dictionary_input_ids_path, tokens_paths.dictionary_attention_mask_path, max_length=dictionary_max_length, 
                batch_size=tokenize_batch_size, tokenizer=tokenizer)
meta = {"shape": (len(dictionary_cuis), dictionary_max_length)}
with open(tokens_paths.dictionary_meta  , "w") as f:
    json.dump(meta, f)

# tokens_paths  = TokensPaths(cfg, dictionary_key='small_dictionary', queries_key='train_queries')
tokens_paths  = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')

my_encoder = MyEncoder(cfg)
dataset = MyDataset(tokens_paths, cfg)


dictionary_inputs = dataset.dictionary_input_ids
dictionary_att = dataset.dictionary_attention_masks

N = tokens_paths.dictionary_shape[0]
batch_size = 2

for start in range(0, N, batch_size):
    end = min(start + batch_size, N)
    print(f"embeding cuis: {dataset.dictionary_cuis[start:end]}")
    inps = torch.as_tensor(dictionary_inputs[start:end], device = device)
    atts = torch.as_tensor(dictionary_att[start:end], device = device)
    embs = my_encoder.get_emb(inps, atts, use_amp=True, use_no_grad=True)

print("finished embeding dictionary successfully")
query_inputs = dataset.queries_input_ids
query_att = dataset.queries_attention_mask

N = tokens_paths.queries_shape[0]
batch_size = 2

for start in range(0, N, batch_size):
    end = min(start + batch_size, N)
    print(f"embeding queries cuis: {dataset.queries_cuis[start:end]}")
    inps = torch.as_tensor(query_inputs[start:end], device = device)
    atts = torch.as_tensor(query_att[start:end], device = device)
    embs = my_encoder.get_emb(inps, atts, use_amp=True, use_no_grad=True)


for idx, d in enumerate(dictionary_inputs):
    assert len( (dictionary_inputs[0] == mention_end_token_id).nonzero()[0] ) > 0, f"Dictionary idx: {idx} does not have end token, cui={dataset.dictionary_cuis[idx]}"
    assert len( (dictionary_inputs[0] == mention_start_token_id).nonzero()[0] ) > 0, f"Dictionary idx: {idx} does not have end token, cui={dataset.dictionary_cuis[idx]}"



tokenizer_meta_path  = cfg.paths.tokenizer_meta_path
with open(tokenizer_meta_path) as f:
    tokenizer_meta = json.load(f)
ee = tokenizer_meta["mention_end_token_id"]




dictionary_inputs_ids = dictionary_inputs
N = dictionary_inputs_ids.shape[0]
dictionary_attention_masks = dataset.dictionary_attention_masks
cfg_faiss = cfg.faiss
sample_size= cfg_faiss.clusters_samples(cfg_faiss.num_clusters(N))

sample_indices = torch.randperm(N)[:sample_size]

hidden_size = cfg.model.hidden_size
samples_batch_size = 4_000
samples_embeds = torch.empty((sample_size, hidden_size), dtype=torch.float32)
cursor = 0

start = 203
end = 204
batch_idx = sample_indices[start:end]

inp  = torch.as_tensor(dictionary_inputs_ids[326121], device=device)
mention_end_positions = (inp == mention_end_token_id).nonzero()[0]
tokens = tokenizer.convert_ids_to_tokens(inp)


dataset.dictionary_cuis[326121]

C0617442
att = torch.as_tensor(dictionary_attention_masks[394925],device=device)

batch_embeds = my_encoder.get_emb(inp, att, use_amp=True, use_no_grad=True)

batch_embeds = batch_embeds.contiguous()
samples_embeds[cursor : cursor+(end-start)] = batch_embeds
cursor += (end -start)
del batch_embeds, inp, att

nn = []
with open(cfg.paths.dictionary_raw_path, mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == "": continue
        cui, name = line.split("||")
        if cui != 'C0617442': continue
        name_annotated = " [MS] "  + name + " [ME] "
        nn.append(name_annotated)
