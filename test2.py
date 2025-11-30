import faiss
import torch 
import numpy as np
from config import GlobalConfig
from helpers.Data import TokensPaths, MyDataset, load_dictionary, load_queries

import logging

from helpers.utils import compute_metrics
from main_classes.Reranker import Reranker
from main_classes.MyEncoder import MyEncoder
from main_classes.MyFaiss import MyFaiss
from helpers.MyLogger import CheckPointing
from helpers.MyLogger import MyLogger
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import json


LOGGER = logging.getLogger()

cfg = GlobalConfig()
cfg.logger.tag = "TRAIN"
checkpointing = CheckPointing(cfg)
logger = MyLogger(LOGGER , checkpointing.current_experiment_log_path, cfg.logger.tag )


tokenize_batch_size = cfg.tokenize.tokenize_batch_size
dictionary_max_length = cfg.tokenize.dictionary_max_length
dictionary_max_chars_length = cfg.tokenize.dictionary_max_chars_length
queries_max_length = cfg.tokenize.queries_max_length
mention_start_special_token = cfg.tokenize.special_tokens_dict["mention_start"]
mention_end_special_token = cfg.tokenize.special_tokens_dict["mention_end"]

queries_cuis = None
dictionary_cuis = None

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


train_queries = load_queries(
    data_dir=cfg.paths.queries_raw_dir,
    queries_max_length=queries_max_length,
    special_token_start=mention_start_special_token ,
    special_token_end=mention_end_special_token,
    tokenizer=tokenizer)

queries_cuis = [q[1] for q in train_queries]
queries_names = [q[0].replace("MESH:", "") for q in train_queries]
queries_names[:10]

dictionary_names_normal, dictionary_cuis, dictionary_names_annotated = load_dictionary(cfg.paths.dictionary_raw_path, 
                                special_token_start=mention_start_special_token, 
                                special_token_end=mention_end_special_token,
                            dictionary_max_chars_length=dictionary_max_chars_length,
                            add_synonyms=bool(cfg.tokenize.dictionaries_annotate and cfg.tokenize.dictionary_annotation_add_synonyms)
                                )
