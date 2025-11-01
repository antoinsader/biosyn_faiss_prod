from multiprocessing import  set_start_method
import os
import random

import numpy as np
from tqdm import tqdm
import json


from datasets import Dataset
from transformers import AutoTokenizer


from config import GlobalConfig, tokenizer_parse_args
from helpers.Data import TokensPaths, load_queries, load_dictionary


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TOKENIZERS_NUM_THREADS"] = str(min(8, os.cpu_count() or 8))



def tokenize_queries(queries, tokenizer, queries_paths, cfg:GlobalConfig):


    window_words = cfg.tokenize.query_tokens_window_words_in_text
    max_length = cfg.tokenize.queries_max_length
    batch_size = cfg.tokenize.tokenize_batch_size
    special_tokens_dict = cfg.tokenize.special_tokens_dict

    print(f"Building full queries")
    full_queries = [buid_full_query(q, window_words, special_tokens_dict) for q in tqdm(queries)]
    print(f"We have: {len(full_queries)} queries..")
    print(f"First 5 queries: {full_queries[:5]}")

    queries_cuis = [q[1] for q in queries]
    queries_cuis = [q.replace("MESH:", "") for q in queries_cuis]
    np.save(queries_paths["ids"], queries_cuis)
    N = len(queries_cuis)


    input_ids_mmap = np.memmap(
        queries_paths['inp'],
        mode="w+",
        dtype=np.int32,
        shape=(N, max_length)
    )
    att_mask_mmap = np.memmap(
        queries_paths['att'],
        mode="w+",
        dtype=np.int32,
        shape=(N, max_length)
    )

    meta = {"shape": (N, max_length)}
    with open(queries_paths['meta'], "w") as f:
        json.dump(meta, f)

    dataset = Dataset.from_dict({"text": full_queries})
    tokenized = dataset.map(
        lambda e: tokenizer(e["text"], padding="max_length", truncation=True, max_length=max_length),
        batched=True,
        num_proc=min(8, os.cpu_count())
    )

    for start in tqdm(range(0, N, batch_size), desc=f"Tokenizing"):
        end = min(start+batch_size, N)
        enc = tokenized[start:end]
        input_ids_mmap[start:end] = np.asarray(enc["input_ids"], np.int32)
        att_mask_mmap[start:end] = np.asarray(enc["attention_mask"], np.int32)
        del enc

    input_ids_mmap.flush()
    att_mask_mmap.flush()

    lengths = []

    for q in tqdm(full_queries, desc="Measuring token lengths"):
        encoded = tokenizer(
            q,
            add_special_tokens=True,   # includes [CLS] and [SEP]
            truncation=False           # we want true length, not truncated
        )
        lengths.append(len(encoded["input_ids"]))

    lengths = np.array(lengths)

    print(f"Total queries: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Median length: {np.median(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95)}")
    print(f"99th percentile: {np.percentile(lengths, 99)}")
    print(f"Max length: {np.max(lengths)}")

    return True



def tokenize_dictionary(cuis, names, paths_key, tokenizer, cfg:GlobalConfig,  cui_to_semantic=None ):
    max_length = cfg.tokenize.dictionary_max_length
    batch_size = cfg.tokenize.tokenize_batch_size
    special_tokens_dict=  cfg.tokenize.special_tokens_dict

    if cui_to_semantic is None:
        cui_to_semantic = {}


    print("Saving cuis..")
    np.save(paths[paths_key]['ids'] , cuis)
    names_size = len(names)

    print(f"Creating memmap...")
    input_ids_mmap = np.memmap(
        paths[paths_key]['inp'],
        mode="w+",
        dtype=np.int32,
        shape=(names_size, max_length)
    )
    att_mask_mmap = np.memmap(
        paths[paths_key]['att'],
        mode="w+",
        dtype=np.int32,
        shape=(names_size, max_length)
    )

    meta = {"shape": (names_size, max_length)}
    with open(paths[paths_key]["meta"], "w") as f:
        json.dump(meta, f)


    for start in tqdm(range(0, names_size, batch_size), desc=f"Tokenizing"):
        end = min(start+batch_size, names_size)
        batch_texts = names[start:end].tolist()
        batch_cuis = cuis[start:end]
        batch_texts = [
            f"{special_tokens_dict['mention_name_start']} {n} {special_tokens_dict['mention_name_end']} "
            f"{special_tokens_dict['context_start']} the name is {special_tokens_dict['mention_in_sentence_start']} {n} {special_tokens_dict['mention_in_sentence_end']} which is showing  {special_tokens_dict['context_end']} "
            f"{special_tokens_dict['type_start']} {cui_to_semantic.get(batch_cuis[idx], "NONE")} {special_tokens_dict['type_end']}"
            for idx,n in enumerate(batch_texts)
        ]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids_mmap[start:end] = np.asarray(enc["input_ids"], np.int32)
        att_mask_mmap[start:end] = np.asarray(enc["attention_mask"], np.int32)
        del batch_texts, enc

    input_ids_mmap.flush()
    att_mask_mmap.flush()
    print("tokenized")
    # names is your np.array or list of dictionary names
    sample_size = min(750_000, len(names))
    sample_indices = random.sample(range(len(names)), sample_size)
    sample_texts = [names[i] for i in sample_indices]
    sample_cuis = [cuis[i] for i in sample_indices]

    print(f"Sampling {len(sample_texts)} out of {len(names)} dictionary entries")
    # tokenize in batches for speed
    batch_size = 16000
    lengths = []

    for start in tqdm(range(0, len(sample_texts), batch_size), desc="Measuring lengths"):
        end = min(start + batch_size, len(sample_texts))
        batch_texts = sample_texts[start:end]
        batch_cuis = sample_cuis[start:end]
        batch_texts = [
            f"{special_tokens_dict['mention_name_start']} {n} {special_tokens_dict['mention_name_end']} "
            f"{special_tokens_dict['context_start']} the name is {special_tokens_dict['mention_in_sentence_start']} {n} {special_tokens_dict['mention_in_sentence_end']} which is showing  {special_tokens_dict['context_end']} "
            f"{special_tokens_dict['type_start']} {cui_to_semantic.get(batch_cuis[idx], "NONE")} {special_tokens_dict['type_end']}"
            for idx,n in enumerate(batch_texts)
        ]
        enc = tokenizer(batch_texts, add_special_tokens=True, truncation=False)
        lengths.extend([len(x) for x in enc["input_ids"]])

    lengths = np.array(lengths)
    print(f"Mean={np.mean(lengths):.1f}, Median={np.median(lengths)}, 95th={np.percentile(lengths,95)}, Max={np.max(lengths)}")
    return True

def tokenize_names(names, input_ids_memmap_path, attention_masks_memmap_path, max_length):
    batch_size = cfg.tokenize.tokenize_batch_size
    N = len(names)


    input_ids_mmap = np.memmap(
        input_ids_memmap_path,
        mode="w+",
        dtype=np.int32,
        shape=(N, max_length)
    )

    att_mask_mmap = np.memmap(
        attention_masks_memmap_path,
        mode="w+",
        dtype=np.int32,
        shape=(N, max_length)
    )

    print(f"Tokenizing...")
    dataset = Dataset.from_dict({"text": names})
    tokenized = dataset.map(
        lambda e: tokenizer(e["text"], padding="max_length", truncation=True, max_length=max_length),
        batched=True,
        num_proc=min(8, os.cpu_count())
    )
    print(f"Finished tokenizing, saving..")

    for start in tqdm(range(0, N, batch_size), desc=f"Saving tokens"):
        end = min(start+batch_size, N)
        enc = tokenized[start:end]
        input_ids_mmap[start:end] = np.asarray(enc["input_ids"], np.int32)
        att_mask_mmap[start:end] = np.asarray(enc["attention_mask"], np.int32)
        del enc

    input_ids_mmap.flush()
    att_mask_mmap.flush()

def split_queries(cfg: GlobalConfig, train_queries_key='train_queries', test_queries_key='test_queries'):
    train_queries_paths = cfg.paths.get_token_group(train_queries_key)
    test_queries_paths = cfg.paths.get_token_group(test_queries_key)

    assert os.path.exists(train_queries_paths['input_ids']) and os.path.exists(train_queries_paths['attention_mask'])  and os.path.exists(train_queries_paths['cuis']), f'Trying to split the train queries but train queries tokens do not exits {train_queries_paths['input_ids']}'

    train_shape = TokensPaths.load_mmap_shape(train_queries_paths['meta'])
    train_inputs_mmap  = np.memmap(
        train_queries_paths['input_ids'],
        mode="r+",
        dtype=np.int32,
        shape=train_shape
    )
    train_attention_mask_mmap  = np.memmap(
        train_queries_paths['attention_mask'],
        mode="r+",
        dtype=np.int32,
        shape=train_shape
    )

    train_cuis = np.load(train_queries_paths['cuis'])
    train_n = len(train_cuis)
    assert train_n == train_shape[0], f"train shape: {train_shape} is not the same as train_n: {train_n}"

    split_idx = int(train_n * cfg.tokenize.test_split_percentage)

    random_rng = np.random.default_rng(seed=42)
    shuffled_random_indices = random_rng.permutation(train_n)

    new_train_idxs = shuffled_random_indices[:split_idx]
    new_test_idxs = shuffled_random_indices[split_idx:]

    new_train_cuis = train_cuis[new_train_idxs]
    np.save(train_queries_paths['cuis'], new_train_cuis)
    new_test_cuis = train_cuis[new_test_idxs]
    np.save(test_queries_paths['cuis'], new_test_cuis)





    new_train_inputs = train_inputs_mmap[new_train_idxs]
    new_train_attentions = train_attention_mask_mmap[new_train_idxs]
    new_test_inputs = train_inputs_mmap[new_test_idxs]
    new_test_attentions = train_attention_mask_mmap[new_test_idxs]

    del train_inputs_mmap, train_attention_mask_mmap

    train_inputs_mmap = np.memmap(
        train_queries_paths['input_ids'],
        mode="r+",
        dtype=np.int32,
        shape=new_train_inputs.shape
    )
    train_inputs_mmap[:] = new_train_inputs[:]
    train_inputs_mmap.flush()

    train_attention_mmap = np.memmap(
        train_queries_paths['attention_mask'],
        mode="r+",
        dtype=np.int32,
        shape=new_train_attentions.shape
    )
    train_attention_mmap[:] = new_train_attentions[:]
    meta = {"shape": train_attention_mmap.shape}
    with open(train_queries_paths['meta'], "w") as f:
        json.dump(meta, f)
    train_attention_mmap.flush()


    test_inputs_mmap = np.memmap(
        test_queries_paths['input_ids'],
        mode="r+",
        dtype=np.int32,
        shape=new_test_inputs.shape
    )
    test_inputs_mmap[:] = new_train_inputs[:]
    test_inputs_mmap.flush()

    test_attention_mmap = np.memmap(
        test_queries_paths['attention_mask'],
        mode="r+",
        dtype=np.int32,
        shape=new_test_attentions.shape
    )
    test_attention_mmap[:] = new_train_attentions[:]
    meta = {"shape": test_attention_mmap.shape}
    with open(test_queries_paths['meta'], "w") as f:
        json.dump(meta, f)

    test_attention_mmap.flush()

    print(f"Train queries were having {train_n} entries, we split them {len(new_train_cuis)} entries for new train queries, and {len(new_test_cuis)} entries for new test queries")




if __name__=="__main__":
    set_start_method("spawn", force=True)


    cfg = tokenizer_parse_args()
    dictionary_max_length = cfg.tokenize.dictionary_max_length
    queries_max_length = cfg.tokenize.queries_max_length

    tokens_paths  = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)

    meta = {"len_tokenizer": len(tokenizer)}
    with open(cfg.paths.tokenizer_meta_path, "w") as f:
        json.dump(meta, f)



    if not cfg.tokenize.skip_tokenize_queries:
        print(f"Reading queries...")
        train_queries = load_queries(cfg.paths.queries_raw_dir)
        queries_cuis = [q[1] for q in train_queries]
        queries_names = [q[0] for q in train_queries]
        np.save(tokens_paths.queries_cuis_path, queries_cuis)

        tokenize_names(queries_names, tokens_paths.queries_input_ids_path, tokens_paths.queries_attention_mask_path, max_length=queries_max_length)

        meta = {"shape": (len(queries_cuis), queries_max_length)}
        with open(tokens_paths.queries_meta  , "w") as f:
            json.dump(meta, f)



    if not cfg.tokenize.skip_tokenize_dictionary:
        print(f"Reading dictionary...")
        dictionary = load_dictionary(cfg.paths.queries_raw_dir)
        dictionary_cuis = [q[1] for q in train_queries]
        dictionary_names = [q[0] for q in train_queries]
        np.save(tokens_paths.dictionary_cuis_path, dictionary_cuis)

        tokenize_names(dictionary_names, tokens_paths.dictionary_input_ids_path, tokens_paths.dictionary_attention_mask_path, max_length=dictionary_max_length)
        meta = {"shape": (len(dictionary_cuis), dictionary_max_length)}
        with open(tokens_paths.dictionary_meta  , "w") as f:
            json.dump(meta, f)


    if not cfg.tokenize.skip_split:
        split_queries(cfg, 'train_queries', 'test_queries')