from multiprocessing import  set_start_method
import os

from tqdm import tqdm
import numpy as np
import json


from datasets import Dataset
from transformers import AutoTokenizer


from config import GlobalConfig, tokenizer_parse_args
from helpers.Data import TokensPaths, load_queries, load_dictionary


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_NUM_THREADS"] = str(min(8, os.cpu_count() or 8))


def tokenize_names(names, input_ids_memmap_path, attention_masks_memmap_path, max_length):
    batch_size = cfg.tokenize.tokenize_batch_size
    N = len(names)

    print(f"Tokenizing...")
    dataset = Dataset.from_dict({"text": names})
    tokenized = dataset.map(
        lambda e: tokenizer(e["text"], padding="max_length", truncation=True, max_length=max_length),
        batched=True,
        batch_size=20_000,
        num_proc=os.cpu_count()
    )


    
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

    print(f"Finished tokenizing, saving..")

    for start in tqdm(range(0, N, batch_size), desc=f"Saving tokens"):
        end = min(start+batch_size, N)
        enc = tokenized[start:end]
        input_ids_mmap[start:end] = np.asarray(enc["input_ids"], np.int32)
        att_mask_mmap[start:end] = np.asarray(enc["attention_mask"], np.int32)
        del enc

    input_ids_mmap.flush()
    att_mask_mmap.flush()

    return (N, max_length)
    
def split_queries(cfg: GlobalConfig, train_queries_key='train_queries', test_queries_key='test_queries'):
    token_groups =  cfg.paths.get_default_token_groups()
    train_queries_paths = token_groups[train_queries_key]
    test_queries_paths = token_groups[test_queries_key]

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

    split_idx = int(train_n * ( 1- cfg.tokenize.test_split_percentage) )

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
        mode="w+",
        dtype=np.int32,
        shape=new_test_inputs.shape
    )
    test_inputs_mmap[:] = new_test_inputs[:]
    test_inputs_mmap.flush()

    test_attention_mmap = np.memmap(
        test_queries_paths['attention_mask'],
        mode="w+",
        dtype=np.int32,
        shape=new_test_attentions.shape
    )
    test_attention_mmap[:] = new_test_attentions[:]
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
        dictionary = load_dictionary(cfg.paths.dictionary_raw_path)
        dictionary_cuis = [q[1] for q in dictionary]
        dictionary_names = [q[0] for q in dictionary]
        np.save(tokens_paths.dictionary_cuis_path, dictionary_cuis)

        tokenize_names(dictionary_names, tokens_paths.dictionary_input_ids_path, tokens_paths.dictionary_attention_mask_path, max_length=dictionary_max_length)
        meta = {"shape": (len(dictionary_cuis), dictionary_max_length)}
        with open(tokens_paths.dictionary_meta  , "w") as f:
            json.dump(meta, f)


    if not cfg.tokenize.skip_tokenize_test_queries:

        print(f"Reading test queries...")
        test_tokens_paths = TokensPaths(cfg, dictionary_key='dictionary', queries_key='test_queries')


        test_queries = load_queries(cfg.paths.queries_raw_dir)
        test_queries_cuis = [q[1] for q in test_queries]
        test_queries_names = [q[0] for q in test_queries]
        np.save(test_tokens_paths.queries_cuis_path, test_queries_cuis)

        tokenize_names(test_queries_names, test_tokens_paths.queries_input_ids_path, test_tokens_paths.queries_attention_mask_path, max_length=queries_max_length)

        meta = {"shape": (len(test_queries_cuis), queries_max_length)}
        with open(test_tokens_paths.queries_meta  , "w") as f:
            json.dump(meta, f)



    if cfg.tokenize.split_train_queries:
        split_queries(cfg, 'train_queries', 'test_queries')
