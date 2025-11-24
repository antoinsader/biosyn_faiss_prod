import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer

from main_classes.MyEncoder import MyEncoder
from main_classes.MyFaiss import MyFaiss
from helpers.Data import TokensPaths, MyDataset
from config import GlobalConfig, inference_parse_args



def load_dictionary_names(dictionary_path='./data/raw/train_dictionary.txt'):
    """
    Load dictionary names and CUIs
    Returns: list of (name, cui) tuples
    """
    dictionary = []
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '||' not in line:
                continue
            cui, name = line.split('||', 1)
            dictionary.append((name.strip(), cui.strip()))
    return dictionary


def tokenize_mention(cfg,  tokenizer):
    """
    Tokenize a single mention
    """
    mention = cfg.inference.mention
    max_length = cfg.tokenize.queries_max_length
    
    mention_start_special_token = cfg.tokenize.special_tokens_dict["mention_start"]
    mention_end_special_token = cfg.tokenize.special_tokens_dict["mention_end"]

    mention = mention_start_special_token + " " + mention + " " + mention_end_special_token
    
    encoded = tokenizer(
        mention,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }


def main():
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")
    
    # Load configuration
    cfg = inference_parse_args()
    
    # Load encoder
    print(f"Loading encoder from: {cfg.paths.result_encoder_dir}")
    encoder = MyEncoder(cfg)
    encoder.load_state(cfg.paths.result_encoder_dir)
    encoder.encoder.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
    tokenizer.add_special_tokens(cfg.tokenize.special_tokens)
    
    # Tokenize mention
    print(f"\nProcessing mention: '{cfg.inference.mention}'")
    mention_tokens = tokenize_mention(cfg, tokenizer)
    mention_tokens = {k: v.to(device) for k, v in mention_tokens.items()}
    
    # Get mention embedding
    print("Computing mention embedding...")
    with torch.inference_mode():
        mention_embed = encoder.get_emb(
            mention_tokens['input_ids'], 
            mention_tokens['attention_mask'],
            use_amp=False,
            use_no_grad=True
        )
    
    # Load FAISS index
    print(f"Loading FAISS index from: {cfg.paths.faiss_path}")
    
    # Create a minimal dataset and tokens_paths for FAISS
    tokens_paths = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')
    dataset = MyDataset(tokens_paths, cfg)
    
    faiss = MyFaiss(
        cfg, 
        save_index_path=cfg.paths.faiss_path,
        dataset=dataset,
        tokens_paths=tokens_paths,
        encoder=encoder
    )
    faiss.load_faiss_index(cfg.paths.faiss_path)
    # faiss.build_faiss(cfg.faiss.build_batch_size)
    # Search FAISS
    print(f"Searching for top-{cfg.inference.topk} candidates...")
    if use_cuda:
        mention_embed = mention_embed.contiguous()
    else:
        mention_embed = mention_embed.cpu().numpy().astype(np.float32)
    
    _, candidate_idxs = faiss.faiss_index.search(mention_embed, cfg.inference.topk * 3)
    candidate_idxs = candidate_idxs[0]  # Get first (and only) query's results
    
    # Load dictionary CUIs
    dictionary_cuis = np.load(tokens_paths.dictionary_cuis_path)
    
    # Load dictionary names (optional, for display)
    try:
        dictionary_data = load_dictionary_names()
        dictionary_dict = {cui: name for name, cui in dictionary_data}
    except Exception as e:
        print(f"Warning: Could not load dictionary names: {e}")
        dictionary_dict = {}
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Top-{cfg.inference.topk} Candidates for: '{cfg.inference.mention}'")
    print(f"{'='*80}\n")
    
    results = []
    seen_cuis = set()
    seen_names = set()
    for rank, idx in enumerate(candidate_idxs, 1):
        if rank > cfg.inference.topk:
            break


        cui = dictionary_cuis[idx]
        name = dictionary_dict.get(cui, "N/A")

        if cui in seen_cuis or name in seen_names:
            continue

        seen_cuis.add(cui)
        seen_names.add(name)

        results.append({
            'rank': rank,
            'cui': cui,
            'name': name,
            'index': int(idx)
        })
        print(f"{rank}. CUI: {cui}")
        if name != "N/A":
            print(f"   Name: {name}")
        print()
    
    return results


if __name__ == '__main__':
    results = main()
