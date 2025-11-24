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


def tokenize_mention(mention, tokenizer, max_length=75):
    """
    Tokenize a single mention
    """
    
    tokens = tokenizer.encode(mention, add_special_tokens=False)
    
    # Pad or truncate
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Create input_ids and attention_mask
    input_ids = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
    attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
    
    return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long),
        'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
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
    encoder.encoder.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
    
    # Tokenize mention
    print(f"\nProcessing mention: '{cfg.inference.mention}'")
    mention_tokens = tokenize_mention(cfg.inference.mention, tokenizer, cfg.tokenize.queries_max_length)
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
    
    # Search FAISS
    print(f"Searching for top-{cfg.inference.topk} candidates...")
    if use_cuda:
        mention_embed = mention_embed.contiguous()
    else:
        mention_embed = mention_embed.cpu().numpy().astype(np.float32)
    
    _, candidate_idxs = faiss.faiss_index.search(mention_embed, cfg.inference.topk)
    candidate_idxs = candidate_idxs[0]  # Get first (and only) query's results
    
    # Load dictionary CUIs
    print(f"Loading dictionary CUIs from: {cfg.paths.dictionary_cuis}")
    dictionary_cuis = np.load(cfg.paths.dictionary_cuis)
    
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
    for rank, idx in enumerate(candidate_idxs, 1):
        cui = dictionary_cuis[idx]
        name = dictionary_dict.get(cui, "N/A")
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
