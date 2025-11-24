import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer

from main_classes.MyEncoder import MyEncoder
from main_classes.MyFaiss import MyFaiss
from helpers.Data import TokensPaths, MyDataset
from config import GlobalConfig


def parse_args():
    """
    Parse input arguments for inference
    """
    parser = argparse.ArgumentParser(description='BioSyn-FAISS Inference')
    
    # Required arguments
    parser.add_argument('--mention', type=str, required=True, 
                        help='Medical mention/entity to normalize (e.g., "breast cancer")')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained encoder directory (e.g., ./output/encoder_1/)')
    
    # Optional arguments
    parser.add_argument('--faiss_index', type=str, default=None,
                        help='Path to FAISS index file (default: <model_dir>/faiss_index.faiss)')
    parser.add_argument('--dictionary_cuis', type=str, default='./data/tokens/_dictionary_cuis.npy',
                        help='Path to dictionary CUIs file')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top candidates to retrieve (default: 5)')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA if available (default: True)')
    parser.add_argument('--max_length', type=int, default=75,
                        help='Max tokenization length (default: 75)')
    
    args = parser.parse_args()
    
    # Default FAISS index path
    if args.faiss_index is None:
        args.faiss_index = os.path.join(args.model_dir, 'faiss_index.faiss')
    
    # Validate paths
    if not os.path.isdir(args.model_dir):
        raise ValueError(f"Model directory does not exist: {args.model_dir}")
    if not os.path.exists(args.faiss_index):
        raise ValueError(f"FAISS index not found: {args.faiss_index}")
    if not os.path.exists(args.dictionary_cuis):
        raise ValueError(f"Dictionary CUIs file not found: {args.dictionary_cuis}")
    
    return args


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
    # Add special tokens for mention boundaries
    mention_start = tokenizer.convert_tokens_to_ids('[MS]')
    mention_end = tokenizer.convert_tokens_to_ids('[ME]')
    
    # Simple tokenization: [MS] mention [ME]
    tokens = tokenizer.encode(mention, add_special_tokens=False)
    tokens = [mention_start] + tokens + [mention_end]
    
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


def main(args):
    # Setup device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")
    
    # Load configuration
    cfg = GlobalConfig()
    cfg.model.model_name = args.model_dir
    cfg.paths.result_encoder_dir = args.model_dir
    cfg.paths.faiss_path = args.faiss_index
    
    # Load encoder
    print(f"Loading encoder from: {args.model_dir}")
    encoder = MyEncoder(cfg)
    encoder.to(device)
    encoder.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    
    # Tokenize mention
    print(f"\nProcessing mention: '{args.mention}'")
    mention_tokens = tokenize_mention(args.mention, tokenizer, args.max_length)
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
    print(f"Loading FAISS index from: {args.faiss_index}")
    
    # Create a minimal dataset and tokens_paths for FAISS
    tokens_paths = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')
    dataset = MyDataset(tokens_paths, cfg)
    
    faiss = MyFaiss(
        cfg, 
        save_index_path=args.faiss_index,
        dataset=dataset,
        tokens_paths=tokens_paths,
        encoder=encoder
    )
    faiss.load_faiss_index(args.faiss_index)
    
    # Search FAISS
    print(f"Searching for top-{args.topk} candidates...")
    if use_cuda:
        mention_embed = mention_embed.contiguous()
    else:
        mention_embed = mention_embed.cpu().numpy().astype(np.float32)
    
    _, candidate_idxs = faiss.faiss_index.search(mention_embed, args.topk)
    candidate_idxs = candidate_idxs[0]  # Get first (and only) query's results
    
    # Load dictionary CUIs
    print(f"Loading dictionary CUIs from: {args.dictionary_cuis}")
    dictionary_cuis = np.load(args.dictionary_cuis)
    
    # Load dictionary names (optional, for display)
    try:
        dictionary_data = load_dictionary_names()
        dictionary_dict = {cui: name for name, cui in dictionary_data}
    except Exception as e:
        print(f"Warning: Could not load dictionary names: {e}")
        dictionary_dict = {}
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Top-{args.topk} Candidates for: '{args.mention}'")
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
    args = parse_args()
    results = main(args)
