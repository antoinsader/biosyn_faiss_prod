import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer

from main_classes.MyEncoder import MyEncoder
from main_classes.MyFaiss import MyFaiss
from helpers.Data import TokensPaths, MyDataset
from config import GlobalConfig, inference_parse_args

def debug_inference():
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")
    
    # Load configuration
    # We use inference_parse_args to get the mention and result_encoder_dir
    cfg = inference_parse_args()
    
    print(f"\n{'='*80}")
    print(f"DEBUGGING INFERENCE FOR MENTION: '{cfg.inference.mention}'")
    print(f"{'='*80}\n")

    # 1. Load Encoder
    print(f"Loading encoder from: {cfg.paths.result_encoder_dir}")
    try:
        encoder = MyEncoder(cfg)
        encoder.encoder.eval()
        print("Encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return

    # 2. Load Tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
        print(f"Tokenizer loaded: {tokenizer.name_or_path}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 3. Tokenize Mention (Debug)
    print(f"\n--- Tokenization Debug ---")
    mention = cfg.inference.mention
    max_length = cfg.tokenize.queries_max_length
    
    # Use the SAME logic as the fixed inference.py
    encoded = tokenizer(
        mention,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    # Decode back to string to see what's actually being fed
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"Decoded (with special tokens): '{decoded}'")
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Tokens: {tokens}")

    # 4. Generate Embedding (Debug)
    print(f"\n--- Embedding Debug ---")
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.inference_mode():
        mention_embed = encoder.get_emb(
            input_ids, 
            attention_mask,
            use_amp=False,
            use_no_grad=True
        )
    
    print(f"Embedding shape: {mention_embed.shape}")
    print(f"Embedding norm: {torch.norm(mention_embed, dim=1).item():.4f}")
    print(f"First 10 values: {mention_embed[0][:10].tolist()}")

    # 5. FAISS Search (Debug)
    print(f"\n--- FAISS Search Debug ---")
    print(f"Loading FAISS index from: {cfg.paths.faiss_path}")
    
    if not os.path.exists(cfg.paths.faiss_path):
        print(f"FAISS index not found at {cfg.paths.faiss_path}. Skipping search.")
        return

    try:
        # Create a minimal dataset and tokens_paths for FAISS
        # Note: This requires the data/tokens directories to exist on the machine
        tokens_paths = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')
        dataset = MyDataset(tokens_paths, cfg)
        
        faiss_wrapper = MyFaiss(
            cfg, 
            save_index_path=cfg.paths.faiss_path,
            dataset=dataset,
            tokens_paths=tokens_paths,
            encoder=encoder
        )
        faiss_wrapper.load_faiss_index(cfg.paths.faiss_path)
        
        if use_cuda:
            mention_embed = mention_embed.contiguous()
        else:
            mention_embed = mention_embed.cpu().numpy().astype(np.float32)
        
        distances, candidate_idxs = faiss_wrapper.faiss_index.search(mention_embed, cfg.inference.topk)
        candidate_idxs = candidate_idxs[0]
        distances = distances[0]
        
        print(f"\nTop-{cfg.inference.topk} Candidates:")
        
        # Load dictionary CUIs
        dictionary_cuis = np.load(tokens_paths.dictionary_cuis_path)
        
        for rank, (idx, dist) in enumerate(zip(candidate_idxs, distances), 1):
            cui = dictionary_cuis[idx]
            print(f"{rank}. Index: {idx}, Distance: {dist:.4f}, CUI: {cui}")
            
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        print("Ensure that data directories and token files exist on this machine.")

if __name__ == '__main__':
    debug_inference()
