import os
import time
import argparse
import logging
import json
import glob
import re
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
import faiss
import faiss.contrib.torch_utils

# ==========================================
# CONFIGURATION & ARGS
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized BioSyn Training for RTX 4090")
    
    # Paths
    parser.add_argument('--tokens_dir', type=str, default='./data/tokens', help='Path to tokenized data')
    parser.add_argument('--output_dir', type=str, default='./output/optimized_model', help='Where to save model')
    parser.add_argument('--model_name', type=str, default='dmis-lab/biobert-base-cased-v1.1')
    parser.add_argument('--tokenizer_meta_path', type=str, default='./data/tokenizer.json')
    
    # Training Params
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training. Adjust based on VRAM.')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--topk', type=int, default=40, help='Number of candidates to retrieve')
    parser.add_argument('--loss_temperature', type=float, default=0.05)
    
    # Optimization Flags
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use Automatic Mixed Precision (Float16/BFloat16)')
    parser.add_argument('--compile_model', action='store_true', default=True, help='Use torch.compile for speedup')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--grad_acc_steps', type=int, default=1, help='Gradient accumulation steps')
    
    # FAISS Params
    parser.add_argument('--faiss_build_batch', type=int, default=16384, help='Batch size for building FAISS index')
    parser.add_argument('--faiss_search_batch', type=int, default=4096, help='Batch size for searching FAISS')
    parser.add_argument('--update_faiss_every', type=int, default=1, help='Update FAISS index every N epochs')
    
    # Hard Negatives
    parser.add_argument('--hard_negatives_num', type=int, default=0)
    parser.add_argument('--hard_positives_num', type=int, default=0)

    # Debugging
    parser.add_argument('--training_log_name', type=str, required=True)
    parser.add_argument('--debug_cpu', action='store_true', help='Force CPU mode for debugging on non-CUDA machines')
    parser.add_argument('--dry_run', action='store_true', help='Run a few batches and exit')

    return parser.parse_args()

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ==========================================
# DATASET
# ==========================================

class FastDataset(Dataset):
    """
    Optimized Dataset class.
    - Loads data from mmap.
    - Handles negative sampling efficiently.
    """
    def __init__(self, tokens_dir, dictionary_key='dictionary', queries_key='train_queries', 
                 topk=20, load_to_ram=True, debug_cpu=False):
        self.tokens_dir = tokens_dir
        self.topk = topk
        self.debug_cpu = debug_cpu
        
        # Load Paths
        self.dict_paths = self._get_paths(tokens_dir, f"_{dictionary_key}")
        self.query_paths = self._get_paths(tokens_dir, f"_{queries_key}")
        
        # Load Shapes
        self.dict_shape = self._load_shape(self.dict_paths['meta'])
        self.query_shape = self._load_shape(self.query_paths['meta'])
        
        # Load Data
        logger.info(f"Loading data from {tokens_dir}...")
        
        # Queries
        self.queries_input_ids = np.memmap(self.query_paths['input_ids'], mode='r', dtype=np.int32, shape=self.query_shape)
        self.queries_att_mask = np.memmap(self.query_paths['attention_mask'], mode='r', dtype=np.int32, shape=self.query_shape)
        self.queries_cuis = np.load(self.query_paths['cuis'])
        
        # Dictionary
        if load_to_ram and not debug_cpu:
            logger.info("Loading dictionary to RAM for speed...")
            self.dict_input_ids = np.array(np.memmap(self.dict_paths['input_ids'], mode='r', dtype=np.int32, shape=self.dict_shape))
            self.dict_att_mask = np.array(np.memmap(self.dict_paths['attention_mask'], mode='r', dtype=np.int32, shape=self.dict_shape))
        else:
            logger.info("Keeping dictionary on disk (mmap)...")
            self.dict_input_ids = np.memmap(self.dict_paths['input_ids'], mode='r', dtype=np.int32, shape=self.dict_shape)
            self.dict_att_mask = np.memmap(self.dict_paths['attention_mask'], mode='r', dtype=np.int32, shape=self.dict_shape)
            
        self.dict_cuis = np.load(self.dict_paths['cuis'])
        
        # Pre-compute CUI to Index mapping for fast positive sampling
        # self.cui_to_idx = defaultdict(list) # Too slow to build in python for 4M?
        # Optimization: We assume we might not need this if we rely on FAISS for hard negatives mostly.
        # But for hard positives, we need it. Let's build it if needed or skip.
        
        self.candidates = None # Will be set by FAISS search
        
        if debug_cpu:
            # Subset for debugging
            logger.info("DEBUG CPU MODE: Slicing data to small subset")
            limit = 100
            self.queries_input_ids = self.queries_input_ids[:limit]
            self.queries_att_mask = self.queries_att_mask[:limit]
            self.queries_cuis = self.queries_cuis[:limit]
            # Dictionary subset? No, we need full dictionary for valid indices usually, but for debug we can slice
            # self.dict_input_ids = self.dict_input_ids[:1000] 
            # self.dict_att_mask = self.dict_att_mask[:1000]
            # self.dict_cuis = self.dict_cuis[:1000]
            # self.dict_shape = (1000, self.dict_shape[1])

    def _get_paths(self, base_dir, prefix):
        return {
            "input_ids": os.path.join(base_dir, f"{prefix}_inp.mmap"),
            "attention_mask": os.path.join(base_dir, f"{prefix}_att.mmap"),
            "cuis": os.path.join(base_dir, f"{prefix}_cuis.npy"),
            "meta": os.path.join(base_dir, f"{prefix}_meta.json"),
        }

    def _load_shape(self, path):
        with open(path) as f:
            return tuple(json.load(f)["shape"])

    def set_candidates(self, candidates_indices):
        """
        candidates_indices: (num_queries, topk)
        """
        self.candidates = candidates_indices

    def __len__(self):
        return len(self.queries_input_ids)

    def __getitem__(self, idx):
        # Query
        q_ids = self.queries_input_ids[idx]
        q_att = self.queries_att_mask[idx]
        q_cui = self.queries_cuis[idx]
        
        # Candidates
        assert self.candidates is not None, "Candidates must be set before training"
        cand_idxs = self.candidates[idx]
            
        # Optimization: We can inject hard negatives/positives here if we had the logic.
        # For now, stick to retrieved candidates.
        
        c_ids = self.dict_input_ids[cand_idxs]
        c_att = self.dict_att_mask[cand_idxs]
        c_cuis = self.dict_cuis[cand_idxs]
        
        # Labels: 1 if candidate CUI matches Query CUI, 0 otherwise
        labels = (c_cuis == q_cui).astype(np.float32)
        
        return {
            "query_input_ids": torch.tensor(q_ids, dtype=torch.long),
            "query_attention_mask": torch.tensor(q_att, dtype=torch.long),
            "cand_input_ids": torch.tensor(c_ids, dtype=torch.long),
            "cand_attention_mask": torch.tensor(c_att, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

# ==========================================
# MODEL
# ==========================================

class OptimizedBioSynModel(nn.Module):
    def __init__(self, model_name, tokenizer_meta_path, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load Tokenizer Meta
        with open(tokenizer_meta_path) as f:
            meta = json.load(f)
        self.mention_start_id = meta["mention_start_token_id"]
        self.mention_end_id = meta["mention_end_token_id"]
        
        # Load Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(meta['len_tokenizer'])
        
        # Projection (assuming CLS + Between Spans strategy from original config)
        self.hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Gradient Checkpointing
        self.encoder.gradient_checkpointing_enable()

    def get_embedding(self, input_ids, attention_mask):
        """
        Compute embeddings for a batch of tokens.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state # (B, Seq, H)
        
        # CLS Embedding
        cls_emb = last_hidden[:, 0, :]
        
        # Mention Span Embedding
        # Find [MS] and [ME] positions
        # Optimization: argmax is fast on GPU
        start_idxs = (input_ids == self.mention_start_id).float().argmax(dim=1)
        end_idxs = (input_ids == self.mention_end_id).float().argmax(dim=1)
        
        # Create mask for between spans
        seq_len = input_ids.size(1)
        arange = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        mask = (arange > start_idxs.unsqueeze(1)) & (arange < end_idxs.unsqueeze(1))
        mask = mask.unsqueeze(-1).float()
        
        # Mean pooling of span
        sum_emb = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        span_emb = sum_emb / counts
        
        # Combine
        combined = torch.cat([cls_emb, span_emb], dim=1)
        projected = self.projection(combined)
        
        # Normalize
        return F.normalize(projected, p=2, dim=1)

    def forward(self, query_ids, query_mask, cand_ids, cand_mask):
        """
        Forward pass for training.
        query: (B, L)
        cand: (B, K, L)
        """
        B, K, L = cand_ids.shape
        
        # Flatten candidates
        cand_ids_flat = cand_ids.view(B * K, L)
        cand_mask_flat = cand_mask.view(B * K, L)
        
        # Embed
        q_emb = self.get_embedding(query_ids, query_mask) # (B, H)
        c_emb = self.get_embedding(cand_ids_flat, cand_mask_flat) # (B*K, H)
        
        # Reshape candidates
        c_emb = c_emb.view(B, K, -1) # (B, K, H)
        
        # Compute Scores (Cosine Similarity)
        # q_emb: (B, 1, H)
        # c_emb: (B, K, H) -> transpose -> (B, H, K)
        scores = torch.bmm(q_emb.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1) # (B, K)
        
        return scores

# ==========================================
# FAISS INDEXER
# ==========================================

class FaissIndexer:
    def __init__(self, hidden_size, device='cuda', debug_cpu=False):
        self.hidden_size = hidden_size
        self.device = device
        self.debug_cpu = debug_cpu
        self.index = None
        self.res = faiss.StandardGpuResources() if not debug_cpu else None
        
    def build(self, embeddings):
        """
        embeddings: numpy array (N, H)
        """
        N, D = embeddings.shape
        logger.info(f"Building FAISS index for {N} vectors of dim {D}...")
        
        if self.debug_cpu:
            logger.info("Using CPU Flat Index for Debugging")
            self.index = faiss.IndexFlatIP(D)
            self.index.add(embeddings)
            return

        # GPU Optimization: Use IVFPQ for large datasets to fit in VRAM
        # If N < 1M, FlatIP is faster and more accurate.
        # If N > 1M, IVFPQ is needed for memory.
        
        if False:
            logger.info("Using GPU Flat Index (Exact Search)")
            config = faiss.GpuIndexFlatConfig()
            config.device = 0 # GPU 0
            self.index = faiss.GpuIndexFlatIP(self.res, D, config)
            self.index.add(embeddings)
        else:
            logger.info("Using GPU IVFPQ Index (Approximate Search)")
            # Quantizer (IVF)
            nlist = int(math.sqrt(N)) # Standard heuristic
            quantizer = faiss.IndexFlatIP(D) # CPU quantizer initially
            
            # Index
            m = 32 # Sub-quantizers (must divide D=768 -> 24)
            nbits = 8
            index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits)
            
            # Train
            logger.info("Training IVFPQ...")
            # Train on a subset
            train_size = min(N, 65536)
            index.train(embeddings[:train_size])
            
            # Add
            logger.info("Adding vectors...")
            index.add(embeddings)
            
            # Move to GPU
            logger.info("Moving index to GPU...")
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True 
            self.index = faiss.index_cpu_to_gpu(self.res, 0, index, co)
            
    def search(self, queries, k):
        """
        queries: numpy array (Q, H)
        """
        logger.info(f"Searching FAISS for {len(queries)} queries...")
        if self.debug_cpu:
            D, I = self.index.search(queries, k)
            return I
            
        # GPU Search
        # Ensure queries are on CPU numpy (FAISS handles transfer usually, or we can pass torch tensors)
        # faiss.contrib.torch_utils allows passing tensors directly!
        
        # But here we assume queries is numpy for simplicity with existing pipeline
        D, I = self.index.search(queries, k)
        return I

# ==========================================
# MAIN TRAINING
# ==========================================

def main():
    args = parse_args()
    
    # Setup Device
    if args.debug_cpu:
        device = torch.device('cpu')
        logger.warning("!!! RUNNING IN DEBUG CPU MODE - EXTREMELY SLOW !!!")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # RTX 4090 Optimization
            torch.set_float32_matmul_precision('high') 
        else:
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
            args.debug_cpu = True

    # Initialize Model
    logger.info("Initializing Model...")
    model = OptimizedBioSynModel(args.model_name, args.tokenizer_meta_path, device=device)
    model.to(device)
    
    # Compile Model (PyTorch 2.0+)
    if args.compile_model and not args.debug_cpu:
        logger.info("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.use_amp and not args.debug_cpu)

    # Dataset
    dataset = FastDataset(args.tokens_dir, debug_cpu=args.debug_cpu)
    
    # FAISS Indexer
    indexer = FaissIndexer(model.hidden_size, device=device, debug_cpu=args.debug_cpu)

    # Training Loop
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{args.num_epochs} ===")
        
        # ----------------------------------
        # 1. Build FAISS Index (Dictionary)
        # ----------------------------------
        if (epoch == 1) or ((epoch - 1) % args.update_faiss_every == 0):
            logger.info("Encoding Dictionary...")
            model.eval()
            
            # Create DataLoader for Dictionary
            dict_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(dataset.dict_input_ids),
                torch.from_numpy(dataset.dict_att_mask)
            )
            dict_loader = DataLoader(
                dict_dataset, 
                batch_size=args.faiss_build_batch if not args.debug_cpu else 32, 
                shuffle=False, 
                num_workers=args.num_workers if not args.debug_cpu else 0,
                pin_memory=not args.debug_cpu
            )
            
            all_embs = []
            with torch.no_grad(), torch.amp.autocast(device_type="cuda" if not args.debug_cpu else "cpu", enabled=args.use_amp and not args.debug_cpu):
                for batch in dict_loader:
                    ids, mask = batch
                    ids, mask = ids.to(device), mask.to(device)
                    emb = model.get_embedding(ids, mask)
                    all_embs.append(emb.cpu().numpy())
                    
                    if args.dry_run and len(all_embs) > 2: break
            
            dictionary_embs = np.concatenate(all_embs, axis=0)
            indexer.build(dictionary_embs)
            del dictionary_embs, all_embs
            if not args.debug_cpu: torch.cuda.empty_cache()

        # ----------------------------------
        # 2. Search Candidates (Queries)
        # ----------------------------------
        logger.info("Searching Candidates...")
        model.eval()
        
        # Encode Queries
        query_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(dataset.queries_input_ids),
            torch.from_numpy(dataset.queries_att_mask)
        )
        query_loader = DataLoader(
            query_dataset, 
            batch_size=args.faiss_search_batch if not args.debug_cpu else 32, 
            shuffle=False, 
            num_workers=args.num_workers if not args.debug_cpu else 0,
            pin_memory=not args.debug_cpu
        )
        
        all_query_embs = []
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if not args.debug_cpu else "cpu", enabled=args.use_amp and not args.debug_cpu):
            for batch in query_loader:
                ids, mask = batch
                ids, mask = ids.to(device), mask.to(device)
                emb = model.get_embedding(ids, mask)
                all_query_embs.append(emb.cpu().numpy())
                
                if args.dry_run and len(all_query_embs) > 2: break

        query_embs = np.concatenate(all_query_embs, axis=0)
        candidates = indexer.search(query_embs, k=args.topk)
        dataset.set_candidates(candidates)
        del query_embs, all_query_embs
        
        # ----------------------------------
        # 3. Train Epoch
        # ----------------------------------
        logger.info("Training...")
        model.train()
        
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers if not args.debug_cpu else 0,
            pin_memory=not args.debug_cpu,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            # Move to device
            q_ids = batch['query_input_ids'].to(device, non_blocking=True)
            q_mask = batch['query_attention_mask'].to(device, non_blocking=True)
            c_ids = batch['cand_input_ids'].to(device, non_blocking=True)
            c_mask = batch['cand_attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward
            with torch.amp.autocast(device_type="cuda" if not args.debug_cpu else "cpu", enabled=args.use_amp and not args.debug_cpu):
                scores = model(q_ids, q_mask, c_ids, c_mask)
                
                # Loss (Marginal NLL)
                # scores: (B, K)
                # labels: (B, K) -> 1.0 for positive, 0.0 for negative
                scores = scores / args.loss_temperature
                
                # Marginal NLL Implementation
                # log_sum_exp of scores - log_sum_exp of (scores where label=1)
                # But standard implementation: -log( sum(exp(pos)) / sum(exp(all)) )
                # = log(sum(exp(all))) - log(sum(exp(pos)))
                
                log_sum_exp_all = torch.logsumexp(scores, dim=1)
                
                # For positives, we mask out negatives with -inf
                pos_scores = scores.clone()
                pos_scores[labels == 0] = float('-inf')
                log_sum_exp_pos = torch.logsumexp(pos_scores, dim=1)
                
                loss = (log_sum_exp_all - log_sum_exp_pos).mean()
                
            # Backward
            scaler.scale(loss).backward()
            
            if (step + 1) % args.grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
            if args.dry_run and step >= 5:
                logger.info("Dry run limit reached. Stopping epoch.")
                break
        
        avg_loss = total_loss / (len(train_loader) + 1)
        logger.info(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
