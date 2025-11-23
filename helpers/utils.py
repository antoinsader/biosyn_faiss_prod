import torch
import pickle
import json
import torch.nn.functional as F
import numpy as np

def save_pkl(ar, fp):
    with open(fp, 'wb') as f:
        pickle.dump(ar, f)
def get_pkl(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)



def compute_metrics(scores, targets, k=5):
    """
    Compute top-k accuracy, MRR, and Margin for retrieval scores.
    Optimized for GPU execution.
    """
    with torch.no_grad():
        # --- Margin Calculation ---
        # Normalize targets to float for margin calc
        if targets.dtype == torch.long:
            # Convert info_nce targets (indices) to one-hot-like float
            # Ensure targets is on the same device as scores
            targets = targets.to(scores.device)
            targets_float = torch.zeros_like(scores)
            valid_mask = (targets >= 0) & (targets < scores.size(1))
            targets_float.scatter_(1, targets[valid_mask].unsqueeze(1), 1.0)
        else:
            targets_float = targets.to(scores.device).float()

        # Positives and Negatives count
        num_pos = targets_float.sum(dim=1)
        num_neg = (1.0 - targets_float).sum(dim=1)
        
        # Avoid division by zero
        num_pos_safe = num_pos.clamp(min=1.0)
        num_neg_safe = num_neg.clamp(min=1.0)
        
        pos_scores_sum = (scores * targets_float).sum(dim=1)
        neg_scores_sum = (scores * (1.0 - targets_float)).sum(dim=1)
        
        avg_pos_score = pos_scores_sum / num_pos_safe
        avg_neg_score = neg_scores_sum / num_neg_safe
        
        # Margin: avg_pos - avg_neg
        # If a query has no positives, avg_pos_score is 0.
        margin = (avg_pos_score - avg_neg_score).mean().item()

        # --- Accuracy and MRR ---
        batch_size, n_candidates = scores.shape
        
        # Sort scores descending
        _, sorted_indices = scores.sort(descending=True, dim=1)
        
        if targets.dtype == torch.long:
            # targets is [B]
            t = targets.unsqueeze(1)
            # Hits: where sorted index matches target index
            hits = (sorted_indices == t) & (t != -100)
        else:
            # targets is [B, N]
            # Ensure targets is on the same device as sorted_indices
            targets = targets.to(scores.device)
            # Gather targets in sorted order
            sorted_targets = targets.gather(1, sorted_indices)
            hits = sorted_targets > 0.5

        # Accuracy @ k
        hits_k = hits[:, :k]
        acc_at_k = hits_k.any(dim=1).float().mean().item()
        
        # MRR
        # Ranks: 1, 2, 3...
        ranks = torch.arange(1, n_candidates + 1, device=scores.device).unsqueeze(0)
        ranks = ranks.expand(batch_size, -1)
        
        # Mask ranks where there is no hit
        masked_ranks = ranks.clone().float()
        masked_ranks[~hits] = float('inf')
        
        # Find first rank (min rank)
        first_rank = masked_ranks.min(dim=1).values
        
        # Reciprocal rank (1/inf = 0)
        reciprocal_ranks = 1.0 / first_rank
        mrr = reciprocal_ranks.mean().item()

    return acc_at_k, mrr, margin


def marginal_nll(scores, labels):

    # Mask out invalid entries (e.g., queries with no positives)
    valid_mask = labels.sum(dim=1) > 0
    if not valid_mask.any():
        return scores.sum() * 0.0

    scores = scores[valid_mask]
    labels = labels[valid_mask]

    # Exponentiate scores
    exp_scores = torch.exp(scores)

    # For each query, numerator = sum of exp(sim) over positive candidates
    numerator = (exp_scores * labels).sum(dim=1)
    denominator = exp_scores.sum(dim=1) + 1e-8  # avoid division by zero

    # Negative log-likelihood
    loss = -torch.log(numerator / denominator + 1e-8)

    return loss.mean()



def info_nce_loss(scores,targets):
    return F.cross_entropy(scores, targets, ignore_index=-100)

