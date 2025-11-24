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
        if targets.device != scores.device:
            targets = targets.to(scores.device)

        pos_mask = (targets > 0.5)
        neg_mask =  ~pos_mask

        num_pos = pos_mask.sum(dim=1).clamp(min=1.0)
        num_neg = neg_mask.sum(dim=1).clamp(min=1.0)

        avg_pos = (scores * pos_mask.float()).sum(dim=1) / num_pos
        avg_neg = (scores * neg_mask.float()).sum(dim=1) / num_neg

        margin = avg_pos - avg_neg


        _, sorted_indices = scores.sort(descending=True, dim=1)
        hits = (sorted_targets > 0.5)

        acc_at_k = hits[:, :k].any(dim=1).float().mean()
        first_hit_idx = hits.float().argmax(dim=1)
        has_hit = hits.any(dim = 1)
        mrr_scores = 1.0 / (first_hit_idx.float() + 1.0)
        mrr = (mrr_scores * has_hit.float()).mean()
        
    return acc_at_k.item(), mrr.item(), margin.item()


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

