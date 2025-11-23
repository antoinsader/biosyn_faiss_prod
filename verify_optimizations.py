import torch
import numpy as np
import time
from unittest.mock import MagicMock
from main_classes.MyFaiss import MyFaiss
from helpers.Data import MyDataset
from config import GlobalConfig

def original_compute_faiss_recall_at_k(dataset, candidates_idxs, k=10, topk=20):
    k = min(topk, k)
    queries_cuis = dataset.queries_cuis
    dictionary_cuis = np.array(dataset.dictionary_cuis)
    num_queries = len(queries_cuis)

    correct = 0
    for i in range(num_queries):
        query_cui = queries_cuis[i]
        retreived_candidates_cuis = dictionary_cuis[candidates_idxs[i, :k]]
        if query_cui in retreived_candidates_cuis:
            correct += 1

    return correct / max(num_queries, 1)

def original_change_candidates_pool_opt(dataset):
    # Mocking the original logic
    assert dataset.all_candidates_idxs is not None
    
    num_queries, topk = dataset.all_candidates_idxs.shape
    new_cands = dataset.all_candidates_idxs.clone()
    dict_cuis = dataset.dictionary_cuis
    queries_cuis = dataset.queries_cuis

    for query_idx in range(num_queries):
        query_cui = queries_cuis[query_idx]
        current_idxs = new_cands[query_idx].numpy()
        current_cuis = dict_cuis[current_idxs]

        # ORIGINAL LOGIC: Mask creation inside loop
        negative_mask = (current_cuis != query_cui)
        available_positions = np.flatnonzero(negative_mask)

        # Inject hard positives
        if dataset.inject_hard_positives_candidates:
            pos_dict_idxs = dataset.dictionary_cui_to_idx.get(query_cui, [])
            if len(pos_dict_idxs) > 0:
                available_pos_dict = np.setdiff1d(pos_dict_idxs, current_idxs, assume_unique=False)
                if len(available_pos_dict) > 0 and len(available_positions) > 0:
                    pos_n = min(dataset.hard_positives_num,
                                len(available_pos_dict),
                                len(available_positions))
                    chosen_pos_dict = np.random.choice(available_pos_dict, size=pos_n, replace=False)
                    chosen_slots = np.random.choice(available_positions, size=pos_n, replace=False)
                    new_cands[query_idx, chosen_slots] = torch.from_numpy(chosen_pos_dict)
                    available_positions = np.setdiff1d(available_positions, chosen_slots, assume_unique=False)

        if dataset.inject_hard_negatives_candidates and dataset.previous_epoch_candidates is not None:
            prev_idxs = np.array(dataset.previous_epoch_candidates[query_idx])
            prev_cuis = dict_cuis[prev_idxs]
            neg_candidates = prev_idxs[prev_cuis != query_cui]
            if len(neg_candidates) > 0 and len(available_positions) > 0:
                neg_n = min(dataset.hard_negatives_num,
                            len(neg_candidates),
                            len(available_positions))
                chosen_neg_dict = np.random.choice(neg_candidates, size=neg_n, replace=False)
                chosen_slots = np.random.choice(available_positions, size=neg_n, replace=False)
                new_cands[query_idx, chosen_slots] = torch.from_numpy(chosen_neg_dict)

    return new_cands

def verify_faiss_recall():
    print("Verifying compute_faiss_recall_at_k...")
    
    # Setup Mock Data
    num_queries = 100
    topk = 20
    dict_size = 1000
    
    cfg = GlobalConfig()
    cfg.train.topk = topk
    
    dataset = MagicMock(spec=MyDataset)
    dataset.queries_cuis = np.random.randint(0, 100, size=num_queries)
    dataset.dictionary_cuis = np.random.randint(0, 100, size=dict_size)
    
    candidates_idxs = np.random.randint(0, dict_size, size=(num_queries, topk))
    
    # Instantiate MyFaiss
    my_faiss = MyFaiss(cfg, None, dataset, None, None)
    
    # Run Original
    start = time.time()
    recall_orig = original_compute_faiss_recall_at_k(dataset, candidates_idxs, k=5, topk=topk)
    time_orig = time.time() - start
    
    # Run Optimized
    start = time.time()
    recall_opt = my_faiss.compute_faiss_recall_at_k(candidates_idxs, k=5)
    time_opt = time.time() - start
    
    print(f"Original Recall: {recall_orig:.4f}, Time: {time_orig:.6f}s")
    print(f"Optimized Recall: {recall_opt:.4f}, Time: {time_opt:.6f}s")
    
    if np.isclose(recall_orig, recall_opt):
        print("SUCCESS: Recall values match.")
    else:
        print("FAILURE: Recall values do NOT match.")

def verify_data_injection():
    print("\nVerifying change_candidates_pool_opt...")
    
    # Setup Mock Data
    num_queries = 50
    topk = 20
    dict_size = 200
    
    dataset = MagicMock(spec=MyDataset)
    dataset.all_candidates_idxs = torch.randint(0, dict_size, (num_queries, topk))
    dataset.dictionary_cuis = np.random.randint(0, 50, size=dict_size)
    dataset.queries_cuis = np.random.randint(0, 50, size=num_queries)
    
    # Create dictionary mapping
    dataset.dictionary_cui_to_idx = {}
    for idx, cui in enumerate(dataset.dictionary_cuis):
        if cui not in dataset.dictionary_cui_to_idx:
            dataset.dictionary_cui_to_idx[cui] = []
        dataset.dictionary_cui_to_idx[cui].append(idx)
    
    dataset.inject_hard_positives_candidates = True
    dataset.hard_positives_num = 2
    dataset.inject_hard_negatives_candidates = False # Simplify for now
    dataset.previous_epoch_candidates = None
    
    # Run Original with Seed
    np.random.seed(42)
    cands_orig = original_change_candidates_pool_opt(dataset)
    
    # Run Optimized with Seed (Need to bind method or use class method if static, but it's instance method)
    # We need to inject the method into the mock or use the real class but mock attributes.
    # Let's use the real class method logic by creating a dummy class or just calling it if we can import it.
    # But we modified the file on disk. So we can import MyDataset.
    
    # We need to set attributes on the real MyDataset instance
    real_dataset = MyDataset.__new__(MyDataset) # Skip init
    real_dataset.all_candidates_idxs = dataset.all_candidates_idxs.clone()
    real_dataset.dictionary_cuis = dataset.dictionary_cuis
    real_dataset.queries_cuis = dataset.queries_cuis
    real_dataset.dictionary_cui_to_idx = dataset.dictionary_cui_to_idx
    real_dataset.inject_hard_positives_candidates = True
    real_dataset.hard_positives_num = 2
    real_dataset.inject_hard_negatives_candidates = False
    real_dataset.previous_epoch_candidates = None
    
    np.random.seed(42)
    cands_opt = real_dataset.change_candidates_pool_opt()
    
    # Compare
    if torch.equal(cands_orig, cands_opt):
        print("SUCCESS: Candidate tensors are identical.")
    else:
        print("FAILURE: Candidate tensors differ.")
        # print("Original:\n", cands_orig)
        # print("Optimized:\n", cands_opt)
        diff = (cands_orig != cands_opt).sum()
        print(f"Number of differing elements: {diff}")

if __name__ == "__main__":
    verify_faiss_recall()
    verify_data_injection()
