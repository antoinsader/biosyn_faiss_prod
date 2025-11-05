
import glob, json, os, torch, re

from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from config import  GlobalConfig





class TokensPaths():
    """
        Class depending on the keys of the queries and dictionary responsible for: 
            1- giving the paths to tokens and cuis
            2- Giving the shape of dictionary and queries

        dictionary_key could be 'dictionary' for the normal one or 'small_dictionary' for the minimized one
        queries_key could be 'train_queries' for the normal one or 'test_queries' for the test queries

    """
    def __init__(self, cfg:GlobalConfig, dictionary_key='dictionary', queries_key='train_queries'):

        tokens_paths=  cfg.paths.get_default_token_groups()
        dictionary_paths = tokens_paths[dictionary_key]
        queries_paths = tokens_paths[queries_key]

        self.dictionary_input_ids_path = dictionary_paths['input_ids']
        self.dictionary_attention_mask_path = dictionary_paths['attention_mask']
        self.dictionary_cuis_path = dictionary_paths['cuis']
        self.dictionary_meta = dictionary_paths['meta']
        if os.path.exists(dictionary_paths['meta']):
            self.dictionary_shape = self.load_mmap_shape(dictionary_paths['meta'])

        self.queries_input_ids_path = queries_paths['input_ids']
        self.queries_attention_mask_path = queries_paths['attention_mask']
        self.queries_cuis_path = queries_paths['cuis']
        self.queries_meta = queries_paths['meta']
        if os.path.exists(queries_paths['meta']):
            self.queries_shape = self.load_mmap_shape(queries_paths['meta'])


    @staticmethod
    def load_mmap_shape(json_file):
        with open(json_file) as f:
            meta = json.load(f)
        return tuple(meta["shape"])



# ======================
# MY DATASET
# ======================

class MyDataset(Dataset):
    """
        This class responsible for:
            1- Open mmap file for the tokens to be used
            2- Inject into the candidates pool negative and positive candidates depending on config
            3- Giving batches for the training
    """
    def __init__(self,tokens_paths: TokensPaths, cfg: GlobalConfig):
        self.tokens_paths  = tokens_paths

        self.topk = cfg.train.topk
        self.loss_type = cfg.train.loss_type
        self.all_candidates_idxs = None


        self.dictionary_cuis  = np.load(self.tokens_paths.dictionary_cuis_path)
        self.queries_cuis  = np.load(self.tokens_paths.queries_cuis_path)


        self.previous_epoch_candidates = None


        self.inject_hard_negatives_candidates = cfg.train.inject_hard_negatives_candidates
        self.hard_negatives_num = cfg.train.hard_negatives_num

        self.inject_hard_positives_candidates = cfg.train.inject_hard_positives_candidates
        self.hard_positives_num = cfg.train.hard_positives_num


        self.queries_input_ids = np.memmap(
                self.tokens_paths.queries_input_ids_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.queries_shape
            )

        self.queries_attention_mask = np.memmap(
                self.tokens_paths.queries_attention_mask_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.queries_shape
            )

        self.dictionary_input_ids = np.memmap(
                self.tokens_paths.dictionary_input_ids_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.dictionary_shape
            )
        self.dictionary_attention_masks = np.memmap(
                self.tokens_paths.dictionary_attention_mask_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.dictionary_shape
            )


        self.dictionary_cui_to_idx = defaultdict(list)
        for idx, cui in enumerate(self.dictionary_cuis):
            self.dictionary_cui_to_idx[cui].append(idx)

    def __len__(self,):
        return len(self.queries_input_ids)

    def __getitem__(self, query_idx):
        """
            This function is for the batching, takes as an argument the query_idx:
                query_idx is list of size batch_size for the queries of the batch
                return (query_tokens, candidate_tokens), labels
                query_tokens and candidate_tokens are dictionaries having input_ids, attention_mask ready to be fed into the encoder
                query_tokens['input_ids'], query_tokens['attention_mask'] has shape (batch_size, max_length)
                candidate_tokens['input_ids'], candidate_tokens['attention_mask'] has shape (batch_size, topk, max_length)
                labels (if error_type is marginal_nlll) has shape (batch_size, topk), where each item is 0.0 or 1.0
        """
        assert self.all_candidates_idxs is not None
        query_tokens = {
            "input_ids": self.queries_input_ids[query_idx],
            "attention_mask": self.queries_attention_mask[query_idx],
        }
        candidate_idxs = self.all_candidates_idxs[query_idx]
        assert len(candidate_idxs) == self.topk

        query_cui = self.queries_cuis[query_idx] #1

        query_positive_idxs = self.dictionary_cui_to_idx.get(query_cui, [])
        query_positive_idxs= [q for q in query_positive_idxs if q != query_idx]
        assert len(query_positive_idxs) > 0, f"Query idx: {query_idx} with cui: {query_cui} does not have any positives idxs"


        candidate_tokens = {
            "input_ids": self.dictionary_input_ids[candidate_idxs],
            "attention_mask": self.dictionary_attention_masks[candidate_idxs],
        }

        query_candidates_cuis = np.array(self.dictionary_cuis)[candidate_idxs] #(batch_size, topk)
        #   if error_type == 'info_nce_loss', will return [batch_size] for each item is the first match 
        #       for marginal_nll error_type will return  (batch_size, topk) for each item 0 if false, 1 for true
        labels = self.get_labels(query_candidates_cuis, query_cui) 
        return (query_tokens, candidate_tokens), labels


    def get_labels(self, query_candidates_cuis, query_cui):
        """
            Generate labels for a query:
            - InfoNCE: integer index of first positive, -100 if no match
            - Marginal NLL: float vector of 0.0/1.0 per candidate
        """
        if self.loss_type == "info_nce_loss":
            matches = np.where(query_candidates_cuis == query_cui)[0]
            if len(matches) == 0:
                return torch.tensor(-100, dtype=torch.long)
            else:
                return torch.tensor(matches[0], dtype=torch.long)
        elif self.loss_type == "marginal_nll":
            labels = (query_candidates_cuis == query_cui).astype(np.float32)
            return torch.tensor(labels, dtype=torch.float)


    def change_candidates_pool(self):
        """
            If this is the first epoch (we know from previous_epoch_candidates) then just return the candidates idxs
            Otherwise:
                    looping through the queries
                    If inject_hard_positive_candidates is true, we will choose random in the query candidates array to be replaced by positive candidates (same cui as the query)
                    If inject_hard_negative_candidates is true, we will choose random in the query candidates array to be replaced by negative candidates (same cui as the query)

            The hard negative candidates would be from previous epoch candidates where FAISS thought those are positive candidates but he was wrong, this process will allow us to make our encoder learn from its mistakes
            The positions for hard candidates and negative candidates will not overlap
            Number of positive and negative candidates to be injected is depending on the config

            all_candidates_idxs are current candidates (N queries, topk)
            previous_epoch_candidates are candidates from last_epoch (N queries, topk)
            return new_cands (N queries, topk)

        """
        assert self.all_candidates_idxs is not None, "Candidates are not set"

        if self.previous_epoch_candidates is None:
            return self.all_candidates_idxs





        num_queries = self.all_candidates_idxs.shape[0]
        new_cands = self.all_candidates_idxs.clone()

        for query_idx in range(num_queries):
            query_cui = self.queries_cuis[query_idx]
            current_query_candidates_idxs = new_cands[query_idx].tolist()
            
            current_candidates_cuis = self.dictionary_cuis[current_query_candidates_idxs]
            positive_positions = np.where(current_candidates_cuis == query_cui)[0]
            candidates_idxs_available = list(set(range(self.topk))  - set(positive_positions)  )


            candidates_idxs_to_be_replaced = np.array([])
            if self.inject_hard_positives_candidates:
                #those are dictionary idxs having the same cui as the query
                positive_candidates_indexes = self.dictionary_cui_to_idx.get(query_cui, [])
                if len(positive_candidates_indexes) > 0:
                    # positives indexes other than the  current candidates indexes
                    available_positives = list(set(positive_candidates_indexes) - set(current_query_candidates_idxs))
                    if available_positives:
                        # how many positives we will inject, in case available are less than the one in config
                        positive_n = min(self.hard_positives_num, len(available_positives))
                        #  random positive candidates, to choose from available positives (index of dictionary_cui)
                        positive_candidates = np.random.choice(available_positives, size=positive_n, replace=False)
                        # random indexes in candidate list to be replaced
                        candidates_idxs_to_be_replaced = np.random.choice(candidates_idxs_available , size=positive_n, replace=False)
                        new_cands[query_idx, candidates_idxs_to_be_replaced] = torch.from_numpy(positive_candidates)


            candidates_idxs_available = list(set(candidates_idxs_available) - set(candidates_idxs_to_be_replaced))
            inj_hard_negatives = (self.previous_epoch_candidates is not None) and self.inject_hard_negatives_candidates
            if inj_hard_negatives:
                # choose negatives from last epoch candidates because the faiss search thought they are similar (because their cosine difference is less) 
                # so we call them hard, and they can be good to enforce encoder embeding them far from the places they were
                prev_cands_idxs = self.previous_epoch_candidates[query_idx]
                # getting cuis of the candidates to get the negatives
                prev_dictionary_cuis = self.dictionary_cuis[prev_cands_idxs]
                neg_mask = prev_dictionary_cuis != query_cui
                # We will choose hard negatives from those indexes
                hard_negative_indexes = prev_cands_idxs[neg_mask]

                if len(hard_negative_indexes) > 0:
                    negatives_n = min(self.hard_negatives_num, len(hard_negative_indexes))
                    hard_negative_candidates = np.random.choice(hard_negative_indexes, size=negatives_n, replace=False)
                    # candidates_to_replace_positive
                    candidates_idxs_to_be_replaced = np.random.choice(candidates_idxs_available, size=negatives_n, replace=False)
                    new_cands[query_idx, candidates_idxs_to_be_replaced] = torch.from_numpy(hard_negative_candidates)

        return new_cands


    def change_candidates_pool_opt(self):
        assert self.all_candidates_idxs is not None, "Candidates are not set"

        if self.previous_epoch_candidates is None:
            return self.all_candidates_idxs

        num_queries, topk = self.all_candidates_idxs.shape
        new_cands = self.all_candidates_idxs.clone()
        dict_cuis = self.dictionary_cuis
        queries_cuis = self.queries_cuis

        for query_idx in range(num_queries):
            query_cui = queries_cuis[query_idx]
            current_idxs = new_cands[query_idx].numpy()
            current_cuis = dict_cuis[current_idxs]

            negative_mask = (current_cuis == query_cui)
            available_positions = np.flatnonzero(negative_mask)

            # Inject hard positives
            if self.inject_hard_positives_candidates:
                pos_dict_idxs = self.dictionary_cui_to_idx.get(query_cui, [])
                if len(pos_dict_idxs) > 0:
                    available_pos_dict = np.setdiff1d(pos_dict_idxs, current_idxs, assume_unique=False)
                    if len(available_pos_dict) > 0 and len(available_positions) > 0:
                        pos_n = min(self.hard_positives_num,
                                    len(available_pos_dict),
                                    len(available_positions))
                        chosen_pos_dict = np.random.choice(available_pos_dict, size=pos_n, replace=False)
                        chosen_slots = np.random.choice(available_positions, size=pos_n, replace=False)
                        new_cands[query_idx, chosen_slots] = torch.from_numpy(chosen_pos_dict)
                        available_positions = np.setdiff1d(available_positions, chosen_slots, assume_unique=False)

            if self.inject_hard_negatives_candidates and self.previous_epoch_candidates is not None:
                prev_idxs = np.array(self.previous_epoch_candidates[query_idx])
                prev_cuis = dict_cuis[prev_idxs]
                neg_candidates = prev_idxs[prev_cuis != query_cui]
                if len(neg_candidates) > 0 and len(available_positions) > 0:
                    neg_n = min(self.hard_negatives_num,
                                len(neg_candidates),
                                len(available_positions))
                    chosen_neg_dict = np.random.choice(neg_candidates, size=neg_n, replace=False)
                    chosen_slots = np.random.choice(available_positions, size=neg_n, replace=False)
                    new_cands[query_idx, chosen_slots] = torch.from_numpy(chosen_neg_dict)

        return new_cands

    def set_candidates(self,cands):
        self.all_candidates_idxs = torch.as_tensor(cands, dtype=torch.long)
        new_cands = self.change_candidates_pool_opt()
        self.previous_epoch_candidates = self.all_candidates_idxs.clone()
        self.all_candidates_idxs = new_cands




# =====================================
#       LOADING DATA
# ======================================

def get_annotated_query(s, mention_start, mention_end, mention, special_token_start, special_token_end,total_window_tokens=40):
    """
        args:
            s: full sentence containing the mention 
            mention_start, mention_end: char index of sentence where the mention starts and ends
            mention: the mention tokens
            total_window_tokens: how many tokens the annotated sentence would be
        returns:
            LEFT PART [MS] MENTION [ME] RIGHT PART
    """

    left_dot_pos = s.rfind('.', 0, mention_start)
    left_comma_pos = s.rfind(',', 0, mention_start)
    left_start_pos = max(left_dot_pos, left_comma_pos, 0)
    left_part = s[left_start_pos + 1:mention_start]

    right_dot_pos = s.find('.', mention_end)
    right_end = right_dot_pos + 1 if right_dot_pos > -1 else len(s)
    right_part = s[mention_end + 1: right_end]

    s_trunc = left_part.strip() + " " + mention.strip() + " " + right_part.strip()
    mention_start_new = len(left_part.strip()) + 1
    mention_end_new = len(left_part.strip())  + len(mention.strip()) + 1
    

    tokens = re.findall(r"\w+|[^\w\s]", s_trunc)
    tokens_offsets = []
    cursor = 0
    for t in tokens:
        start = s_trunc.find(t, cursor)
        tokens_offsets.append((start, start + len(t)))
        cursor = start + len(t)
    s_trunc[mention_start_new: mention_end_new]

    mention_token_start = next(i for i, (a, b) in enumerate(tokens_offsets) if a <= mention_start_new < b)
    mention_token_end = next(i for i, (a, b) in enumerate(tokens_offsets) if b >= mention_end_new) + 1


    max_left = int(total_window_tokens * 0.70)
    tokens_left = tokens[:mention_token_start ][:max_left]

    tokens_right = tokens[mention_token_end : ][:total_window_tokens - (len(tokens_left) + (mention_token_end - mention_token_start))]
    annotated = " ".join(tokens_left).strip() + " "  + special_token_start +  " " + mention + " " + special_token_end + " " + " ".join(tokens_right).strip()




    annotated = re.sub(r'\s+', ' ', annotated).strip()

    return annotated

def load_queries(data_dir, special_token_start="[MS]" , special_token_end="[ME]", total_window_tokens=50, filter_composite=True, filter_cuiless=True,filter_duplicate=True):
    data = []
    concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
    for concept_file in tqdm(concept_files):
        with open(concept_file, "r", encoding='utf-8') as f:
            concepts = f.readlines()

        for concept in concepts:
            concept = concept.split("||")
            document_number = concept[0]
            mention_start_idx, mention_end_idx = concept[1].split("|")
            mention_start_idx, mention_end_idx = int(mention_start_idx), int(mention_end_idx)
            semantic_type = concept[2].strip()
            mention = concept[3].strip()
            cui = concept[4].strip()
            is_composite = (cui.replace("+","|").count("|") > 0)

            # filter composite cui
            if filter_composite and is_composite:
                continue
            # filter cuiless
            if filter_cuiless and cui == '-1':
                continue

            txt_path = f"{data_dir}/{document_number}.txt"
            if not os.path.exists(txt_path):
                print(f"{txt_path} does not exists")
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            annotated = get_annotated_query(full_text, 
                                            mention_start_idx, 
                                            mention_end_idx, 
                                            mention, 
                                            special_token_start, 
                                            special_token_end,
                                            total_window_tokens=total_window_tokens)
            data.append((mention, cui, annotated, txt_path, mention_start_idx, mention_end_idx))
    if filter_duplicate:
        data = list(dict.fromkeys(data))
    data = np.array(data)
    return data

def load_dictionary(dictionary_path, special_token_start="[MS]" , special_token_end="[ME]"):
    data = []
    with open(dictionary_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if line == "": continue
            cui, name = line.split("||")
            name = special_token_start + " "  + name + " " + special_token_end
            data.append((name.strip(),cui))
    data = np.array(data)
    return data
