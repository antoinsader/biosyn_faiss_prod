import numpy as np
from config import GlobalConfig
from helpers.Data import TokensPaths, MyDataset

def check_dataset():
    cfg = GlobalConfig()
    tokens_paths = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')
    dataset = MyDataset(tokens_paths, cfg)
    
    print(f"Total queries: {len(dataset.queries_cuis)}")
    print(f"Total dictionary entries: {len(dataset.dictionary_cuis)}")
    
    no_positives_count = 0
    for i, query_cui in enumerate(dataset.queries_cuis):
        positives = dataset.dictionary_cui_to_idx.get(query_cui, [])
        # Filter out self if present (though usually query is not in dict in same index)
        # But here we just check if there are ANY positives in the dictionary
        if len(positives) == 0:
            no_positives_count += 1
            if no_positives_count <= 10:
                print(f"Query {i} (CUI {query_cui}) has NO positives in dictionary.")
    
    print(f"Total queries with no positives: {no_positives_count}")

if __name__ == "__main__":
    check_dataset()
