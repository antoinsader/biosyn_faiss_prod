
import gc,  os
import faiss 
import faiss.contrib.torch_utils
from main_classes import MyEncoder
import torch


import numpy as np
from tqdm import tqdm

from config import  GlobalConfig
from helpers.Data import TokensPaths, MyDataset


# ======================
# MY FAISS
# ======================
class MyFaiss():
    """
        Class responsible for:
            1- Create faiss index based on dictionary size 
                    If dictionary size is bigger than 1m, we chose the GpuIndexIVFPQ index with the help of IndexHNSWFlat quantizer
                    If dictionary size is less than 1m, we chose the GpuIndexFlatIP index or IndexFlatIP depending if cuda is available
            2- Train index on samples of dictionary in the case where the index is  ----
            3- Build the index based on dictionary embeddings
            4- Search through the index  for candidates (chosen from the dictionary embedings) of queries embedings
            5- Compute recall for the retreived candidates based on dictionary and queries cuis 
            6- Save the index in the file specified with save_index_path parameter
            7- Load the index from the file
            
            https://medium.com/%40statfusionai/different-types-vector-database-indexing-125cdc4ddc37
    """
    
    def __init__(self, cfg: GlobalConfig,  save_index_path, dataset: MyDataset, tokens_paths:TokensPaths, encoder : MyEncoder):
        self.cfg_faiss = cfg.faiss
        self.tokens_paths = tokens_paths
        self.encoder = encoder
        self.save_index_path = save_index_path
        self.dataset = dataset


        self.topk = cfg.train.topk
        self.hidden_size = cfg.model.hidden_size

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda"    if self.use_cuda else "cpu"
        self.use_amp = cfg.train.use_amp

        num_threads = min(32, os.cpu_count() or 8)
        faiss.omp_set_num_threads(num_threads)

        self.faiss_index = None
        self.dictionary_entries_n = None


    def _create_ivfpq_index(self, N):
        """
            IVFPQ index do clusters and needs to train on sample of dictionaries, after doing this function, we have to call self.train_samples()
        """
        assert self.use_cuda, 'You have more than 1M dictionary records, FAISS is not available with not cuda device'
        gpu_resources = faiss.StandardGpuResources()

        #bigger number means faster search and lower accuracy
        # now we are doing sqrt(N)
        num_clusters = self.cfg_faiss.num_clusters(N)

        #Default to 32
        # Means 768-d vector embeding would be split into 32 sub-vectors of 24d.
        # each sub-vector is encoded with nbit(def. 8bit) centroid id
        num_quantizers = self.cfg_faiss.num_quantizers
        nbits= self.cfg_faiss.nbits 

        #build hnsw graph where each node can be connected to 32 neighbors at most
        quantizer = faiss.IndexHNSWFlat(self.hidden_size, 32)
        quantizer.hnsw.efConstruction = self.cfg_faiss.hnsw_efConstruction
        #how many nodes in the graph are visited during search
        quantizer.hnsw.efSearch = self.cfg_faiss.hnsw_efSearch

        #Init the index
        index = faiss.GpuIndexIVFPQ(gpu_resources, quantizer, self.hidden_size, num_clusters, num_quantizers, nbits)

        # useFloat16LookupTables was removed in newer FAISS versions
        if hasattr(index, 'useFloat16LookupTables'):
            index.useFloat16LookupTables = self.use_amp
        #nprobe is the numbers of clusters to be visited during search, higher means more accurate but slower
        # 1-10% of nlist, we are using now 6%
        index.nprobe = self.cfg_faiss.n_probe(num_clusters)
        self.faiss_index = index

    def _create_flat_index(self):
        if self.use_cuda:
            gpu_resources = faiss.StandardGpuResources()
            index_conf = faiss.GpuIndexFlatConfig()
            index_conf.device = torch.cuda.current_device()
            index_conf.useFloat16 = bool(self.use_amp)
            self.faiss_index = faiss.GpuIndexFlatIP(gpu_resources, self.hidden_size, index_conf)
        else:
            self.faiss_index = faiss.IndexFlatIP(self.hidden_size)

    def init_index(self, N):
        if N >= 1_000_000 or self.cfg_faiss.force_ivfpq:
            self._create_ivfpq_index(N)
            self.train_samples(N)
        else:
            self._create_flat_index()



    def train_samples(self, N):
        """
            In case of IVFPQ index, the index need to be trained on samples from the dictionary entries
            Number of samples is token from cfg
        """
        assert self.faiss_index is not None

        dictionary_inputs_ids =   self.dataset.dictionary_input_ids
        dictionary_attention_masks = self.dataset.dictionary_attention_masks

        #sanity check
        assert dictionary_inputs_ids.shape[0] == dictionary_attention_masks.shape[0] == N, f"Something is wrong! N={N}, dtionary att shape is: {dictionary_attention_masks.shape}"


        sample_size= self.cfg_faiss.clusters_samples(self.cfg_faiss.num_clusters(N))
        #from N choose random sample_size indexes
        sample_indices = torch.randperm(N)[:sample_size]


        samples_batch_size = 4_000
        samples_embeds = torch.empty((sample_size, self.hidden_size), dtype=torch.float32)
        cursor = 0
        for start in tqdm(range(0, len(sample_indices), samples_batch_size),  desc="embed samples"):
            end = min(start+samples_batch_size, len(sample_indices))
            batch_idx = sample_indices[start:end]


            inp  = torch.as_tensor(dictionary_inputs_ids[batch_idx], device=self.device)
            att = torch.as_tensor(dictionary_attention_masks[batch_idx],device=self.device)
        
            batch_embeds = self.encoder.get_emb(inp, att, use_amp=self.use_amp, use_no_grad=True)
            batch_embeds = batch_embeds.contiguous()
            samples_embeds[cursor : cursor+(end-start)] = batch_embeds
            cursor += (end -start)
            del batch_embeds, inp, att
        del dictionary_attention_masks, dictionary_inputs_ids
        self.faiss_index.train(samples_embeds)
        del samples_embeds
        torch.cuda.empty_cache()
        gc.collect()


    def build_faiss(self, batch_size):
        """
            Build with batches the faiss dictionary, 
            dictionary_input_ids + dictionary_attention_masks ==> embed ==> index.add(emb)
        """
        N = self.tokens_paths.dictionary_shape[0]
        self.dictionary_entries_n  = N


        if self.faiss_index is None:
            self.init_index(N)
        else:
            self.faiss_index.reset()
        assert self.faiss_index is not None


        dictionary_inputs = self.dataset.dictionary_input_ids
        dictionary_att = self.dataset.dictionary_attention_masks

        for start in tqdm(range(0, N, batch_size), desc="Building faiss index"):
            end = min(start + batch_size, N)
            inp  = torch.as_tensor(dictionary_inputs[start:end], device=self.device)
            att = torch.as_tensor(dictionary_att[start:end],device=self.device)
            embs = self.encoder.get_emb(inp, att, use_amp=self.use_amp, use_no_grad=True)
            self.faiss_index.add(embs.contiguous())
            del inp, att, embs
        del dictionary_inputs, dictionary_att
        torch.cuda.empty_cache()
        gc.collect()

    def search_faiss(self, batch_size):
        """
            For each query_batch, search top-k candidates from the dictionary
            Return candidates (queries_num , topk)
        """
        assert self.faiss_index is not None, 'FAISS index has to be loaded from path or initialized'
        (tokens_size, max_length ) = self.tokens_paths.queries_shape
        N = tokens_size
        candidates = np.zeros((N,self.topk))
        faiss_index = self.faiss_index

        query_inputs = self.dataset.queries_input_ids
        query_att = self.dataset.queries_attention_mask

        for start in range(0, N,batch_size):
            end = min(start + batch_size, N)
            inp  = torch.as_tensor(query_inputs[start:end], device=self.device)
            att = torch.as_tensor(query_att[start:end],device=self.device)
            embs = self.encoder.get_emb(inp, att, use_amp=self.use_amp, use_no_grad=True)
            if self.use_cuda:
                embs = embs.contiguous()
            else:
                embs = embs.cpu().numpy().astype(np.float32)

            _, chunk_cand_idxs = faiss_index.search(embs, self.topk)
            if self.use_cuda:
                candidates[start:end] = chunk_cand_idxs.cpu().detach().numpy()
            else:
                candidates[start:end] = chunk_cand_idxs
            
            del inp, att, embs
        del query_inputs, query_att
        gc.collect()
        return candidates

    def compute_faiss_recall_at_k(self,candidates_idxs, k=10, k2=5):
        """
            For each query, there are topk candidates.
            How many queries have in their first k candidates the correct cui.
        """

        assert candidates_idxs is not None
        k = min(self.topk, k)
        k2 = min(self.topk, k2)

        queries_cuis = self.dataset.queries_cuis
        dictionary_cuis = np.array(self.dataset.dictionary_cuis)
        num_queries = len(queries_cuis)

        # Vectorized implementation
        # Get CUIs for all retrieved candidates: (num_queries, k)
        retrieved_candidates_cuis = dictionary_cuis[candidates_idxs[:, :k]]
        retrieved_candidates_cuis_k2 = dictionary_cuis[candidates_idxs[:, :k2]]
        
        # Check if query_cui is present in each row of retrieved candidates
        # queries_cuis[:, None] adds a dimension to make it (num_queries, 1) for broadcasting
        matches = (retrieved_candidates_cuis == queries_cuis[:, None])
        matches_k2 = (retrieved_candidates_cuis_k2 == queries_cuis[:, None])
        
        # A query is correct if there is at least one match in the top k
        correct = np.any(matches, axis=1).sum()
        correct_k2 = np.any(matches_k2, axis=1).sum()

        return correct / max(num_queries, 1), correct_k2 / max(num_queries, 1)


    def load_faiss_index(self, path):
        """
            Read FAISS index from path
        """
        assert os.path.exists(path),f'Path faiss {path} not exists'
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.read_index(path)
        if self.use_cuda:
            co = faiss.GpuClonerOptions()
            co.allowCpuCoarseQuantizer = True
            index = faiss.index_cpu_to_gpu(gpu_resources, 0 , index, co)

        self.faiss_index = index

    def save_index(self):
        """
            Save index to save_index_path
        """
        faiss.write_index(faiss.index_gpu_to_cpu(self.faiss_index), self.save_index_path)
        print(f'FAISS Index saved to {self.save_index_path}')
        return self.save_index_path

