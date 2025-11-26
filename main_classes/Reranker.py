import torch
import torch.nn as nn
import torch.optim as optim


from config import  GlobalConfig
from helpers.utils import  marginal_nll
from main_classes.MyEncoder import MyEncoder

# =======================
#       Reranker
#========================

class Reranker(nn.Module):
    """
        This is our model that we will use
        We are using optimization function AdamW
        We are using marginal_nll as the loss function (we tried info_nce_loss as well)
        We are training transformer encoder on constrastive retreival loss with normalized embeding and cosine similarity
        
    """
    def __init__(self, encoder : MyEncoder,  cfg:GlobalConfig ):
        super().__init__()

        self.cfg = cfg.train
        self.encoder = encoder
        self.criterion =  marginal_nll

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        assert self.cfg.optimizer_name == 'AdamW', f'Currently only AdamW available'

        params = list(self.encoder.encoder.parameters())
        if self.encoder.projection is not None:
            params += list(self.encoder.projection.parameters())

        self.optimizer = optim.AdamW(
            params,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            fused=self.use_cuda
        )


    def forward(self, query_tokens, candidate_tokens=None, cand_embeddings=None):
        """
        query_tokens: dict of input_ids and attention_mask (B, L)
        candidate_tokens: dict of input_ids and attention_mask (B, K, L) - Optional
        cand_embeddings: (B, K, H) - Optional
        """
        # Embed Query
        q_emb = self.encoder.get_emb(query_tokens) # (B, H)

        if cand_embeddings is not None:
             # Use cached embeddings
             c_emb = cand_embeddings
        else:
            # Encode Candidates
            B, K, L = candidate_tokens['input_ids'].shape
            
            # Flatten candidates
            cand_input_ids = candidate_tokens['input_ids'].view(B * K, L)
            cand_attention_mask = candidate_tokens['attention_mask'].view(B * K, L)
            
            cand_tokens_flat = {
                "input_ids": cand_input_ids,
                "attention_mask": cand_attention_mask
            }
            
            c_emb = self.encoder.get_emb(cand_tokens_flat) # (B*K, H)
            
            # Reshape candidates
            c_emb = c_emb.view(B, K, -1) # (B, K, H)
        
        # Compute Scores (Cosine Similarity)
        # q_emb: (B, 1, H)
        # c_emb: (B, K, H) -> transpose -> (B, H, K)
        scores = torch.bmm(q_emb.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1) # (B, K)
        
        return scores
    
    def get_loss(self, outputs, targets):
        """
            outputs has shape (batch_size, topk)
            targets if marginal_nll then (batch_size, topk) if other (batch_size)
        """
        if self.use_cuda:
            targets = targets.cuda()
            outputs = outputs.cuda()

        outputs = outputs / self.cfg.loss_temperature
        loss = self.criterion(outputs, targets)
        return loss
