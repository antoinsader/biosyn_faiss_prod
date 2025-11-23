import torch
import torch.nn as nn
import torch.optim as optim


from config import  GlobalConfig
from helpers.utils import  info_nce_loss, marginal_nll
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
        self.criterion = info_nce_loss if self.cfg.loss_type == 'info_nce_loss' else marginal_nll

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        assert self.cfg.optimizer_name == 'AdamW', f'Currently only AdamW available'

        self.optimizer = optim.AdamW(
            self.encoder.encoder.parameters() + list(self.encoder.projection.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            fused=self.use_cuda
        )


    def forward(self, query_tokens, candidates_tokens):
        """
        What we are doing in forward pass is:
            1- we are getting as args the batch_query_tokens and batch_candidate_tokens for those queries
                    query_tokens has shape (batch_size, max_length)
                    candidate_tokens has shape (batch_size, topk, max_length)
            2- We are transforming the candidate_tokens into shape (batch_size * topk, max_length)
            3- We are embeding the query_tokens and candidates_tokens (we are normalizing those embedings inside the get_emb() function of the encoder)
            4- We convert candidates_embeddings from shape (batch_size * topk, hidden_size) into shape (batch_size, topk, hidden_size)
            5- We convert query_embeddings from shape(batch_size, hidden_size)
            6- We use torch.bmm to calculate the score (coosine similarity)
        Scores shape is (batch_size, topk), for each query, what is the cosine similarity with its candidates
        """

        # MOVE TO CUDA
        if self.use_cuda:
            candidates_tokens["input_ids"] = candidates_tokens["input_ids"].to("cuda", non_blocking=True)
            candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].to("cuda", non_blocking=True)
            query_tokens["input_ids"] = query_tokens["input_ids"].to("cuda", non_blocking=True)
            query_tokens["attention_mask"] = query_tokens["attention_mask"].to("cuda", non_blocking=True)

        batch_size, topk, max_length = candidates_tokens["input_ids"].size()

        # TRANSFER SHAPE
        candidates_tokens["input_ids"] = candidates_tokens["input_ids"].view(batch_size * topk, max_length)
        candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].view(batch_size * topk, max_length)

        #   EMBEDING
        #(batch_size, hidden_size)
        query_embeddings = self.encoder.get_emb(query_tokens["input_ids"], query_tokens["attention_mask"], use_amp=self.cfg.use_amp, use_no_grad=False)
        #(batch_size * topk , hidden_size)
        candidates_embeddings = self.encoder.get_emb(candidates_tokens["input_ids"], candidates_tokens["attention_mask"], use_amp=self.cfg.use_amp, use_no_grad=False)


        # TRANSFER SHAPE
        candidates_embeddings = candidates_embeddings.view(batch_size, topk, -1)
        query_embeddings = query_embeddings.unsqueeze(1) # [batch_size, 1, hidden_size]

        # CALCULATE SCORE 
        score = torch.bmm(query_embeddings, candidates_embeddings.transpose(1, 2)).squeeze(1) #batch_size, topk
        del candidates_embeddings, query_embeddings
        #score (batch_size, topk)
        return score
    
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
