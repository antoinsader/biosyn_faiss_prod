import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModel

from config import GlobalConfig



class MyEncoder():
    """
        This class is responsible for getting the embeddings of tokens
        All the embeding for FAISS sampling + build + search and training forward pass are using this encoder
        The objective of the project is to fine-tune this encoder to embed similar entities near each other

        The encoder would be initialized from the model_name set in config
        If you're using special tokens in the tokenizer, the encoder should be extended, this process is done through the meta_data of the tokenizer
        You will get the meta data of the tokenizer from the config.paths.tokenizer_meta_path (this should be saved in the tokenizer process) 
    """
    def __init__(self, cfg:GlobalConfig):
        self.cfg = cfg.model

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        encoder = AutoModel.from_pretrained(self.cfg.model_name, use_safetensors=True)

        tokenizer_meta_path  = cfg.paths.tokenizer_meta_path
        with open(tokenizer_meta_path) as f:
            tokenizer_meta = json.load(f)
        tokenizer_len = tokenizer_meta['len_tokenizer']
        encoder.resize_token_embeddings(tokenizer_len)

        self.mention_start_token_id = tokenizer_meta["mention_start_token_id"]
        self.mention_end_token_id = tokenizer_meta["mention_end_token_id"]

        self.encoder = encoder.to(self.device)
        self.cfg.hidden_size = self.encoder.config.hidden_size

    def get_emb(self, input_ids_tensor, attention_mask_tensor, use_amp=False, use_no_grad=False):
        """
            The parameters are input_ids and attention_mask should be tensors
            use_amp if set to true the encoder will embed with fp16 (faster if available)
            use_no_grad if True, torch will save the gradient graph (only use in training forward pass, otherwise set to False)
        
          cfg.normalize would normalize the result.
                It is better to normalize all embedings so that the inner product became cosine similarity
        """
        context = torch.inference_mode() if use_no_grad else torch.enable_grad()
        with context, torch.amp.autocast(device_type="cuda", enabled=(self.use_cuda and use_amp)):
            # Hidden state, (batch, seq_len, hidden)
            emb = self.encoder(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)[0]


        batch_size, seq_len, hidden_size = emb.size()
        embs = torch.zeros((batch_size, hidden_size), device=self.device)
        for i in range(batch_size):
            input_ids = input_ids_tensor[i]

            mention_start_positions = (input_ids == self.mention_start_token_id).nonzero(as_tuple=True)[0]
            mention_end_positions = (input_ids == self.mention_end_token_id).nonzero(as_tuple=True)[0]

            assert len(mention_start_positions) > 0 and len(mention_end_positions) > 0, f"No markers found"
            mention_start_idx = mention_start_positions[0].item()
            mention_end_idx = mention_end_positions[0].item()

            assert mention_end_idx > mention_start_idx + 1, f"Malformed span !!"
            span_emb = emb[i, mention_start_idx + 1: mention_end_idx] # (span len, hidden)
            embs[i] = span_emb.mean(dim=0)



        if self.cfg.normalize:
            ret = F.normalize(embs , p=2, dim=1)

        return ret

    def freeze_lower_layers(self, num_layers_to_freeze=6):
        """
        freeze first num_layers_to_freeze encoder layers
        We are using this at first epochs of the training where we freeze lower layers in order to warm up the model
        """
        for name, param in self.encoder.named_parameters():
            if "encoder.layer." in name:
                layer_id = int(name.split(".")[2])
                param.requires_grad = layer_id >= num_layers_to_freeze

    def unfreeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def save_state(self, dir):
        """
        This is used at checkpointing where after each epoch, we are saving the state of the encoder
        """
        os.makedirs(dir, exist_ok=True)
        self.encoder.save_pretrained(dir)
        return True

    def load_state(self, state):
        """
            For restoring checkpoint
        """
        self.encoder.load_state_dict(state)

    def get_state_dict(self):
        return self.encoder.state_dict()
