import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel

from config import GlobalConfig, EncodingType



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
        
        
        if not os.path.exists(tokenizer_meta_path):
            if os.path.exists(os.path.join(cfg.paths.result_encoder_dir, "tokenizer_meta.json") ):
                tokenizer_meta_path = os.path.join(cfg.paths.result_encoder_dir, "tokenizer_meta.json")
            else:
                raise Exception("tokenizer_meta_path does not exist")
    
        with open(tokenizer_meta_path) as f:
            tokenizer_meta = json.load(f)
        self.tokenizer_meta = tokenizer_meta
        tokenizer_len = tokenizer_meta['len_tokenizer']
        encoder.resize_token_embeddings(tokenizer_len)
        self.mention_start_token_id = tokenizer_meta["mention_start_token_id"]
        self.mention_end_token_id = tokenizer_meta["mention_end_token_id"]


        self.encoder = encoder.to(self.device)
        self.hidden_size = self.encoder.config.hidden_size
        self.cfg.hidden_size = self.hidden_size

        # Initialize projection only if needed
        self.projection = None
        if self.cfg.encoding_type in [EncodingType.CLS_AND_BETWEEN_SPANS, EncodingType.CLS_AND_SPANS]:
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.projection.to(self.device)

    def get_emb(self, input_ids_tensor, attention_mask_tensor, use_amp=False, use_no_grad=False):
        """
            The parameters are input_ids and attention_mask should be tensors
            use_amp if set to true the encoder will embed with fp16 (faster if available)
            use_no_grad if True, torch will save the gradient graph (only use in training forward pass, otherwise set to False)
          cfg.normalize would normalize the result.
                It is better to normalize all embedings so that the inner product became cosine similarity
        """

        if not torch.is_tensor(input_ids_tensor):
            input_ids_tensor = torch.as_tensor(input_ids_tensor, device = self.device)

        
        context = torch.inference_mode() if use_no_grad else torch.enable_grad()
        with context, torch.amp.autocast(device_type="cuda", enabled=(self.use_cuda and use_amp)):
            # (batch, seq_len, hidden)
            outputs = self.encoder(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            last_hidden_state = outputs.last_hidden_state

            # extract [cls]
            cls_emb =  last_hidden_state[:, 0, :]

            # Find start and end indices
            # (batch_size, )
            mention_start_indices = (input_ids_tensor == self.mention_start_token_id).float().argmax(dim=1)
            mention_end_indices = (input_ids_tensor == self.mention_end_token_id).float().argmax(dim=1)

            embs = None

            if self.cfg.encoding_type == EncodingType.CLS_ONLY:
                embs = cls_emb

            elif self.cfg.encoding_type == EncodingType.BETWEEN_SPANS or self.cfg.encoding_type == EncodingType.CLS_AND_BETWEEN_SPANS:
                # extract span between [MS] and [ME]
                batch_size, seq_len, _ = last_hidden_state.size()
                
                # Create range tensor for masking
                seq_range = torch.arange(seq_len, device=self.device).unsqueeze(0)
                
                # Create mask: start < index < end
                mask = (seq_range > mention_start_indices.unsqueeze(1)) & (seq_range < mention_end_indices.unsqueeze(1))
                
                # Compute mean
                mask_expanded = mask.unsqueeze(-1).float()
                sum_embs = (last_hidden_state * mask_expanded).sum(dim=1)
                counts = mask_expanded.sum(dim=1)
                counts = torch.clamp(counts, min=1.0)
                span_embs = sum_embs / counts

                if self.cfg.encoding_type == EncodingType.BETWEEN_SPANS:
                    embs = span_embs
                else: # CLS_AND_BETWEEN_SPANS
                    combined_embs = torch.cat((cls_emb, span_embs), dim=1)
                    embs = self.projection(combined_embs)

            elif self.cfg.encoding_type == EncodingType.SPANS_ONLY or self.cfg.encoding_type == EncodingType.CLS_AND_SPANS:
                # Extract embeddings of [MS] and [ME] tokens
                # gather expects index to have same dims as input except on gathered dim
                # last_hidden_state: (batch, seq, hidden)
                # indices: (batch, ) -> (batch, 1, hidden)
                
                # We need to gather along dim 1 (seq_len)
                # indices must be (batch, 1, hidden) to gather (batch, seq, hidden) ?? No.
                # torch.gather(input, dim, index)
                # index should have same number of dims.
                
                # Easier way:
                # mention_start_indices: (batch,)
                # We want to select specific sequence indices for each batch element.
                # last_hidden_state[torch.arange(batch_size), mention_start_indices] works
                
                batch_indices = torch.arange(last_hidden_state.size(0), device=self.device)
                ms_embs = last_hidden_state[batch_indices, mention_start_indices]
                me_embs = last_hidden_state[batch_indices, mention_end_indices]
                
                # Mean of [MS] and [ME]
                spans_mean_embs = (ms_embs + me_embs) / 2.0

                if self.cfg.encoding_type == EncodingType.SPANS_ONLY:
                    embs = spans_mean_embs
                else: # CLS_AND_SPANS
                    combined_embs = torch.cat((cls_emb, spans_mean_embs), dim=1)
                    embs = self.projection(combined_embs)

            if self.cfg.normalize:
                embs = F.normalize(embs, p=2, dim=1)

        return embs


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
        
        if self.projection is not None:
            torch.save(self.projection.state_dict(), os.path.join(dir, "projection.pth"))
                
        with open(os.path.join(dir, "tokenizer_meta.json"), "w") as f:
            json.dump(self.tokenizer_meta, f)

        return True

    def load_state(self, state):
        """
            For restoring checkpoint
            Args:
                state: Either a dict with 'encoder' and 'projection' keys (from checkpoint),
                       or a str path to a directory containing saved model files
        """
        if isinstance(state, dict):
            # Loading from checkpoint dictionary (during training)
            self.encoder.load_state_dict(state['encoder'])
            if self.projection is not None and 'projection' in state:
                self.projection.load_state_dict(state['projection'])
        else:
            # Loading from saved model directory (during eval/inference)
            assert isinstance(state, str), "state must be either dict or str"
            assert os.path.exists(state), f"Model directory does not exist: {state}"
            
            # Load encoder from saved pretrained model (handles config.json and model.safetensors)
            loaded_encoder = AutoModel.from_pretrained(state, use_safetensors=True)
            self.encoder.load_state_dict(loaded_encoder.state_dict())
            
            # Load projection weights if needed
            if self.projection is not None:
                projection_path = os.path.join(state, "projection.pth")
                if os.path.exists(projection_path):
                    self.projection.load_state_dict(torch.load(projection_path, map_location=self.device))
                else:
                    print(f"Warning: Projection file not found at {projection_path}, but projection layer is initialized.")


    def get_state_dict(self):
        return self.encoder.state_dict()
