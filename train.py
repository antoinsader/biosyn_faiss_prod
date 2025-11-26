
# python train.py --training_log_name='big dictionary' --hard_positives_num=0 --hard_negatives_num=0 --num_epochs=6  --topk=50 --train_batch_size=32 --enable_gradient_checkpoint

import logging
import torch
import gc
import os
import time
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

from main_classes.MyFaiss import MyFaiss
from main_classes.MyEncoder import MyEncoder
from main_classes.Reranker import Reranker 
from helpers.Data import TokensPaths, MyDataset
from helpers.MyLogger import MyLogger, CheckPointing
from helpers.utils import compute_metrics, save_pkl
from config import GlobalConfig, train_parse_args

class Trainer:
    """
        This class is responsible for initiating instances of the main classes that we will use in our training 
    """
    def __init__(self,  my_logger: MyLogger, checkpointing: CheckPointing, cfg:GlobalConfig):
        self.cfg = cfg
        self.result_encoder_dir = cfg.paths.result_encoder_dir
        self.topk = cfg.train.topk
        self.faiss_path  = cfg.paths.faiss_path
        self.logger = my_logger
        self.checkpointing = checkpointing

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda"    if self.use_cuda else "cpu"

        dictionary_key='dictionary'
        if cfg.train.use_small_dictionary:
            dictionary_key = 'small_dictionary'
        self.tokens_paths = TokensPaths(cfg, dictionary_key=dictionary_key, queries_key='train_queries')
        self.encoder = MyEncoder(cfg)
        self.model = Reranker(self.encoder, self.cfg)
        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.dataset = MyDataset(self.tokens_paths, cfg)
        self.faiss = MyFaiss(cfg, save_index_path=self.faiss_path,  dataset=self.dataset, tokens_paths=self.tokens_paths, encoder=self.encoder)
        self.chkpoint_path = cfg.paths.checkpoint_path



        self.scaler = torch.amp.GradScaler(enabled=cfg.train.use_amp)
        num_training_steps = len(self.dataset) // self.cfg.train.batch_size * self.cfg.train.num_epochs
        num_warmup_steps = int(0.05 * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.model.optimizer,
            num_warmup_steps= num_warmup_steps,
            num_training_steps=num_training_steps
        )



    def train_one_batch(self, data_loader_item, batch_idx):
        """
            What we will do in one batch is:
                - extracting batch_x, batch_y from the dataloader item
                - Extract batch_query_tokens, batch_candidates_tokens from batch_x
                - Forward pass in the reranker and getting scores. (scores shape is batch_size, topk which is cosine similarity between each query and its candidates)
                - we calculate the loss based on the scores, we use marginal_nll 
                - We do backpropogation, scale, optimize, update and step
                - We calculate accuracy and mrr metrics for the batch
        """
        # self.model.optimizer.zero_grad(set_to_none=True) # Moved to step logic
        
        with torch.amp.autocast(device_type="cuda", enabled=(self.use_cuda and self.cfg.train.use_amp)):
            batch_x, batch_y = data_loader_item
            batch_query_tokens, batch_candidates_tokens = batch_x
            
            cand_embeddings = None
            if "cand_embeddings" in batch_candidates_tokens:
                 cand_embeddings = batch_candidates_tokens["cand_embeddings"].to(self.device, non_blocking=True)
            
            batch_scores = self.model(batch_query_tokens, batch_candidates_tokens, cand_embeddings=cand_embeddings)
            loss = self.model.get_loss(batch_scores, batch_y)

        # Scale loss
        loss = loss / self.cfg.train.gradient_accumulation_steps
        self.scaler.scale(loss).backward()
        
        # Step only after accumulation steps
        if (batch_idx + 1) % self.cfg.train.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.model.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.encoder.encoder.parameters(), max_norm=1.0)

            self.scaler.step(self.model.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.model.optimizer.zero_grad(set_to_none=True)


        # Compute metrics only at intervals
        if (batch_idx + 1) % self.cfg.train.metric_compute_interval == 0:
            accuracy_5, mrr, batch_margin = compute_metrics(batch_scores, batch_y, k=5)
            return loss.detach() * self.cfg.train.gradient_accumulation_steps, accuracy_5, mrr, batch_margin
        
        # Return None for metrics if not computed
        return loss.detach() * self.cfg.train.gradient_accumulation_steps, None, None, None



    def train_one_epoch(self, epoch):
        """
            Train one epoch:
                1-  freeze lower layers depending on epoch size and cfg.train.freeze_lower_layer_epoch_max
                2- build faiss index using self.faiss instance
                3- Search for candidates of the queries using faiss
                4- set the candidates in the dataset instance and inject hard negative and positive candidates if allowed
                5- compute FAISS recall for k=topk
                6- make the dataloader from the dataset
                7- for each batch in the dataloader execute self.train_one_batch
                8- calculate accuracy mrr and loss from the returned values of train_one_batch
                9- log epoch summary
        """
        torch.cuda.empty_cache()
        gc.collect()
        

        self.encoder.unfreeze_all()
        if epoch <= self.cfg.train.freeze_lower_layer_epoch_max:
            self.encoder.freeze_lower_layers(max(0, 7 - epoch ))

        
        

        # Build FAISS and optionally get embeddings
        embeddings = self.faiss.build_faiss(
            self.cfg.faiss.build_batch_size, 
            return_embeddings=self.cfg.train.use_cached_candidates
        )
        
        if self.cfg.train.use_cached_candidates:
            self.logger.log_event("Caching Candidates", message="Setting dictionary embeddings in dataset", epoch=epoch, log_memory=True)
            self.dataset.set_dict_embeddings(embeddings)
        else:
            if embeddings is not None: del embeddings
            self.dataset.set_dict_embeddings(None)

        faiss_search_start_time = time.time()
        candidates = self.faiss.search_faiss(self.cfg.faiss.search_batch_size)
        self.logger.log_event("Faiss search finished", message=f"Candidates shape: {candidates.shape}", t0=faiss_search_start_time, log_memory=True, epoch=epoch)
        
        self.dataset.set_candidates(candidates)
        
        recall_faiss, recall_faiss_k2 = self.faiss.compute_faiss_recall_at_k(candidates, k=self.topk, k2=5)
        self.logger.log_event("Faiss recall", message=f"Recall@{self.topk}: {recall_faiss:.4f}, Recall@5: {recall_faiss_k2:.4f}", epoch=epoch)


        my_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            pin_memory=self.use_cuda,
            num_workers=self.cfg.train.num_workers,
            persistent_workers=(self.cfg.train.num_workers > 0),  # Keep workers alive between batches
            prefetch_factor=2 if self.cfg.train.num_workers > 0 else None  # Prefetch 2 batches per worker
        )
        batches_train_start_time = time.time()
        epoch_loss, epoch_accuracy_5, epoch_mrr, epoch_margin = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        self.model.optimizer.zero_grad(set_to_none=True) # Ensure grads are zero at start of epoch
        
        n_metric_batches = 0
        epoch_loss_tensor = 0.0
        for i, data_loader_item in tqdm(enumerate(my_loader), total=len(my_loader), desc=f"epoch@{epoch} - Training batches"):
            loss, accuracy_5, mrr, batch_margin = self.train_one_batch(data_loader_item, i)
            
            if torch.is_tensor(loss):
                epoch_loss_tensor += loss.detach()
            else:
                epoch_loss_tensor += loss
            
            if accuracy_5 is not None:
                epoch_accuracy_5 += accuracy_5
                epoch_mrr += mrr
                epoch_margin += batch_margin
                n_metric_batches += 1
            
            n_batches += 1

            # if i % 100 == 0:
            #     self.logger.log_event(f"Training stats - batch {i}", message=f"Loss: {loss.item():.4f}", log_memory=True, epoch=epoch)

        if torch.is_tensor(epoch_loss_tensor):
            epoch_loss = epoch_loss_tensor.item()
        else:
            epoch_loss = epoch_loss_tensor

        avg_loss = (epoch_loss / max(1, n_batches))
        avg_mrr = (epoch_mrr / max(1, n_metric_batches))
        avg_accuracy_5 = (epoch_accuracy_5 / max(1, n_metric_batches))
        avg_margin = (epoch_margin / max(1, n_metric_batches))  # Average

        self.logger.log_event(
            "Epoch summary",
            message=f"Epoch average loss={avg_loss:.3f}, average margin={avg_margin:.4f}, average mrr={avg_mrr:.4f}, average accuracy@5={avg_accuracy_5:.4f}.   loss_temp={self.cfg.train.loss_temperature:.3f}",
            epoch=epoch,
            t0=batches_train_start_time
        )


        del my_loader
        torch.cuda.empty_cache()
        gc.collect()
        return avg_loss, avg_mrr, avg_accuracy_5, recall_faiss




    def restore_chkpoint(self):
        chkpt = self.checkpointing.restore_checkpoint()
        self.model.encoder.load_state(chkpt['model_state'])
        self.model.optimizer.load_state_dict(chkpt['optimizer_state'])
        self.scheduler.load_state_dict(chkpt['scheduler_state'])
        self.scaler.load_state_dict(chkpt['scaler_state'])
        self.faiss.load_faiss_index(chkpt['faiss_index_path'])
        return chkpt["epoch"] + 1

    def save_checkpoint(self,epoch):
        model_state = {
            "encoder": self.model.encoder.get_state_dict(),
        }
        if self.model.encoder.projection is not None:
             model_state["projection"] = self.model.encoder.projection.state_dict()

        ckpt = {
            "epoch": epoch,
            "model_state": model_state,
            
            "optimizer_state": self.model.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "faiss_index_path": self.faiss.save_index(),
        }
        self.checkpointing.save_checkpoint(ckpt)
        return True


    def train(self):
        """
            1- restore checkpoint if cfg.train.load_last_checkpoint is True and if last checkpoint file exists
            2- Call epoch training and save checkpoint, num_epochs is set in cfg.train.num_epochs
        """


        full_train_start_time = time.time()

        # ===================================================
        #       1- RESTORE CHECKPOINT
        # ===================================================
        start_epoch = 1
        if self.cfg.train.load_last_checkpoint:
            if os.path.exists(self.chkpoint_path):
                self.logger.log_event("checkpoint resotred", f"Restoring checkpoint from file: {self.chkpoint_path}", log_memory=False)
                start_epoch = self.restore_chkpoint()


        # ===================================================
        #       2- EPOCH TRAINING
        # ===================================================
        avergae_loss, average_mrr, average_accuracy_5, last_faiss_recall = 0.0, 0.0, 0.0, 0.0
        assert int(start_epoch) > 0 and int(start_epoch) <= self.cfg.train.num_epochs 
        for epoch in range(start_epoch, self.cfg.train.num_epochs + 1):
            if self.use_cuda:
                torch.cuda.reset_peak_memory_stats()
            # self.cfg.train.loss_temperature = max(0.05, 0.15 * (0.88 ** (epoch - 1)))

            avergae_loss, average_mrr, average_accuracy_5, last_faiss_recall = self.train_one_epoch(epoch)
            if self.cfg.train.save_checkpoints:
                self.save_checkpoint(epoch)



        # ===================================================
        #       3- LOG TRAINING STATS AND SAVE ENCODER
        # ===================================================
        training_time = time.time() - full_train_start_time
        training_time_str = f"{int(training_time/60/60)}h, {int(training_time/60 % 60)}mins, {int(training_time % 60)}secs"

        self.checkpointing.log_finished(
            queries_len=len(self.dataset.queries_cuis),
            dict_len=len(self.dataset.dictionary_cuis),
            training_time_str=training_time_str,
            last_acc_5 = average_accuracy_5,
            last_mrr = average_mrr,
            last_faiss_recall= last_faiss_recall,
            last_loss=avergae_loss
        )

        self.encoder.save_state(self.result_encoder_dir)
        self.save_checkpoint(epoch='last')
        self.faiss.save_index()

    

        self.logger.log_event("Train finished", t0=full_train_start_time, log_memory=False)
        self.logger.log_event("Main info: " , message=self.checkpointing.current_experiment, log_memory=False)


LOGGER = logging.getLogger()
if __name__ == '__main__':
    cfg : GlobalConfig = train_parse_args()
    cfg.logger.tag = "TRAIN"

    checkpointing = CheckPointing(cfg)
    logger = MyLogger(LOGGER , checkpointing.current_experiment_log_path, cfg.logger.tag )
    trainer = Trainer(logger, checkpointing, cfg)

    trainer.train()

    result_encoder_dir = cfg.paths.result_encoder_dir 

    torch.cuda.empty_cache()
    gc.collect()

    print(f"TRAIN FINISHED: result encoder dir: {result_encoder_dir}")


