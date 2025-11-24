
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



    def train_one_batch(self, data_loader_item):
        """
            What we will do in one batch is:
                - extracting batch_x, batch_y from the dataloader item
                - Extract batch_query_tokens, batch_candidates_tokens from batch_x
                - Forward pass in the reranker and getting scores. (scores shape is batch_size, topk which is cosine similarity between each query and its candidates)
                - we calculate the loss based on the scores, we use either marginal_nll (better in our case) or info_nce_loss
                - We do backpropogation, scale, optimize, update and step
                - We calculate accuracy and mrr metrics for the batch
        """
        self.model.optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(self.use_cuda and self.cfg.train.use_amp)):
            batch_x, batch_y = data_loader_item
            batch_query_tokens, batch_candidates_tokens = batch_x
            batch_scores = self.model(batch_query_tokens, batch_candidates_tokens)
            loss = self.model.get_loss(batch_scores, batch_y)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.model.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.encoder.encoder.parameters(), max_norm=1.0)


        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.scheduler.step()



        accuracy_5, mrr, batch_margin = compute_metrics(batch_scores, batch_y, k=5)


        del batch_x, batch_y, batch_scores
        return loss.detach(), accuracy_5, mrr, batch_margin



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



        # ====================================
        #       BUILD FAISS
        # ====================================
        build_faiss_start_time = time.time()
        self.faiss.build_faiss(self.cfg.faiss.build_batch_size)
        self.logger.log_event("FAISS BUILD", message="FAISS built from dictionary embs",
                 epoch=epoch, t0=build_faiss_start_time)



        # ====================================
        #       SEARCH FAISS
        # ====================================
        search_faiss_start_time = time.time()
        candidates_idxs = self.faiss.search_faiss(self.cfg.faiss.search_batch_size) # (queries_N, topk)
        candidates_idxs = candidates_idxs.astype(np.int64)
        self.logger.log_event("FAISS SEARCH", message=f"FAISS searched the candidates and resulted candidates_idxs with shape={candidates_idxs.shape}", t0=search_faiss_start_time)



        self.dataset.set_candidates(candidates_idxs)
        recall_faiss = self.faiss.compute_faiss_recall_at_k(candidates_idxs, k=self.topk)
        self.logger.log_event(f"Faiss recall@{self.topk}", message= f"{recall_faiss:.4f}", epoch=epoch, log_memory=False)



        my_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            pin_memory=self.use_cuda,
            num_workers=self.cfg.train.num_workers,
            persistent_workers=False
        )
        batches_train_start_time = time.time()
        epoch_loss, epoch_accuracy_5, epoch_mrr, epoch_margin = 0.0, 0.0, 0.0, 0.0
        n_batches = 0


        for i, data_loader_item in tqdm(enumerate(my_loader), total=len(my_loader), desc=f"epoch@{epoch} - Training batches"):
            loss, accuracy_5, mrr, batch_margin = self.train_one_batch(data_loader_item)
            epoch_loss += loss.item()
            epoch_accuracy_5 += accuracy_5
            epoch_mrr += mrr
            epoch_margin += batch_margin
            n_batches += 1
            if i % 100 == 0:
                self.logger.log_event(f"Training stats - batch {i}", message=f"Loss: {loss.item():.4f}", log_memory=True, epoch=epoch)


        avg_loss = (epoch_loss / max(1, n_batches))
        avg_mrr = (epoch_mrr / max(1, n_batches))
        avg_accuracy_5 = (epoch_accuracy_5 / max(1, n_batches))
        avg_margin = (epoch_margin / max(1, n_batches))  # Average

  
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
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.encoder.get_state_dict(),
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
        assert int(start_epoch) > 0 and int(start_epoch) < self.cfg.train.num_epochs 
        for epoch in range(start_epoch, self.cfg.train.num_epochs + 1):
            if self.use_cuda:
                torch.cuda.reset_peak_memory_stats()
            self.cfg.train.loss_temperature = max(0.05, 0.15 * (0.88 ** (epoch - 1)))

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


