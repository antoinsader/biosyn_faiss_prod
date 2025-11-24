import faiss
import torch 
import numpy as np
from config import GlobalConfig
from helpers.Data import TokensPaths, MyDataset

import logging

from helpers.utils import compute_metrics
from main_classes.Reranker import Reranker
from main_classes.MyEncoder import MyEncoder
from main_classes.MyFaiss import MyFaiss
from helpers.MyLogger import CheckPointing
from helpers.MyLogger import MyLogger
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

LOGGER = logging.getLogger()

cfg = GlobalConfig()
cfg.logger.tag = "TRAIN"
checkpointing = CheckPointing(cfg)
logger = MyLogger(LOGGER , checkpointing.current_experiment_log_path, cfg.logger.tag )


topk = 15
batch_size = 16
cfg.train.num_epochs = 8
cfg.train.topk = topk
cfg.train.batch_size = batch_size

tokens_paths  = TokensPaths(cfg, dictionary_key='dictionary', queries_key='train_queries')

my_encoder = MyEncoder(cfg)
dataset = MyDataset(tokens_paths, cfg)
faiss_path = cfg.paths.faiss_path
faiss = MyFaiss(cfg, save_index_path=faiss_path,  dataset=dataset, tokens_paths=tokens_paths, encoder=my_encoder)
model = Reranker(my_encoder, cfg)



num_training_steps = len(dataset) // cfg.train.batch_size * cfg.train.num_epochs
num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
            model.optimizer,
            num_warmup_steps= num_warmup_steps,
            num_training_steps=num_training_steps
        )




for epoch in range(0, cfg.train.num_epochs):
    faiss.build_faiss(512)
    candidates_idxs = faiss.search_faiss(512) # (queries_N, topk)
    candidates_idxs = candidates_idxs.astype(np.int64)


    dataset.set_candidates(candidates_idxs)
    recall_faiss = faiss.compute_faiss_recall_at_k(candidates_idxs, k=topk)
    my_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True
            )
    epoch_loss, epoch_accuracy_5, epoch_mrr, epoch_margin = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    for i, data_loader_item in tqdm(enumerate(my_loader), total=len(my_loader), desc=f"epoch@{epoch} - Training batches"):
        loss, accuracy_5, mrr, batch_margin = train_one_batch(data_loader_item)
        model.optimizer.zero_grad(set_to_none=True)
        
        
        batch_x, batch_y = data_loader_item
        batch_query_tokens, batch_candidates_tokens = batch_x

        batch_scores = model(batch_query_tokens, batch_candidates_tokens)
        loss = model.get_loss(batch_scores, batch_y)

        loss.backward()
        model.optimizer.step()
        scheduler.step()

        accuracy_5, mrr, batch_margin = compute_metrics(batch_scores, batch_y, k=5)

        epoch_loss += loss
        epoch_accuracy_5 += accuracy_5
        epoch_mrr += mrr
        epoch_margin += batch_margin
        n_batches += 1

    avg_loss = (epoch_loss / max(1, n_batches)).item()
    avg_mrr = (epoch_mrr / max(1, n_batches)).item()
    avg_accuracy_5 = (epoch_accuracy_5 / max(1, n_batches)).item()
    avg_margin = (epoch_margin / max(1, n_batches)).item()  # Average

    logger.log_event(
        "Epoch summary",
        message=f"Epoch average loss={avg_loss:.3f}, average margin={avg_margin:.4f}, average mrr={avg_mrr:.4f}, average accuracy@5={avg_accuracy_5:.4f}. ",
        epoch=epoch,
        t0=batches_train_start_time
    )
