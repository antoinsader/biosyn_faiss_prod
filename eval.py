
import logging
import torch
import gc
import numpy as np

from tqdm import tqdm

from config import GlobalConfig, eval_parse_args

from main_classes.MyFaiss import MyFaiss
from main_classes.MyEncoder import MyEncoder
from main_classes.Reranker import Reranker 
from helpers.MyLogger import MyLogger, CheckPointing
from helpers.Data import MyDataset, TokensPaths








class Evaluator:
    def __init__(self, logger: MyLogger, cfg: GlobalConfig):
        self.logger : MyLogger = logger
        self.cfg :GlobalConfig = cfg

        
        self.encoder_dir = cfg.paths.result_encoder_dir
        cfg.model.model_name = self.encoder_dir

        
        cfg.train.inject_hard_negatives_candidates = False
        cfg.train.inject_hard_positives_candidates = False

        self.tokens_paths = TokensPaths(cfg, dictionary_key="dictionary", queries_key='test_queries')

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.dataset = MyDataset(self.tokens_paths, cfg)
        self.encoder = MyEncoder(cfg)
        self.model = Reranker(self.encoder, self.cfg)
        self.faiss = MyFaiss(cfg, save_index_path="",  dataset=self.dataset, tokens_paths=self.tokens_paths, encoder=self.encoder)
        self.faiss.load_faiss_index(cfg.paths.faiss_path)
        self.topk = cfg.train.topk

    def eval(self):
        self.model.eval()
        candidates_idxs = self.faiss.search_faiss(self.cfg.faiss.search_batch_size) # (queries_N, topk)
        candidates_idxs = candidates_idxs.astype(np.int64)
        self.dataset.set_candidates(candidates_idxs)
        recall_faiss = self.faiss.compute_faiss_recall_at_k(candidates_idxs, k=self.topk)
        self.logger.log_event(f"Faiss recall@{self.topk}", message= f"{recall_faiss:.4f}",  log_memory=False)

        my_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            pin_memory=self.use_cuda,
            num_workers=self.cfg.train.num_workers,
            persistent_workers=False
        )

        total_samples = 0
        n_eval = 0

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.use_cuda):
            all_metrics = []
            for batch_x, batch_y in tqdm(my_loader, desc="Evaluating"):
                batch_y = batch_y.to(self.device)
                batch_size = batch_y.size(0)


                query_tokens, candidate_tokens = batch_x
                query_tokens = {k: v.to(self.device) for k, v in query_tokens.items()}
                candidate_tokens = {k: v.to(self.device) for k, v in candidate_tokens.items()}
                batch_x = (query_tokens, candidate_tokens)

                # Forward pass
                query_tokens, candidate_tokens = batch_x
                batch_pred = self.model(query_tokens, candidate_tokens)  # [batch_size, hidden_size]
                loss = self.model.get_loss(batch_pred, batch_y)

                res = self._compute_metrics_eval(batch_pred.detach(), batch_y, multiple_ks=[1, 2,4, 5, 7, 10, 12, 15, 17, 20])
                res["loss"] = loss.item()
                all_metrics.append(res)
                total_samples += batch_size
                n_eval += 1
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        self.logger.log_event(
            event_tag="Evaluation results",
            message=f"Average Loss: {avg_metrics['loss']:.4f} \n  Mean Reciprocal Rank (MRR): {avg_metrics['mrr']:.4f} \n ",
            log_memory=False
        )


        acc_results = []
        for k in sorted([kk for kk in avg_metrics.keys() if kk.startswith('acc@')],
                        key=lambda x: int(x.split('@')[1])):
            acc_results.append(f"{k:>10}: {avg_metrics[k]:.4f}")


        self.logger.log_event(
            event_tag="Evaluation  accuracy results",
            message="\n".join(acc_results),
            log_memory=False
        )

        return True

    def _compute_metrics_eval(self, scores, targets, multiple_ks=None, k=5):
        """
        Compute top-k accuracy and MRR for retrieval scores.
        Works for both info_nce_loss (int targets) and marginal_nll (float vector targets).

        Args:
            scores (Tensor): shape [batch_size, topk]
            targets (Tensor): either long (for info_nce) or float (for marginal_nll)
            multiple_ks (list[int], optional): e.g., [1, 5, 10]. If provided, returns multiple accuracies.
            k (int): default top-k if multiple_ks not provided

        Returns:
            dict: {'acc@1': ..., 'acc@5': ..., 'mrr': ...}
        """
        with torch.no_grad():
            batch_size = scores.size(0)
            topk = scores.size(1)
            ks = multiple_ks if multiple_ks is not None else [k]

            results = {}
            total_rr_sum, total_valid = 0.0, 0

            # Handle info_nce style targets (single positive index)
            if targets.dtype == torch.long:
                topk_preds = scores.argsort(dim=-1, descending=True)  # [B, topk]
                acc_counts = {kk: 0 for kk in ks}

                for i in range(batch_size):
                    t = targets[i].item()
                    if t == -100 or t < 0 or t >= topk:
                        continue
                    total_valid += 1
                    preds = topk_preds[i].tolist()

                    # Reciprocal rank
                    rank = preds.index(t) + 1 if t in preds else topk + 1
                    total_rr_sum += 1.0 / rank

                    for kk in ks:
                        if t in preds[:kk]:
                            acc_counts[kk] += 1

                for kk in ks:
                    results[f"acc@{kk}"] = acc_counts[kk] / max(total_valid, 1)
                results["mrr"] = total_rr_sum / max(total_valid, 1)

            # Handle marginal_nll style targets (float vector of 0/1)
            else:
                positives = (targets > 0.5)
                topk_preds = scores.argsort(dim=-1, descending=True)
                acc_counts = {kk: 0 for kk in ks}

                for i in range(batch_size):
                    pos_idxs = positives[i].nonzero(as_tuple=True)[0].tolist()
                    if len(pos_idxs) == 0:
                        continue
                    total_valid += 1
                    preds = topk_preds[i].tolist()

                    # Reciprocal rank of first correct
                    for r, idx in enumerate(preds, start=1):
                        if idx in pos_idxs:
                            total_rr_sum += 1.0 / r
                            break

                    for kk in ks:
                        if any(p in pos_idxs for p in preds[:kk]):
                            acc_counts[kk] += 1

                for kk in ks:
                    results[f"acc@{kk}"] = acc_counts[kk] / max(total_valid, 1)
                results["mrr"] = total_rr_sum / max(total_valid, 1)

        return results


LOGGER = logging.getLogger()

if __name__ == "__main__":

    cfg: GlobalConfig = eval_parse_args()
    cfg.logger.tag = "eval"

    chkpointing = CheckPointing(cfg, eval=True)
    logger = MyLogger(LOGGER , chkpointing.current_experiment_log_path, cfg.logger.tag )
    evaluator = Evaluator(logger, cfg)
    evaluator.eval()

    torch.cuda.empty_cache()
    gc.collect()







