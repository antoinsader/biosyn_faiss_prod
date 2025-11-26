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
cfg.paths.result_encoder_dir = "./output/encoder_11/"
cfg.paths.faiss_path = os.path.join("./output/encoder_11/", "faiss_index.faiss")

cfg.logger.tag = "eval"

chkpointing = CheckPointing(cfg, eval=True)
logger = MyLogger(LOGGER , chkpointing.current_experiment_log_path, cfg.logger.tag )

encoder_dir = cfg.paths.result_encoder_dir
cfg.model.model_name = encoder_dir

tokens_paths = TokensPaths(cfg, dictionary_key="dictionary", queries_key='test_queries')
dataset = MyDataset(tokens_paths, cfg)

print(f"len test: {len(dataset.queries_cuis)}")
