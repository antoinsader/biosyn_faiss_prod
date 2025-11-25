import argparse
from dataclasses import dataclass, field
import math
import os
from enum import Enum


class EncodingType(str, Enum):
    CLS_ONLY = "cls_only"
    BETWEEN_SPANS = "between_spans"
    SPANS_ONLY = "spans_only"
    CLS_AND_BETWEEN_SPANS = "cls_and_between_spans"
    CLS_AND_SPANS = "cls_and_spans"





@dataclass
class PathsConfig:
    tokens_dir : str = './data/tokens'
    logs_dir: str = "./logs"
    output_dir: str = "./output"
    chkpnts_dir: str = "./checkpoints"
    embeds_dir: str = "./data/embeds"
    raw_dir: str = "./data/raw"
    draft_dir: str = "./data/draft"

    global_log_path: str = f"./logs/logger_all.json"
    
    dictionary_raw_path = "./data/raw/train_dictionary.txt"
    queries_raw_dir = "./data/raw/traindev"
    test_queries_raw_dir = "./data/raw/test"
    tokenizer_meta_path = "./data/tokenizer.json"

    result_encoder_dir = None
    checkpoint_dir = None
    checkpoint_path = None
    faiss_path = None

    def __post_init__(self):
        assert os.path.isdir(self.raw_dir)
        os.makedirs(self.tokens_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embeds_dir, exist_ok=True)
        os.makedirs(self.chkpnts_dir, exist_ok=True)
        os.makedirs(self.draft_dir, exist_ok=True)

    def set_result_encoder_dir(self, dir):
        self.result_encoder_dir = dir
        self.checkpoint_dir = os.path.join(dir, "checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "last.pt")
        self.faiss_path  = self.result_encoder_dir + "/faiss_index.faiss"
        os.makedirs(self.result_encoder_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def get_token_group(self, prefix):
        return {
            "input_ids": os.path.join(self.tokens_dir, f"{prefix}_inp.mmap"),
            "attention_mask": os.path.join(self.tokens_dir, f"{prefix}_att.mmap"),
            "cuis": os.path.join(self.tokens_dir, f"{prefix}_cuis.npy"),
            "meta": os.path.join(self.tokens_dir, f"{prefix}_meta.json"),
        }

    def get_default_token_groups(self):
        return {
            "train_queries": self.get_token_group("_train_queries"),
            "dictionary": self.get_token_group("_dictionary"),
            "small_dictionary": self.get_token_group("_small_dictionary"),
            "test_queries": self.get_token_group("_test_queries"),
        }


@dataclass
class TokensConfig:


    queries_max_length = 75 #max tokens

    dictionary_max_length = 75
    dictionary_max_chars_length = 128 #if less would be skipped

    tokenize_batch_size : int = 128_000
    raw_test_dir:str = None

    skip_tokenize_dictionary: bool = False
    skip_tokenize_queries: bool = False
    skip_tokenize_test_queries: bool = False
    split_train_queries :bool=False
    test_split_percentage: float = 0.2

    

    dictionary_annotation_add_synonyms: bool = False

    query_tokens_window_words_in_text = 10 #5 words before mention start, 5 words after mention start
    special_tokens = {"additional_special_tokens": ["[MS]", "[ME]"]}
    special_tokens_dict = {
        "mention_start": "[MS]",
        "mention_end": "[ME]",
    }


    # special_tokens = {
    #     'additional_special_tokens': [
    #         '[MENTION_CONTEXT_START]', '[MENTION_CONTEXT_END]',  # existing
    #         '[MENTION_NAME_START]', '[MENTION_NAME_END]',
    #         '[CONTEXT_START]', '[CONTEXT_END]',
    #         '[TYPE_START]', '[TYPE_END]'
    #     ]
    # }
    # special_tokens_dict = {
    #     "mention_name_start": "[MENTION_NAME_START]",
    #     "mention_name_end": "[MENTION_NAME_END]",
    #     "mention_in_sentence_start": "[MENTION_CONTEXT_START]",
    #     "mention_in_sentence_end": "[MENTION_CONTEXT_END]",
    #     "context_start": "[CONTEXT_START]",
    #     "context_end": "[CONTEXT_END]",
    #     "type_start": "[TYPE_START]",
    #     "type_end": "[TYPE_END]",
        
    # }



@dataclass
class LoggerConfig:
    tag:str="train"
    train_log_name: str = ""


@dataclass
class InferenceConfig:
    mention: str = ""
    topk: int = 5
@dataclass
class ModelConfig:
    model_name : str = 'dmis-lab/biobert-base-cased-v1.1'
    pooling : str =  'hybrid' #[mean, cls, hybrid]
    normalize: bool = True
    hidden_size: int = 768
    encoding_type : EncodingType = EncodingType.CLS_AND_BETWEEN_SPANS


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 16
    

    learning_rate: float = 1e-5  #5e-6 #5e-5 


    weight_decay: float = 0.001
    num_workers: int = 8
    topk: int = 20
    optimizer_name: str = "AdamW" # Adam
    use_amp: bool = True
    loss_temperature: float = 0.06
    save_checkpoints:bool = True
    load_last_checkpoint:bool = True
    use_small_dictionary: bool = False
    load_data_to_ram: bool = True

    inject_hard_negatives_candidates:bool= False
    hard_negatives_num:int= 0
    inject_hard_positives_candidates:bool= False
    hard_positives_num:int= 0

    freeze_lower_layer_epoch_max:int=2
    enable_gradient_checkpoint:bool=False
    gradient_accumulation_steps: int = 1
    update_faiss_every_n_epochs: int = 2
    metric_compute_interval: int = 500



@dataclass
class FaissConfig:
    build_batch_size: int = 6096
    search_batch_size: int = 6096

    save_index_path = None

    num_quantizers = 32
    nbits= 8
    hnsw_efConstruction = 200
    hnsw_efSearch = 512

    force_ivfpq = False

    def num_clusters(self, dictionary_size):
        #bigger number means faster search and lower accuracy
        return int(math.sqrt(dictionary_size))

    def n_probe(self, num_clusters):
        #nprobe is the numbers of clusters to be visited during search, higher means more accurate but slower
        # 1-10% of nlist
        return int(0.06 * num_clusters)

    def clusters_samples(self, num_clusters):
        return 256 * num_clusters


@dataclass
class GlobalConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    tokenize: TokensConfig = field(default_factory=TokensConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    skip_eval: bool = False
    skip_train: bool = False
    eval_encoder_dir:str = ""
    eval_faiss_dir:str = ""

    minimize_target= 500_000
    def to_dict(self):
        return {
            "model": vars(self.model),
            "train": vars(self.train),
            "faiss": vars(self.faiss),
            "logger": vars(self.logger),
            "paths": vars(self.paths)
        }
class CheckPointModel:
    def __init__(self, chkpt):
        self.model_state= chkpt['model_state']
        self.optimizer_state = chkpt['optimizer_state']
        self.scheduler_state = chkpt['scheduler_state']
        self.scaler_state = chkpt['scaler_state']
        self.epoch = chkpt['epoch']
        self.faiss_index_path = chkpt['faiss_index_path']

class LogDataModel:
    def __init__(self, row):
        self.training_log_name= row['training_log_name']


def tokenizer_parse_args():
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')

    parser.add_argument('--model_name_or_path',  type=str)

    parser.add_argument('--dictionary_path',  type=str)
    parser.add_argument('--queries_dir',  type=str)
    parser.add_argument('--test_queries_dir',  type=str)
    parser.add_argument('--split_train_queries',  action="store_true")
    parser.add_argument('--test_split_percentage',  type=float)

    parser.add_argument('--skip_tokenizing_dictionary',  action="store_true")
    parser.add_argument('--skip_tokenizing_queries',  action="store_true")
    parser.add_argument('--skip_tokenizing_test_queries',  action="store_true")
    
    
    parser.add_argument('--dictionary_annotation_add_synonyms',  action="store_true")

    args = parser.parse_args()


    if args.model_name_or_path:
        cfg.model.model_name = args.model_name_or_path

    if args.skip_tokenizing_dictionary:
        cfg.tokenize.skip_tokenize_dictionary = True
    if args.skip_tokenizing_queries:
        cfg.tokenize.skip_tokenize_queries = True
    if args.skip_tokenizing_test_queries:
        cfg.tokenize.skip_tokenize_test_queries = True
    
    
    if args.dictionary_path:
        assert os.path.exists(args.dictionary_path), f'Dict path: {args.dictionary_path} not exists'
        cfg.paths.dictionary_raw_path = args.dictionary_path

    if args.queries_dir:
        assert os.path.isdir(args.queries_dir), f'Queries dir: {args.queries_dir} not exists'
        cfg.paths.queries_raw_dir = args.queries_dir

    if args.test_queries_dir:
        assert os.path.isdir(args.test_queries_dir), f'Test queries dir: {args.test_queries_dir} not exists'
        cfg.paths.test_queries_raw_dir = args.test_queries_dir


    if args.split_train_queries:
        cfg.tokenize.split_train_queries = True
    
    if args.test_split_percentage:
        cfg.tokenize.split_train_queries = True
        assert 0.0 <= args.test_split_percentage <= 1.0 
        cfg.tokenize.test_split_percentage = args.test_split_percentage


    if args.dictionary_annotation_add_synonyms:
        cfg.tokenize.dictionary_annotation_add_synonyms = True
    else:
        print(f"You are not adding synonyms to the dictionary annotations, recent results shown that adding synonyms is better, if you want to add, consider doing --dictionary_annotation_add_synonyms")

    return cfg


def minimize_parse_args():
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')

    parser.add_argument('--minimize_target', type=int,
                        help='How many dictionary entries you want to keep, it should be more than the number of traindev entries ')


    args = parser.parse_args()

    if args.minimize_target:
        cfg.minimize_target = args.minimize_target

    return cfg

def eval_parse_args():
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')

    parser.add_argument('--result_encoder_dir', required=True,
                        help='Result encoder dir, you should have this after tain, the dir is where the encoder files are saved')


    args = parser.parse_args()

    if args.result_encoder_dir:
        assert os.path.isdir(args.result_encoder_dir)
        cfg.paths.result_encoder_dir = args.result_encoder_dir

        cfg.paths.faiss_path = os.path.join(args.result_encoder_dir, "faiss_index.faiss")
        assert os.path.exists(cfg.paths.faiss_path), f'Faiss not found,  {cfg.paths.faiss_path}'

    return cfg

def inference_parse_args():
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')
 # Required arguments
    parser.add_argument('--mention', type=str, required=True, 
                        help='Medical mention/entity to normalize (e.g., "breast cancer")')
    
    parser.add_argument('--result_encoder_dir', required=True,
                        help='Result encoder dir, you should have this after tain, the dir is where the encoder files are saved')

    # Optional arguments
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top candidates to retrieve (default: 5)')
    

    args = parser.parse_args()

    if args.result_encoder_dir:
        assert os.path.isdir(args.result_encoder_dir)
        cfg.paths.result_encoder_dir = args.result_encoder_dir

        cfg.paths.faiss_path = os.path.join(args.result_encoder_dir, "faiss_index.faiss")
        assert os.path.exists(cfg.paths.faiss_path), f'Faiss not found,  {cfg.paths.faiss_path}'

    if args.topk:
        cfg.inference.topk = args.topk

    cfg.inference.mention = args.mention

    return cfg

def train_parse_args():
    """
    Parse input arguments
    """
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')

    # Required
    parser.add_argument('--training_log_name', required=True,
                        help='Unique name for the training session')


    # optional
    parser.add_argument('--model_name_or_path',
                        help='Directory for pretrained model', required=False)
    
    parser.add_argument('--use_small_dictionary', help='In case you minimized and want to use the small dictionar', action='store_true')
    parser.add_argument('--num_epochs', help='train num epochs', type=int, required=False)
    parser.add_argument('--train_batch_size', help='train batch size', type=int, required=False)
    parser.add_argument('--topk', help='train topk candidates', type=int, required=False)

    parser.add_argument('--hard_positives_num', help='From topk specified, how many hard positive candidates to inject', type=int, required=False)
    parser.add_argument('--hard_negatives_num', help='From topk specified, how many hard negative candidates to inject', type=int, required=False)


    parser.add_argument('--learning_rate', help='train learning rate', type=float, required=False)
    parser.add_argument('--weight_decay', help='train weight decay', type=float, required=False)
    

    parser.add_argument('--build_faiss_batch_size', help='Batch size when building faiss index ', type=int, required=False)
    parser.add_argument('--search_faiss_batch_size', help='Batch size when searching in faiss ', type=int, required=False)

    parser.add_argument('--use_amp',  action="store_true")
    parser.add_argument('--force_ivfpq',  action="store_true")
    parser.add_argument('--no_load_data_to_ram',  action="store_true")
    parser.add_argument('--enable_gradient_checkpoint',  action="store_true")




    args = parser.parse_args()


    if args.training_log_name:
        cfg.logger.train_log_name = args.training_log_name

    if args.model_name_or_path:
        cfg.model.model_name = args.model_name_or_path

    if args.num_epochs:
        assert  1 <= args.num_epochs < 25, f'Num epochs should be between 1 and 25'
        cfg.train.num_epochs = args.num_epochs
    if args.train_batch_size:
        cfg.train.batch_size = args.train_batch_size
    if args.topk:
        assert args.topk > 5, 'Topk candidates should be at least 5'
        cfg.train.topk = args.topk

    if args.hard_negatives_num:
        assert args.hard_negatives_num < cfg.train.topk, f'Hard negatives num should be less than topk ({cfg.train.topk})'
        if args.hard_negatives_num == 0:
            cfg.train.inject_hard_negatives_candidates = False
        else:
            cfg.train.inject_hard_negatives_candidates = True

        cfg.train.hard_negatives_num = args.hard_negatives_num
    if args.hard_positives_num:
        assert args.hard_positives_num < cfg.train.topk, f'Hard positives num should be less than topk ({cfg.train.topk})'
        if args.hard_positives_num == 0:
            cfg.train.inject_hard_positives_candidates = False
        else:
            cfg.train.inject_hard_positives_candidates = True    
        cfg.train.hard_positives_num = args.hard_positives_num


    if args.learning_rate:
        cfg.train.learning_rate = args.learning_rate
    if args.weight_decay:
        cfg.train.weight_decay = args.weight_decay
    if args.build_faiss_batch_size:
        cfg.faiss.build_batch_size = args.build_faiss_batch_size
    if args.search_faiss_batch_size:
        cfg.faiss.search_batch_size = args.search_faiss_batch_size

    if args.use_small_dictionary:
        cuis = cfg.paths.get_default_token_groups()['small_dictionary']['cuis']
        inp_ids = cfg.paths.get_default_token_groups()['small_dictionary']['input_ids']
        assert os.path.exists(cuis) and os.path.exists(inp_ids), f"Mini dictionary was not created, make sure to execute minimize.py"
        cfg.train.use_small_dictionary = True


    if args.use_amp:
        cfg.train.use_amp = args.use_amp

    if args.force_ivfpq :
        cfg.faiss.force_ivfpq = True

    if args.no_load_data_to_ram:
        cfg.train.load_data_to_ram = False

    if args.enable_gradient_checkpoint:
        cfg.train.enable_gradient_checkpoint = True
    else:
        print(f"If your dictionary entries are big (more than 1m), you should consider enabling gradient checkpointing, to not have OOM (it would be slower but more stable)")

    return cfg

