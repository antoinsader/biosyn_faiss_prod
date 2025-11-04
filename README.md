# BioSyn FAISS retreival system

Deep learning retrieval system for learning the matching between mentions (queries) and dictionary entries using encoders and FAISS vector search.

This model was trained on the NCBI dataset and it is designed to learn semantic matching between biomedical terms.

---

## Overview:

### Data structure:

The training dataset consisting of:

    1- **Dictionary file (.txt)** - each line conotains a  CUI and its corresponding entity name
    2- **Traindev folder** - contains `.concept` files, each line representing a mention and its associated CUI

The model learns to relate the mentions in the traindev set to their correesponding terms in the dictionary.

The model was successfully trained with **4m records dictionary**, acheiving the following  results:
    - topk = 20
    - accuracy@5: 0.9659
    - mrr: 0.8903
    - average_loss: 0.3

### Training pipeline:

1. Retreive top-k candidates for each query from the dictionary using **FAISS**.
2. Perform a forward pass to calculate similarity scores (cosine similarity) between each query and its candidates
3. Fine-tune the encoder so that terms having same CUIs are close in the embedding space, while terms having different CUIs are pushed apart.

### FAISS index choosing:
- For dictionaries larger than **1m entries** -> `GpuIndexIVFPQ` index with the  `IndexHNSWFlat` quantizer
- For smaller dictionaries ->  `GpuIndexFlatIP` index or `IndexFlatIP` depending if cuda is available

---

## Installation:

### Recommended (Linux + CUDA GPU)

```bash
    bash install_ds.sh
```

This script:
- Extracts dictionary and traindev zip files from `/raw/` into `./data/raw/`
- Creates a python virtual environment
- Installs all required dependencies

### Manual setup:

If you are using another OS or prefer manual setup:
1. Create python virtual environment
2. Install dependencies from `requirements.txt`

**Notes:**
- `faiss-gpu-cu12` is the faiss using CUDA 12. Use standard `faiss` if you have no GPU. or `faiss-gpu` if you have lower GPU.
- Ensure the folder data/raw contains traindev/ folder  and train_dictionary.txt file before training.

---

## Tokenize:

The `tokenizer.py` script:
    - reads the dictionary (`train_dictionary.txt`) and the `traindev/` folder of query concept files
    - Tokenizes dictionary and query using a Hugging Face tokenizer (by default, BioBERT).
    - Save tokens and cuis in the corresponding files
    - Optionally splits traindev into train/test  datasets

### Usage

```bash
python tokenizer.py
```

### Arguments

| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--dictionary_path` | str | `./data/raw/train_dictionary.txt` | Path to dictionary file |
| `--queries_dir` | str | `./data/raw/traindev` | Path to traindev folder |
| `--dictionary_max_length` | int | 120 | Tokenizer padding/truncation length for dictionary |
| `--queries_max_length` | int | 120 | Tokenizer padding/truncation length for queries |
| `--test_split_percentage` | float | 0.8 | Fraction of traindev used for training |
| `--skip_tokenizing_dictionary` | bool | False | Skip dictionary tokenization |
| `--skip_tokenizing_queries` | bool | False | Skip query tokenization |
| `--skip_split` | bool | False | Skip splitting traindev into train/test |


**Examples:**

- If you want to tokenize only specific dictionary that you have, you can execute

    ```bash
        python tokenizer.py --dictionary_path='./path/to/dictionary.txt' --skip_tokenizing_queries --skip_split
    ```

- After the process is finished, results would be saved in:

```
data/tokens/
 ├── _dictionary_inp.mmap      ← dictionary input_ids
 ├── _dictionary_att.mmap      ← dictionary attention masks
 ├── _train_queries_inp.mmap      ← query input_ids
 ├── _train_queries_att.mmap      ← query attention masks
 ├── _dictionary_cuis.npy       ← dictionary CUIs
 ├── _train_queries_cuis.npy       ← query CUIs
 └── _dictionary_meta.json          ← dictionary metadata (containing shape of array)
```

If --skip_split was not set, you will have also files for test dataset (test_queries_inp.mmap, test_queries_att.mmap, test_queries_cuis.npy, test_queries_meta.json)


---
## Minimize:

If your dictionary is too large and you want to reduce it while keeping all CUIs appearing in traindev set, use:

```bash
    python minimize.py --minimize_target=50000
```

This ensures all required CUIs are included, then pads the dictionary with random entries to reach the target size.

When you minimize, to use the small dictionary you have to specify --use_small_dictionary in train.py

---

## Train

Training is handled by `train.py`.

### Arguments

| Argument | Required | Description | Default |
|-----------|-----------|-------------|----------|
| `--training_log_name` | ✅ | Unique name for the experiment | N/A |
| `--encoder_model_name` | ❌ | Pretrained model name or directory | `dmis-lab/biobert-base-cased-v1.1` |
| `--num_epochs` | ❌ | Number of epochs | 10 |
| `--train_batch_size` | ❌ | Batch size | 16 |
| `--topk` | ❌ | Number of retrieved candidates per query | 20 |
| `--hard_positives_num` | ❌ | Hard positive samples per query | 2 |
| `--hard_negatives_num` | ❌ | Hard negative samples per query | 7 |
| `--learning_rate` | ❌ | Learning rate | 5e-5 |
| `--weight_decay` | ❌ | Weight decay | 0.001 |
| `--loss_type` | ❌ | `marginal_nll` or `info_nce_loss` | `marginal_nll` |
| `--build_faiss_batch_size` | ❌ | Batch size for building FAISS index | 4096 |
| `--search_faiss_batch_size` | ❌ | Batch size for FAISS search | 4096 |
| `--use_amp` | ❌ | Enable automatic mixed precision | True (if available) |
| `--use_small_dictionary` | ❌ | Use the dictionary generated by minimize.py | False |

### Basic training example

```bash
    python train.py --training_log_name='big dictionary training'
```

### What is happening inside train.py:

#### 1. Checkpointing and logger:

- At first, we check if any unfinished experiments exist, if yes we will restore checkpoint from this experiment, otherwise create a new experiment
- The experiments metadata are saved in `./logs/logger_all.json`

#### 2. Loss temperature warmup:
- Each epoch dynamically calculates its own loss temperature based on the equation:

```
    cfg.train.loss_temperature = max(0.05, 0.15 * (0.88 ** (epoch - 1)))
```

Thus, if we have 10 epochs, we will have the loss temperatures:
[0.17045454545454547, 0.15, 0.132, 0.11615999999999999, 0.10222079999999999, 0.089954304, 0.07915978752, 0.0696606130176, 0.061301339455488, 0.05394517872082944]

This lowers gradually the temperature, stabilizing early learning.


#### 3. Each epoch workflow:

Each epoch executes the following pipeline:

1. **Warmup/Layer freezing**:

At the start of training, lower transformer layers are frozen, to make early epochs only train the top layers, gradually unfreezing deeper layers as the model stabilizes.

2. **FAISS index building**:
    - Dictionary is embedded using the current encoder, then those embeddings are added into the FAISS index in batches.
    - Each embeding  is normalized so dot product = cosine similarity.
    - If dictionary size is bigger than 1m, then we will use IVFPQ index, and for this kind of index, we need to train the index on some samples in order that faiss has clusters (faster search), that's why we will do sampling from dictionary embeddings and train the index.
    - You can modify some parameters of FAISS in config.py, though we set those parameters based on FAISS documentation:
        - num_clusters: calculated based on dictionary size: int(math.sqrt(dictionary_size)). the bigger this number is, the faster the index would be, the lower the accuracy would be.
        - n_probe: calculated based on num_clusters: int(0.06 * num_clusters).number of cluster to be visited during search, usually 1-10% of num_clusters. the bigger this number is, the more accuracy you would have, the slower the index would be.
        - clusters_samples: calculated based on 256 * num_clusters. The number of the samples that FAISS would train at initialization.

3. **FAISS search**:

- Queries are embedded in batches and added to FAISS index.
- Each embeding is also normalized.
- Retreive for each query, topk candidates (which are most similar dictionary entries), the retreived candidates are dictionary indices
- Then FAISS recall@k is computed, which is the percentage where candidates has been retreived correctly in the first k candidates for each query

4. **Dataset candidates injection**:

- Dpending on (hard_positives_num, hard_negatives_num default to 2, 7) The retreived candidates pool would be updated.
- Injecting hard negative candidates would make the model better distinguish the candidates. We are injecting negative candidates that FAISS search has thought they are correct candidates in the previous epoch.
- Hard positive candidates will reinforce correct similarity
- Positions of injected candidates are random and made in a way to not be overlapped to 

5. **Train loop**:

Train loop will do in batches:
    - Query and candidate tokens are being embedded and normalized
    - Compute cosine similarity using torch.bmm between the query embeding and its candidates
    - Calculate loss (marginal_nll or info_nce_loss) and divide it by loss_temperature of the epoch
    - backward pass and optimization using AdamW optimizer and scheduler

6. **Calculate metrics**:
- Accuracy@5: for each query, in its first 5 candidates, did the model predicted at least one correct candidate in those 5
- mrr: Mean Reciprocal Rank. How high up the correct answer appears among the candidates for each query.

7. **Save checkpoint**: 
-  encoder, FAISS index, and metrics logs are saved.

---

#### 4. Save output:

- After training finishes, the files that would be produced are:
    - ./output/faiss_index.index: last faiss index used
    - ./output/encoder_<number>/: folder containing encoder stat, you can use this encoder after for evaluation

    - ./logs/all_logger.json: json file containing all experiments with statistics about the training.
    - ./logs/<log_special_name>.log containing detailed training log


---



## Evaluation:

For eval you have to specify --result_encoder_dir which is encoder output dir to be evaluated.
Execution:

```bash
    python eval.py --result_encoder_dir='./output/encoder_1/'
```

Will log top-k accuracy and MRR for retrieval scores

---

## Notes

- All major directories (`data/tokens`, `data/raw`, `logs`, `output`) are auto-created if missing.
- Mixed precision (AMP) can significantly speed up training on GPUs.

---

**Author:** Starky