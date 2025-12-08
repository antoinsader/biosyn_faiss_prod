
set -e

rm -rf logs/
rm -rf output/

# exp 2
python tokenizer.py --no_queries_annotate --no_dictionaries_annotate
python train.py --training_log_name='exp_2' --num_epochs=10 --topk=20 --hard_positives_num=0 --hard_negatives_num=0  --build_faiss_batch_size=8000  --train_batch_size=32

# exp 3
rm -rf ./data/tokens/
python tokenizer.py 
python train.py --training_log_name='exp_3' --num_epochs=8 --topk=20 --hard_positives_num=0 --hard_negatives_num=0 --build_faiss_batch_size=8000  --train_batch_size=32

# exp 4
rm -rf ./data/tokens/
python tokenizer.py --dictionary_annotation_add_synonyms
python train.py --training_log_name='exp_4' --num_epochs=10 --topk=20 --hard_positives_num=0 --hard_negatives_num=0   --build_faiss_batch_size=8000  --train_batch_size=32

# exp 5
python train.py --training_log_name='exp_5' --num_epochs=10 --topk=40 --hard_positives_num=0 --hard_negatives_num=0  --build_faiss_batch_size=8000  --train_batch_size=16


