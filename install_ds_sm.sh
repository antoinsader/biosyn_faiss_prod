
#do either:
#   bash install_ds.sh lg
#or
#   bash install_ds.sh


set -e

# git clone https://github.com/antoinsader/biosyn_faiss_prod
# git clone --branch annotate --single-branch  https://github.com/antoinsader/biosyn_faiss_prod



TARGET_DIR="data/raw"

mkdir -p "$TARGET_DIR"

unzip -o ./raw/train_dictionary.zip -d  "$TARGET_DIR"
unzip -o ./raw/traindev.zip -d "$TARGET_DIR"
unzip -o ./raw/test.zip -d "$TARGET_DIR"





echo "Files successfully extracted to $TARGET_DIR"

python -m venv myenv
source myenv/bin/activate

# === Step 5: Install dependencies ===
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install faiss-gpu-cu12
pip install tqdm transformers requests psutil torchmetrics
pip install datasets

echo "Setup complete!"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
