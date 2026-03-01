set -euo pipefail

# Activate your conda environment.
source /path/to/conda/bin/activate <ENV_NAME>

# Repository root and working directory.
ROOT="/path/to/HRAD"
cd "${ROOT}/main"

# GPU selection (comma-separated). Adjust to your hardware.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NLTK data path (optional). Leave empty if not used.
export NLTK_DATA="/path/to/nltk_data"

export PYTHONUNBUFFERED=1

# Pretrained weights (required).
SWIN_WEIGHTS="/path/to/swin_base_patch4_window12_384_22k.pth"
BERT_WEIGHTS="/path/to/bert-base-uncased"

# Output directory for logs and checkpoints.
OUTPUT_DIR="/path/to/experiments/HRAD_GPG"
MODEL_ID="HRAD_GPG"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"

# Distributed training launch.
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 22469 \
    train.py \
    --model gpg_pfr_rfd \
    --use_gpg \
    --use_gpg_loss \
    --num_tmem 3 \
    --dataset RefOPT \
    --splitBy unc \
    --model_id "${MODEL_ID}" \
    --batch-size 2 \
    --lr 3e-5 \
    --wd 0.01 \
    --swin_type base \
    --pretrained_swin_weights "${SWIN_WEIGHTS}" \
    --ck_bert "${BERT_WEIGHTS}" \
    --epochs 40 \
    --img_size 512 \
    --workers 4 \
    --pin_mem \
    --output-dir "${OUTPUT_DIR}/checkpoints" \
    --refer_data_root "${ROOT}/dataset" \
    --pfr_stages 2 \
    --pfr_channels 128 \
    --loss_contrast_weight 0.06 \
    --loss_ortho_weight 0.06 \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

