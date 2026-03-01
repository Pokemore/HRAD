#!/bin/sh

set -euo pipefail

# Activate your conda environment.
source /opt/conda/bin/activate HRAD

# Repository root and working directory.
ROOT="/root/Documents/Code/HRAD"
cd "${ROOT}"

# GPU selection for testing.
export CUDA_VISIBLE_DEVICES=0
export NLTK_DATA="/root/Documents/Package/NLTK"
export PYTHONUNBUFFERED=1

# Pretrained weights (required).
SWIN_WEIGHTS="/root/Documents/PreTrained/Swin/swin_base_patch4_window12_384_22k.pth"
BERT_WEIGHTS="/root/Documents/PreTrained/BERT/bert-base-uncased"

# Output directory and checkpoint path.
OUTPUT_DIR="/root/Documents/Result"
MODEL_ID="HRAD_GPG"
CHECKPOINT="${OUTPUT_DIR}/checkpoints/model_best_${MODEL_ID}.pth"

# Optional: set to a non-empty path to save predicted masks.
SAVE_MASK_DIR="/root/Documents/Result/mask"

EXTRA_ARGS=()
if [ -n "${SAVE_MASK_DIR}" ]; then
  EXTRA_ARGS+=(--save_mask_dir "${SAVE_MASK_DIR}")
fi

python test.py \
  --model gpg_pfr_rfd \
  --use_gpg \
  --use_gpg_loss \
  --num_tmem 3 \
  --dataset RefOPT \
  --splitBy unc \
  --swin_type base \
  --pretrained_swin_weights "${SWIN_WEIGHTS}" \
  --ck_bert "${BERT_WEIGHTS}" \
  --img_size 512 \
  --window12 \
  --workers 4 \
  --refer_data_root "${ROOT}/dataset" \
  --resume "${CHECKPOINT}" \
  --split test \
  --ddp_trained_weights \
  "${EXTRA_ARGS[@]}"

