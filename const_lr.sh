#!/usr/bin/env bash
set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

OUT=outputs_new
BS_TRAIN=8
BS_EVALUATE=4
ACC=2
EPOCHS=1
MAXLEN=512

# =========================
# Define Models and Datasets
# =========================
MODELS=(
  # "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
  "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
)
DATASETS=("ag_news" "squad")


# =========================
# Define Base LRs (you fill these)
# Order: (ag_news + Llama), (ag_news + Qwen), (squad + Llama), (squad + Qwen)
# =========================
LR_AG_LLAMA=0.000322
LR_AG_QWEN=0.000385
LR_SQ_LLAMA=0.000372
LR_SQ_QWEN=0.000576

# Function to get LR for each combo
get_lr() {
  local model="$1"
  local dataset="$2"

  if [[ "$dataset" == "ag_news" && "$model" == *"Llama"* ]]; then
    echo "$LR_AG_LLAMA"
  elif [[ "$dataset" == "ag_news" && "$model" == *"Qwen"* ]]; then
    echo "$LR_AG_QWEN"
  elif [[ "$dataset" == "squad" && "$model" == *"Llama"* ]]; then
    echo "$LR_SQ_LLAMA"
  elif [[ "$dataset" == "squad" && "$model" == *"Qwen"* ]]; then
    echo "$LR_SQ_QWEN"
  else
    echo "0.0"
  fi
}

# =========================
# Training Loop
# =========================
for M in "${MODELS[@]}"; do
  for D in "${DATASETS[@]}"; do

    BASE_LR=$(get_lr "$M" "$D")

    echo ""
    echo "=============================================="
    echo " Dataset: $D | Model: $M | Base LR: $BASE_LR "
    echo "=============================================="

    for SCALE in 0.5 1.0 2.0; do
      LR=$(python -c "print(${BASE_LR} * ${SCALE})")
      echo "Running with LR=${LR}"

      python main.py \
        --model_name "$M" \
        --dataset "$D" \
        --optimizer "adam" \
        --scheduler "constant" \
        --learning_rate "$LR" \
        --num_train_epochs "$EPOCHS" \
        --per_device_train_batch_size "$BS_TRAIN" \
        --per_device_eval_batch_size "$BS_EVALUATE" \
        --gradient_accumulation_steps "$ACC" \
        --max_seq_len "$MAXLEN" \
        --output_dir "$OUT/${D}_${M//\//_}_adam_constant_lr${SCALE}"
    done
  done
done
