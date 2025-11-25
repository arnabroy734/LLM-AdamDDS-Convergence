#!/usr/bin/env bash
set -e
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

OUT=outputs
BS_TRAIN=8
BS_EVALUATE=4
ACC=2
EPOCHS=1
LR=2e-4
MAXLEN=512

# Models
MODELS=(
  "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
  "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
)

# Datasets
DATASETS=("ag_news" "squad")


# Adam with all schedulers
SCHEDULERS=("step-decay" "linear-decay" "cosine-decay" "exponential-decay" "square-root-decay" "inverse-time-decay")


for M in "${MODELS[@]}"; do
  for D in "${DATASETS[@]}"; do
    for S in "${SCHEDULERS[@]}"; do
      echo "Running $M on $D with adam + $S"
      python main.py \
        --model_name "$M" \
        --dataset "$D" \
        --optimizer adam \
        --scheduler "$S" \
        --learning_rate $LR \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BS_TRAIN \
        --per_device_eval_batch_size $BS_EVALUATE \
        --gradient_accumulation_steps $ACC \
        --max_seq_len $MAXLEN \
        --output_dir "$OUT/${D}_${M//\//_}_adam_${S}"
    done
  done
done

# Other optimizers with cosine-decay
OPTIMIZERS=("adabelief" "adabound" "yogi" "radam" "adamw")

for M in "${MODELS[@]}"; do
  for D in "${DATASETS[@]}"; do
    for O in "${OPTIMIZERS[@]}"; do
      echo "Running $M on $D with $O + cosine-decay"
      python main.py \
        --model_name "$M" \
        --dataset "$D" \
        --optimizer "$O" \
        --scheduler "cosine-decay" \
        --learning_rate $LR \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BS_TRAIN \
        --per_device_eval_batch_size $BS_EVALUATE \
        --gradient_accumulation_steps $ACC \
        --max_seq_len $MAXLEN \
        --output_dir "$OUT/${D}_${M//\//_}_${O}_cosine"
    done
  done
done
