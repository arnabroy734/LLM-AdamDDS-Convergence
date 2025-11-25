# main.py
import argparse
from finetune import run_experiment

def parse_args():
    p = argparse.ArgumentParser(description="QLoRA 4-bit finetuning for two tasks")
    # Core
    p.add_argument("--model_name", type=str, required=True,
                   help="HF model id (e.g., microsoft/phi-2 or google/gemma-2-2b)")
    p.add_argument("--dataset", type=str, required=True, choices=["ag_news", "squad"],
                   help="Which dataset to use")
    # Optimizer / Scheduler
    p.add_argument("--optimizer", type=str, required=True,
                   choices=["adam", "adabelief", "adabound", "yogi", "radam", "adamw"])
    p.add_argument("--scheduler", type=str, default=None,
                   choices=["step-decay","linear-decay","cosine-decay","exponential-decay","square-root-decay","inverse-time-decay", "constant"],
                   help="Only applied if optimizer == adam; others use cosine by default")
    # Optional scheduler params
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--decay_rate", type=float, default=0.95, help="For exponential/step/inverse schedulers")
    p.add_argument("--decay_steps", type=int, default=200, help="For step-decay")
    p.add_argument("--inverse_k", type=float, default=0.0005, help="for inverse-time: 1/(1+k*t)")
    # Training hyperparams
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--max_steps", type=int, default=-1, help="Override epochs if > 0")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Generation (QA)
    p.add_argument("--gen_max_new_tokens", type=int, default=64)
    p.add_argument("--gen_temperature", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_total_limit", type=int, default=1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
