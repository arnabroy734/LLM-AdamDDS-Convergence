# finetune.py
import os, json, math, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorForSeq2Seq, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import DatasetDict
from data import build_datasets_and_tokenize
from metrics import streaming_eval, eval_squad
from optimizers import build_optimizer, build_scheduler
from utils import set_seed_all, ensure_dir
import sys

def run_experiment(args):
    set_seed_all(args.seed)
    ensure_dir(args.output_dir)
    run_name = f"{args.dataset}__{args.model_name.replace('/','_')}__{args.optimizer}{('_'+args.scheduler) if args.scheduler else ''}"
    run_dir = os.path.join(args.output_dir, run_name)
    if os.path.exists(run_dir):
        print(f"Directory {run_dir} already exists. Exiting.")
        return
    ensure_dir(run_dir)

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Safe default for decoder-only models
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # Data
    if "llama" in args.model_name.lower():
        model_name = "llama"
    elif "qwen" in args.model_name.lower():
        model_name = "qwen"
    dataset_dict: DatasetDict = build_datasets_and_tokenize(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        model_name=model_name
    )

    # Collator (seq2seq-style text-to-text; works for both classification prompt and QA prompt)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=run_dir,
        bf16=True,
        remove_unused_columns=False,
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1.0,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        lr_scheduler_type="constant",  # placeholder; overridden by our custom scheduler if needed
        report_to=[],
        eval_accumulation_steps=4,
        dataloader_num_workers=4
    )

    # Build optimizer
    optimizer = build_optimizer(model, args)

    # Trainer metrics per task
    postproc = None
    if args.dataset == "ag_news":
        # compute_metrics = lambda eval_pred: compute_metrics_classification(eval_pred, tokenizer)
        compute_metrics = None
    else:
        # postproc = SquadPostProcessor(tokenizer=tokenizer)
        # compute_metrics = lambda eval_pred: postproc.compute_metrics(eval_pred)
        compute_metrics = None


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_dict["train"],
        # eval_dataset=dataset_dict["validation"],
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
        optimizers=(optimizer, None)  # scheduler is attached after training starts
    )

    # Build scheduler after knowing total steps
    total_steps = (len(dataset_dict["train"]) * math.ceil(training_args.num_train_epochs)
                   // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
    scheduler = build_scheduler(optimizer, training_args, args, total_steps)
    trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    # print(trainer.lr_scheduler)

    # Train + Eval
    train_metrics = trainer.train().metrics
    # eval_metrics = trainer.evaluate()
    # "Test": use the dataset's held-out split; for SQuAD we use validation as test.
    # test_metrics = trainer.evaluate(eval_dataset=dataset_dict.get("test", dataset_dict["validation"]))
    test_dataset = dataset_dict.get("test", dataset_dict["validation"])

    if args.dataset == "ag_news":
        test_metrics = streaming_eval(model, test_dataset, tokenizer)
    else:
        # QA / SQuAD-style dataset
        # postproc = SquadPostProcessor(tokenizer)
        # dataloader = trainer.get_eval_dataloader(test_dataset)
        # test_metrics = postproc(trainer.model, dataloader, next(trainer.model.parameters()).device)
        test_metrics = eval_squad(model, test_dataset, tokenizer)

    # Persist logs
    summary = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler if args.optimizer == "adam" else "cosine-decay (default for non-Adam)",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": training_args.num_train_epochs,
        "max_steps": args.max_steps,
        "batch_train": args.per_device_train_batch_size,
        "batch_eval": args.per_device_eval_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "max_seq_len": args.max_seq_len,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "train_metrics": train_metrics,
        # "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
