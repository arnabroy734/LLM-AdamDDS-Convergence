import os, json, math, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorForSeq2Seq, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import DatasetDict
from data import build_datasets_and_tokenize
from torch.utils.data import DataLoader
from hessian import hessian_power_iteration_ultra_low_memory
import numpy as np
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def calculate_lr(model_name, dataset_name, device):  
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: 
    # Safe default for decoder-only models
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(   
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(  
        r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # Data
    if "llama" in model_name.lower(): 
        model_name = "llama"
    elif "qwen" in model_name.lower():
        model_name = "qwen"

    dataset_dict: DatasetDict = build_datasets_and_tokenize(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        model_name=model_name
    )

    # Collator (seq2seq-style text-to-text; works for both classification prompt and QA prompt)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict["train"],
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
    )
    loader = DataLoader(dataset_dict['train'], batch_size=1, collate_fn= collator, num_workers=10)
    L = -1.0
    total_loss = 0.0
    count = 0
    for i,batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if batch['input_ids'].shape[1] > 265: 
            print(batch['input_ids'].shape[1])
            continue
        L_new, loss = hessian_power_iteration_ultra_low_memory(model, trainer, batch)
        total_loss += loss 
        count += 1
        L = max(L_new, L)
        if (i+1)%10 == 0 or i == 0: 
            print(f"STEP: {i+1}: L = {L}")
    loss = total_loss/ count 
    lr = np.sqrt(2*loss/ L)

    with open(f'lr/{model_name}_{dataset_name}.json', 'w') as f: 
        res = {
            'avg_loss': loss,
            'T': 1, 
            'K' : L,
            'lr' : lr
        }   
        json.dump(res, f)
        f.close()  

if __name__ == "__main__":
    DATASET = ['ag_news', 'squad']
    MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
    for ds in DATASET:
        calculate_lr(MODEL, ds, 'cuda:1')