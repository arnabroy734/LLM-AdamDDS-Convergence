# optimizers.py
import math
import torch
from torch.optim import Adam, AdamW
import torch.optim.lr_scheduler as lrs

# 3rd-party optimizers
try:
    from adabelief_pytorch import AdaBelief
except Exception:
    AdaBelief = None
try:
    import torch_optimizer as topt  # has RAdam, Yogi, AdaBound, MSVAG
except Exception:
    topt = None
try:
    from lion_pytorch import Lion
except Exception:
    Lion = None

def build_optimizer(model, args):
    params = (p for p in model.parameters() if p.requires_grad)
    name = args.optimizer.lower()
    lr = args.learning_rate
    wd = args.weight_decay

    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    if name == "radam":
        assert topt is not None, "Install torch-optimizer for RAdam"
        return topt.RAdam(params, lr=lr, weight_decay=wd)
    if name == "yogi":
        assert topt is not None, "Install torch-optimizer for Yogi"
        return topt.Yogi(params, lr=lr, weight_decay=wd)
    if name == "adabound":
        assert topt is not None, "Install torch-optimizer for AdaBound"
        return topt.AdaBound(params, lr=lr, final_lr=lr*10, weight_decay=wd)
    if name == "msvag":
        assert topt is not None, "Install torch-optimizer for MSVAG"
        return topt.MSVAG(params, lr=lr, weight_decay=wd)
    if name == "adabelief":
        assert AdaBelief is not None, "Install adabelief-pytorch"
        return AdaBelief(params, lr=lr, weight_decay=wd, eps=1e-12, betas=(0.9, 0.999), rectify=False)
    if name == "lion":
        assert Lion is not None, "Install lion-pytorch"
        return Lion(params, lr=lr, weight_decay=wd, beta1=0.9, beta2=0.999)

    raise ValueError(f"Unknown optimizer: {args.optimizer}")

def build_scheduler(optimizer, training_args, args, total_steps: int):
    # For non-Adam optimizers: widely used in LLM finetuning is cosine with warmup
    if args.optimizer.lower() != "adam":
        return lrs.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    
    # Adam-specific scheduler choices
    name = (args.scheduler or "cosine-decay").lower()

    if name == "linear-decay":
        return lrs.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(1, total_steps)
        )
    if name == "cosine-decay":
        return lrs.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    if name == "step-decay":
        return lrs.StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_rate)
    if name == "exponential-decay":
        return lrs.ExponentialLR(optimizer, gamma=args.decay_rate)
    if name == "square-root-decay":
        # lr_t = lr0 / sqrt(1 + t)
        lam = lambda t: 1.0 / math.sqrt(1.0 + t)
        return lrs.LambdaLR(optimizer, lr_lambda=lam)
    if name == "inverse-time-decay":
        # lr_t = lr0 / (1 + k * t)
        k = args.inverse_k
        lam = lambda t: 1.0 / (1.0 + k * t)
        return lrs.LambdaLR(optimizer, lr_lambda=lam)
    if name == "constant":
        # lr_t = lr0 (no decay)
        lam = lambda t: 1.0
        return lrs.LambdaLR(optimizer, lr_lambda=lam)
    

    raise ValueError(f"Unknown scheduler: {args.scheduler}")
