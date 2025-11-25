# LLM-AdamDDS-Convergence

## Problem Statement

Modern large language models (LLMs) rely on adaptive optimizers such as Adam for effective fine-tuning, where the choice of stepsize (learning rate) plays a critical role in both convergence speed and final performance. Traditional stepsize schedulers (linear, exponential decay, etc.) often require costly manual tuning and may lead to instability or suboptimal results, particularly in non-convex landscapes encountered in deep learning. **"On Convergence of Adam with Data Dependent Stepsize"** presents a theoretically grounded constant stepsize scheme that adapts to the dynamics of the data and model, aiming to reduce the need for exhaustive hyperparameter search and mitigate issues due to rapidly decaying rates. This repository implements the proposed stepsize technique in the context of LLM fine-tuning, benchmarks it against conventional and adaptive optimizers, and reports empirical results on both classification and question answering tasks.

---

## LLM Finetuning: Models, Datasets, and Experiment Design

- **Models Evaluated:**  
  - LLaMA-3.2-3B-Instruct  
  - Qwen-3-4B-Instruct

- **Datasets Used:**  
  - SQuAD (QA/generation)  
  - AG News (classification)

- **Optimizers & Schedulers:**  
  - Adam and variants: AdaBelief, AdaBound, AdamW, Yogi, RAdam  
  - Learning rate schedulers: constant, linear, cosine, exponential, square root, and inverse time decay

- **Fine-Tuning Procedure:**  
  All experiments follow a controlled LoRA adapter protocol on attention layers, with batch size 16 for 3125 training steps.

- **Running Experiments:**  
  - Run all experiments:
    ```
    bash run.sh
    ```
  - Models and datasets can be configured via arguments in `run.sh`.

---

## Analytical Stepsize Calculation

To compute the recommended data-dependent stepsize:
```
python lr_calculate.py
```
This calculates the spectral norm-based Lipschitz estimation (learning rate finder) as per the paper's algorithm.

---

## Constant Learning Rate Experiments

To fine-tune with Adam (constant stepsize):
```
const_lr.sh
```
This script benchmarks Adam on the selected LLM/dataset with the paper's analytically derived stepsize.

---

## Results

<!-- The first, second, and third best performing stepsizes are colored in **red**, **green**, and **brown** respectively. -->

### Derived Stepsizes for LLM LoRA Fine-Tuning

| Model                   | Ag News      | SQuAD       |
|-------------------------|-------------|-------------|
| LLaMA-3.2-3B-Instruct   | 3.2×10⁻⁴    | 3.7×10⁻⁴    |
| Qwen-3-4B-Instruct      | 3.8×10⁻⁴    | 5.7×10⁻⁴    |

### SQuAD – Exact Match & F1 (Adam, Various Schedulers)

| Step       | Linear | Cosine | Exponential | Sqrt | Inv-time | α_ours | 2×α_ours | α_ours/2 |
|------------|--------|--------|-------------|------|----------|--------|----------|----------|
| **Exact Match LLaMA** | 0.7061 | 0.7034 | 0.7096 | 0.7013 | 0.7032 | 0.6998 | 0.6941 | **0.7141** |
| **Exact Match Qwen**  | 0.7173 | 0.7153 | **0.7189** | 0.7108 | 0.7123 | 0.7131 | 0.7135 | 0.6965 | **0.7176** |
| **F1 LLaMA**          | 0.8546 | 0.8531 | 0.8572 | 0.8518 | 0.8523 | 0.8529 | 0.8508 | 0.8408 | **0.8585** |
| **F1 Qwen**           | 0.8661 | 0.8642 | 0.8653 | **0.8728** | 0.8612 | 0.8724 | 0.8575 | 0.8577 | **0.8660** |

### AG News – Accuracy & F1 (Adam, Various Schedulers)

| Step       | Linear | Cosine | Exponential | Sqrt | Inv-time | α_ours | 2×α_ours | α_ours/2 |
|------------|--------|--------|-------------|------|----------|--------|----------|----------|
| **Accuracy LLaMA** | 0.9164 | 0.9154 | **0.9169** | 0.9145 | 0.9162 | 0.9161 | 0.9085 | 0.8959 | **0.9166** |
| **Accuracy Qwen**  | 0.9128 | 0.9127 | **0.9130** | 0.9126 | 0.9124 | 0.9126 | 0.9117 | 0.8813 | **0.9132** |
| **F1 LLaMA**       | 0.9161 | 0.9159 | **0.9163** | 0.9152 | 0.9153 | 0.9157 | 0.9077 | 0.8940 | **0.9162** |
| **F1 Qwen**        | 0.9132 | 0.9125 | **0.9130** | 0.9122 | 0.9126 | 0.9123 | 0.9108 | 0.8907 | **0.9134** |

### AG News – Accuracy & F1 (Cosine Scheduler, Modern Optimizers)

| Optimizer     | AdaBelief | AdaBound | AdamW | Yogi | RAdam | Adam α_ours | Adam 2×α_ours | Adam α_ours/2 |
|---------------|-----------|----------|-------|------|-------|-------------|---------------|---------------|
| **Accuracy LLaMA**  | **0.9343** | 0.8913   | **0.9355** | 0.9154 | 0.9307 | 0.9085      | 0.8959        | 0.9166        |
| **Accuracy Qwen**   | **0.9330** | 0.8823   | 0.9328 | 0.9098 | 0.9272 | 0.9117      | 0.8813        | 0.9132        |
| **F1 LLaMA**        | **0.9341** | 0.8910   | **0.9361** | 0.9181 | 0.9308 | 0.9077      | 0.8940        | 0.9162        |
| **F1 Qwen**         | **0.9324** | 0.8822   | 0.9332 | 0.9097 | 0.9267 | 0.9108      | 0.8907        | 0.9134        |

---

**For complete experiment details, see `run.sh`, `constant_lr.sh`, and all scripts in the repo. All experiments follow the theoretical framework in [On Convergence of Adam with Data Dependent Stepsize].**
