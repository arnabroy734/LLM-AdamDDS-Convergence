import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import torch
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import re
import string
from collections import Counter

def streaming_eval(model, test_dataset, tokenizer):
    # model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    # dataloader = trainer.get_eval_dataloader(dataset)  # respects collator/batching
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataset, desc="Streaming eval"):
            input_ids = batch['input_ids']
            input_ids = torch.tensor(input_ids, device=device).view((1,-1))
            label = int(batch['label'])

            pred = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            length = input_ids.shape[1]
            pred = tokenizer.decode(pred[0, length:], skip_special_tokens=True).strip()
            try:
                # print(f"Pred-{pred}")
                pred = int(pred)
                all_labels.append(label)
                all_preds.append(pred)
            except:
                print('SKIPPING PREDICTION')
                continue
         
        if len(all_labels) == 0:
            return {"accuracy": 0.0, "f1": 0.0}
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return {"accuracy": float(accuracy), "f1": float(f1)}


def eval_squad(model, test_dataset, tokenizer):
    # model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    # dataloader = trainer.get_eval_dataloader(dataset)  # respects collator/batching
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(test_dataset, desc="Streaming eval"):
            input_ids = batch['input_ids']
            input_ids = torch.tensor(input_ids, device=device).view((1,-1))
            gt_answer = batch['answers']

            pred_answer = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            length = input_ids.shape[1]
            pred_answer = tokenizer.decode(pred_answer[0, length:], skip_special_tokens=True).strip()
            # print("ACTUAL: ", gt_answer)
            # print("GENERATED: ", pred_answer)
            # if pred_answer == "":
            #     print(f"Empty answer")

            all_gts.append(gt_answer)
            all_preds.append(pred_answer)
        return squad_metric(all_preds, all_gts)



def squad_metric(pred_answers, gt_answers, tokenizer=None):
    """
    Calculate Exact Match (EM) and F1 score for SQuAD-style QA on batch of answers.
    
    Args:
        pred_answers (list): List of model generated answers
        gt_answers (list): List of ground truth answers
        tokenizer: Optional tokenizer (not required)
    
    Returns:
        dict: {'em': average_em, 'f1': average_f1, 'individual_scores': list}
    """
    
    def normalize_answer(s):
        """Normalize answer string"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch if ch not in exclude else ' ' for ch in text)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def f1_score(prediction, ground_truth):
        """Calculate F1 score based on token overlap"""
        pred_tokens = normalize_answer(prediction).split()
        truth_tokens = normalize_answer(ground_truth).split()
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def exact_match_score(prediction, ground_truth):
        """Calculate exact match score"""
        return normalize_answer(prediction) == normalize_answer(ground_truth)
    
    if len(pred_answers) != len(gt_answers):
        raise ValueError(f"Length mismatch: {len(pred_answers)} predictions vs {len(gt_answers)} ground truths")
    
    individual_scores = []
    total_em = 0.0
    total_f1 = 0.0
    
    for pred, gt in zip(pred_answers, gt_answers):
        if isinstance(gt, list):
            gt_list = gt
        else:
            gt_list = [gt]
        
        curr_em = max(exact_match_score(pred, ans) for ans in gt_list)
        curr_f1 = max(f1_score(pred, ans) for ans in gt_list)
        
        individual_scores.append({'em': float(curr_em), 'f1': curr_f1})
        total_em += float(curr_em)
        total_f1 += curr_f1
    
    avg_em = total_em / len(pred_answers)
    avg_f1 = total_f1 / len(pred_answers)
    
    return {
        'em': avg_em,
        'f1': avg_f1,
        # 'individual_scores': individual_scores
    }
