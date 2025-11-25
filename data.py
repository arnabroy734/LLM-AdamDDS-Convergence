# data.py
from typing import Dict
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
import numpy as np
MAX_SAMPLES=50000

# --------------------------
# Prompt templates & targets
# --------------------------

SYSTEM_PROMPT_AG = """
You are a classifier.You will be given a sentence and you have to classify the sentence out of `World`, `Sports`, `Business` and `Sci/Tech`
If your response if `World` output 0, if `Sports` output 1, if `Business` output 2 and if `Sci/Tech` output is 3. So your output will be 
strictly one out of 0, 1, 2 or 3. Do not generate any other token please.
"""

SYSTEM_PROMPT_SQUAD = """
You are a precise extractive question-answering assistant.
Always read the given context carefully and extract the answer directly from it.
The answer must be a short span or phrase copied exactly from the context.
If the answer cannot be found, output an empty string — do not say anything else.
"""


def build_datasets_and_tokenize(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str, 
) -> DatasetDict:
    """
    Build and tokenize the requested dataset for causal LM finetuning
    (text-to-text style), returning a DatasetDict with train/validation/test.
    Fields: input_ids, attention_mask, labels, and 'id' for SQuAD.
    """

    if dataset_name == "ag_news":
        # ------------------ AG NEWS (classification as generation) ------------------
        raw = load_dataset("ag_news")  # splits: train/test; cols: ['text','label']
        raw["train"] = raw["train"].select(range(MAX_SAMPLES))
        # raw["test"] = raw["test"].select(range(MAX_SAMPLES_AG))

        if model_name == "llama":
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
        elif model_name == "qwen":
            response_template = "<|im_start|>assistant\n<think>\n\n</think>"
        def format_ag_news_prompt(examples):
            inputs = examples["text"]
            labels = examples["label"]
            texts = [tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT_AG},
                        {"role": "user", "content": ip + " Classify the sentence"},
                        {"role": "assistant", "content": str(label)} 
                    ]
                    , tokenize = False, add_generation_prompt = False, add_eot_token=False) for ip, label  in zip(inputs, labels)]
            prompts = [tokenizer.apply_chat_template(
                        [
                        {"role": "system", "content": SYSTEM_PROMPT_AG},
                        {"role": "user", "content": ip + " Classify the sentence"}
                        ]
                    , tokenize = False, add_generation_prompt = False, add_eot_token=False) for ip  in inputs]
            prompts = [prompt+response_template for prompt in prompts]
    
            return {"input_ids": texts, "prompts": prompts}

        def format_ag_news_prompt_test(examples):
            inputs = examples["text"]
            texts = [tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT_AG},
                        {"role": "user", "content": ip + " Classify the sentence"},
                    ]
                    , tokenize = False, add_generation_prompt = False, add_eot_token=False) for ip in inputs]
            texts = [text+response_template for text in texts]
            return {"input_ids": texts}

        def tok(examples):
            tokens = [
                tokenizer(ip)['input_ids'] for ip in examples["input_ids"]
            ]
            prompts = [
                tokenizer(ip)['input_ids'] for ip in examples["prompts"]
            ]
            labels = []
            for prompt, token in zip(prompts, tokens):
                n = len(prompt)
                label = np.array(token)
                label[:n] = -100
                labels.append(list(label))
            return {'input_ids': tokens, "labels": labels}

        def tok_test(examples):
            tokens = [
                tokenizer(ip)['input_ids'] for ip in examples["input_ids"]
            ]
            return {'input_ids': tokens}

        train = raw['train'].map(
            format_ag_news_prompt,
              # drop 'text','label'
            desc="Formatting AG News prompts train",
            batched=True
        )
        train = train.map(
            tok,
            remove_columns=train.column_names,
            desc="Tokenizing train",
            batched=True
        )
        test = raw['test'].map(
            format_ag_news_prompt_test,
            desc="Formatting AG News prompts test",
            batched=True
        )
        test = test.map(
            tok_test,
            remove_columns=['text'],
            desc="Tokenising test",
            batched=True
        )


        return DatasetDict(
            train=train,
            validation=test
        )
    
    if dataset_name == "squad":
        
        raw = load_dataset("squad")  # splits: train/validation; cols: ['id', 'title', 'context', 'question', 'answers']
        raw["train"] = raw["train"].select(range(MAX_SAMPLES))
        # raw["validation"] = raw["validation"].select(range(MAX_SAMPLES))

        if model_name == "llama":
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
        elif model_name == "qwen":
            response_template = "<|im_start|>assistant\n<think>\n\n</think>"
        def format_ag_news_prompt(examples):
            contexts = examples["context"]
            questions = examples["question"]
            answers = examples["answers"]

            texts = [tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT_SQUAD},
                        {"role": "user", "content": "Context-" + ctx + "Question-" + qs},
                        {"role": "assistant", "content": ans['text'][0]} 
                    ]
                    , tokenize = False, add_generation_prompt = False, add_eot_token=False) for ctx, qs, ans  in zip(contexts, questions, answers)]
            prompts = [tokenizer.apply_chat_template(
                        [
                        {"role": "system", "content": SYSTEM_PROMPT_SQUAD},
                        {"role": "user", "content": "Context-" + ctx + "Question-" + qs},
                        ]
                    , tokenize = False, add_generation_prompt = False, add_eot_token=False) for ctx, qs  in zip(contexts, questions)]
            prompts = [prompt+response_template for prompt in prompts]
    
            return {"input_ids": texts, "prompts": prompts}

        def format_ag_news_prompt_test(examples):
            contexts = examples["context"]
            questions = examples["question"]
            answers = examples["answers"]

            texts = [tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT_SQUAD},
                        {"role": "user", "content": "Context-" + ctx + "Question-" + qs},
                    ]
                    , tokenize = False, add_generation_prompt = False, add_eot_token=False) for ctx, qs in zip(contexts, questions)]
            texts = [text+response_template for text in texts]
            answers = [ans['text'][0] for ans in answers]
            return {"input_ids": texts, "answers": answers}

        def tok(examples):
            tokens = [
                tokenizer(ip)['input_ids'] for ip in examples["input_ids"]
            ]
            prompts = [
                tokenizer(ip)['input_ids'] for ip in examples["prompts"]
            ]
            labels = []
            for prompt, token in zip(prompts, tokens):
                n = len(prompt)
                label = np.array(token)
                label[:n] = -100
                labels.append(list(label))
            return {'input_ids': tokens, "labels": labels}

        def tok_test(examples):
            tokens = [
                tokenizer(ip)['input_ids'] for ip in examples["input_ids"]
            ]
            return {'input_ids': tokens}

        train = raw['train'].map(
            format_ag_news_prompt,
              # drop 'text','label'
            desc="Formatting SQUAD prompts train",
            batched=True
        )
        train = train.map(
            tok,
            remove_columns=train.column_names,
            desc="Tokenizing train",
            batched=True
        )
        test = raw['validation'].map(
            format_ag_news_prompt_test,
            desc="Formatting SQUAD prompts test",
            batched=True
        )
        test = test.map(
            tok_test,
            # remove_columns=['text'],
            desc="Tokenising test",
            batched=True
        )


        return DatasetDict(
            train=train,
            validation=test
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")