import random
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .Base import BaseDataset, UnlearnDataset


class SST2(BaseDataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset = defaultdict()
        train_dataset = load_dataset(
            "stanfordnlp/sst2",
            split="train",
            cache_dir="./.cache/data",
        )

        dataset["train"] = train_dataset
        print(f"Train dataset: {len(dataset['train'])}")
        dataset["test"] = load_dataset(
            "stanfordnlp/sst2",
            split="validation",
            cache_dir="./.cache/data",
        )

        return dataset

    def __preprocess__(self, tokenizer):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "label": []}
            
            # Build text list
            texts = []
            for i in range(len(examples['sentence'])):
                sample = {
                    'sentence': examples['sentence'][i],
                    'label': examples['label'][i]
                }
                
                label = sample['label']
                question = sample['sentence']
                answer = "positive" if label == 1 else "negative"
                
                sentence = ("Is the sentiment of following sentence positive or negative?"
                          f"{question}"
                          f"\nAnswer: It is {answer}")
                sentence_without_answer = ("Is the sentiment of following sentence positive or negative?"
                          f"{question}"
                          f"\nAnswer: It is ")
                tokenized = tokenizer(
                    sentence,
                    truncation=True,
                    padding="max_length",
                    max_length=200,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                full_input_ids = tokenized.input_ids[0]
                full_attention_mask = tokenized.attention_mask[0]
                
                # Find position of last non-padding token
                last_non_padding_idx = (full_attention_mask == 1).nonzero()[-1].item()
                
                # input_ids: remove last non-padding token
                input_ids = full_input_ids[:last_non_padding_idx]
                # label: remove first token after BOS token, keep BOS, remove second token
                # i.e.: [BOS, token3, token4, ..., last_token]
                label = torch.cat([full_input_ids[:1], full_input_ids[2:last_non_padding_idx + 1]])
                # attention_mask: corresponding to input_ids length
                attention_mask = full_attention_mask[:last_non_padding_idx]
                
                # Directly pad to fixed length
                max_len = 200  # Can be adjusted as needed
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                
                # Padding input_ids
                input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - len(input_ids))
                
                # Padding label (use -100 as padding, as -100 is ignored in loss calculation)
                label_padded = label.tolist() + [-100] * (max_len - len(label))
                
                # Padding attention_mask
                attention_mask_padded = attention_mask.tolist() + [0] * (max_len - len(attention_mask))
                results["input_ids"].append(torch.tensor(input_ids_padded))
                results["attention_mask"].append(torch.tensor(attention_mask_padded))
                results["label"].append(torch.tensor(label_padded))
            
            # Process texts using tokenizer
            
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["sentence", "label"]
        )
        test_dataset = self.dataset["test"].map(
            preprocess, batched=True, remove_columns=["sentence", "label"]
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        self.dataset["train"] = train_dataset
        self.dataset["test"] = test_dataset
        return self.dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)
        return self.dataset
