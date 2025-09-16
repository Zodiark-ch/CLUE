import random
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .Base import BaseDataset, UnlearnDataset


class Winogrande(BaseDataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset = defaultdict()
        train_dataset = load_dataset(
            "allenai/winogrande",
            "winogrande_debiased",
            split="train",
            cache_dir="./.cache/data",
        )

        dataset["train"] = train_dataset
        print(f"Train dataset: {len(dataset['train'])}")
        dataset["test"] = load_dataset(
            "allenai/winogrande",
            "winogrande_debiased",
            split="validation",
            cache_dir="./.cache/data",
        )

        return dataset

    def __preprocess__(self, tokenizer):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "label": []}
            
            # 构建text列表
            texts = []
            for i in range(len(examples['sentence'])):
                sample = {
                    'sentence': examples['sentence'][i],
                    'option1': examples['option1'][i],
                    'option2': examples['option2'][i],
                    'answer': examples['answer'][i]
                }
                
                label = sample['answer']
                question = sample['sentence']
                answer = sample['option1'] if label == 1 else sample['option2']
                
                sentence = f"{question}\nShould the '_' be {answer}?\nAnswer: Yes"
                sentence_without_answer = f"{question}\nShould the '_' be {answer}?\nAnswer: "
                tokenized = tokenizer(
                    sentence,
                    truncation=True,
                    padding="max_length",
                    max_length=200,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                    # 正确处理input_ids和label的对应关系
                full_input_ids = tokenized.input_ids[0]
                full_attention_mask = tokenized.attention_mask[0]
                
                # 找到最后一个非padding token的位置
                last_non_padding_idx = (full_attention_mask == 1).nonzero()[-1].item()
                
                # input_ids: 去掉最后一个非padding token
                input_ids = full_input_ids[:last_non_padding_idx]
                # label: 去掉BOS token之后的第一个token，保留BOS，去掉第二个token
                # 即：[BOS, token3, token4, ..., last_token]
                label = torch.cat([full_input_ids[:1], full_input_ids[2:last_non_padding_idx + 1]])
                # attention_mask: 对应input_ids的长度
                attention_mask = full_attention_mask[:last_non_padding_idx]
                
                # 直接padding到固定长度
                max_len = 200  # 可以根据需要调整
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                
                # Padding input_ids
                input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - len(input_ids))
                
                # Padding label (使用-100作为padding，因为-100在损失计算中会被忽略)
                label_padded = label.tolist() + [-100] * (max_len - len(label))
                
                # Padding attention_mask
                attention_mask_padded = attention_mask.tolist() + [0] * (max_len - len(attention_mask))
            
                results["input_ids"].append(torch.tensor(input_ids_padded))
                results["attention_mask"].append(torch.tensor(attention_mask_padded))
                results["label"].append(torch.tensor(label_padded))
            
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["sentence", "option1", "option2", "answer"]
        )
        test_dataset = self.dataset["test"].map(
            preprocess, batched=True, remove_columns=["sentence", "option1", "option2", "answer"]
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
