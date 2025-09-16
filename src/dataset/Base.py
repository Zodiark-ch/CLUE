import random

import torch
from torch.utils.data import Dataset


class BaseDataset:
    def __init__(self, dataset_name, with_retain=False, if_llama=False):
        self.dataset_name = dataset_name
        self.with_normal = with_retain
        self.if_llama = if_llama
        self.question_start_token = "### Question: "
        self.question_end_token = "\n"
        self.answer_start_token = "### Answer: "
    def get_dataset(self):
        pass

    def __preprocess__(self, tokenizer, forget_ratio, dataset_seed):
        pass

    def build_dataset(self, tokenizer, forget_ratio, dataset_seed):
        pass


class UnlearnDataset(Dataset):
    def __init__(self, datasets, forget_ratio, dataset_seed, self_retain=False):
        self.forget_ratio = forget_ratio
        self.dataset_seed = dataset_seed
        self.self_retain = self_retain

        if "forget" in datasets.keys():
            self.forget_dataset = datasets["forget"]
        else:
            self.forget_dataset = None

        # Handle multiple retain datasets
        self.retain_datasets = {}
        for key, value in datasets.items():
            if key.startswith("retain"):
                self.retain_datasets[key] = value
        
        # If no retain datasets, set to None
        if not self.retain_datasets:
            self.retain_datasets = {"retain": None}

        self.build_unlearn_dataset()

    def __len__(self):
        if self.forget_dataset:
            return len(self.forget_dataset)
        elif self.retain_datasets:
            # Return length of first non-None retain dataset
            for dataset in self.retain_datasets.values():
                if dataset is not None:
                    return len(dataset)
        else:
            raise ValueError("No dataset")

    def build_unlearn_dataset(self):
        if self.forget_dataset:
            if self.forget_ratio > 1:
                length = int(self.forget_ratio)

            elif self.forget_ratio <= 1 and self.forget_ratio > 0:
                length = int(len(self.forget_dataset) * self.forget_ratio)

            random.seed(self.dataset_seed)
            forget_index_list = random.sample(range(len(self.forget_dataset)), length)
            if self.self_retain:
                retain_index_list = list(
                    set(range(len(self.forget_dataset))) - set(forget_index_list)
                )
                # For multiple retain datasets, assign self_retain data to first retain dataset
                first_retain_key = list(self.retain_datasets.keys())[0]
                self.retain_datasets[first_retain_key] = self.forget_dataset.select(retain_index_list)
            self.forget_dataset = self.forget_dataset.select(forget_index_list)

    def __getitem__(self, idx):
        if self.forget_dataset:
            forget_data = self.forget_dataset[idx]
            if self.retain_datasets:
                # Randomly select a retain dataset
                available_retain_keys = [key for key, dataset in self.retain_datasets.items() if dataset is not None]
                if available_retain_keys:
                    selected_retain_key = random.choice(available_retain_keys)
                    selected_retain_dataset = self.retain_datasets[selected_retain_key]
                    retain_idx = random.randint(0, len(selected_retain_dataset) - 1)
                    retain_data = selected_retain_dataset[retain_idx]
                    
                    # If single retain dataset with key "retain", maintain backward compatibility
                    if len(self.retain_datasets) == 1 and "retain" in self.retain_datasets:
                        return {"forget": forget_data, "retain": retain_data}
                    else:
                        return {"forget": forget_data, selected_retain_key: retain_data}
                else:
                    return {"forget": forget_data, "retain": None}
            else:
                return {"forget": forget_data, "retain": None}
        else:
            # If no forget dataset, get data from first available retain dataset
            for key, dataset in self.retain_datasets.items():
                if dataset is not None:
                    retain_data = dataset[idx]
                    # If single retain dataset with key "retain", maintain backward compatibility
                    if len(self.retain_datasets) == 1 and "retain" in self.retain_datasets:
                        return {"forget": None, "retain": retain_data}
                    else:
                        return {"forget": None, key: retain_data}
            raise ValueError("No available dataset")


def unlearncollector(samples):
    res = {}
    if samples[0]["forget"]:
        forget_samples = [sample["forget"] for sample in samples]
        res["forget"] = (
            torch.stack([sample["input_ids"] for sample in forget_samples]),
            torch.stack([sample["attention_mask"] for sample in forget_samples]),
            torch.stack([sample["label"] for sample in forget_samples]),
            torch.stack([sample["refused_label"] for sample in forget_samples]),
            torch.stack([sample["question_length"] for sample in forget_samples]),
        )
    else:
        res["forget"] = None
    
    # Handle multiple retain datasets
    retain_keys = []
    for key in samples[0].keys():
        if key.startswith("retain"):
            retain_keys.append(key)
    
    # If no retain keys found, try backward compatible "retain" key
    if not retain_keys and "retain" in samples[0]:
        retain_keys = ["retain"]
    
    for retain_key in retain_keys:
        if samples[0][retain_key]:
            retain_samples = [sample[retain_key] for sample in samples]
            res[retain_key] = (
                torch.stack([sample["input_ids"] for sample in retain_samples]),
                torch.stack([sample["attention_mask"] for sample in retain_samples]),
                torch.stack([sample["label"] for sample in retain_samples]),
            )
        else:
            res[retain_key] = None
    
    return res
