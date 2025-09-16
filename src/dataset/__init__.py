from collections import defaultdict

import torch
from transformers import default_data_collator

from .Base import UnlearnDataset, unlearncollector
from .C4 import C4
from .HorryPotter import HP
from .SafePku import SafePkuDataset
from .Tofu import ToFU
from .wmdp import WMDPBio, WMDPCyber, WMDPALL
from .wikitext2 import wikitext
from .ioi_dataset import IOIDataset
from .docstring import docstring_induction_prompt_generator
from .gender import load_datasets
from .winogrande import Winogrande
from .sst2 import SST2
import huggingface_hub
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
# IOI数据集包装器，用于将IOI数据集转换为与其他数据集兼容的格式
class IOIDatasetWrapper:
    def __init__(self, ioi_dataset, target_tokenizer):
        self.ioi_dataset = ioi_dataset
        self.target_tokenizer = target_tokenizer
        self.length = len(ioi_dataset)
        
        # 使用GPT-2 tokenizer生成原始数据，然后转换为目标tokenizer格式
        self.converted_data = []
        for i in range(self.length):
            # 获取原始文本
            original_text = self.ioi_dataset.ioi_prompts[i]["text"]
            
            # 使用目标tokenizer重新tokenize
            tokenized = self.target_tokenizer(
                original_text,
                truncation=True,
                padding="max_length",
                max_length=30,
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
            max_len = 50  # 可以根据需要调整
            pad_token_id = self.target_tokenizer.pad_token_id if self.target_tokenizer.pad_token_id is not None else 0
            
            # Padding input_ids
            input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - len(input_ids))
            
            # Padding label (使用-100作为padding，因为-100在损失计算中会被忽略)
            label_padded = label.tolist() + [-100] * (max_len - len(label))
            
            # Padding attention_mask
            attention_mask_padded = attention_mask.tolist() + [0] * (max_len - len(attention_mask))
            
            # 将padding后的数据存储
            self.converted_data.append({
                "input_ids": torch.tensor(input_ids_padded),
                "attention_mask": torch.tensor(attention_mask_padded),
                "label": torch.tensor(label_padded)
            })
            
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = IOIDatasetWrapper.__new__(IOIDatasetWrapper)
        new_wrapper.ioi_dataset = self.ioi_dataset
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.length = len(indices)
        # 直接复制原始数据，让__getitem__方法处理padding
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

# Induction数据集包装器，用于将induction数据集转换为与其他数据集兼容的格式
class InductionDatasetWrapper:
    def __init__(self, induction_dataset, target_tokenizer):
        self.induction_dataset = induction_dataset
        self.target_tokenizer = target_tokenizer
        self.length = len(induction_dataset)
        
        # 导入GPT-2 tokenizer用于解码
        from transformers import GPT2TokenizerFast
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('ArthurConmy/redwood_tokenizer')
        
        # 转换数据
        self.converted_data = []
        for i in range(self.length):
            # 1. 使用GPT-2 tokenizer将token tensor转换回文本
            gpt2_tokens = self.induction_dataset[i]
            original_text = self.gpt2_tokenizer.decode(gpt2_tokens, skip_special_tokens=True)
            
            # 2. 使用目标tokenizer重新tokenize
            tokenized = self.target_tokenizer(
                original_text,
                truncation=True,
                padding="max_length",
                max_length=600,
                add_special_tokens=True,
                return_tensors="pt"
            )
            
            # 3. 正确处理input_ids和label的对应关系
            full_input_ids = tokenized.input_ids[0]
            full_attention_mask = tokenized.attention_mask[0]
            
            # 找到最后一个非padding token的位置
            last_non_padding_idx = (full_attention_mask == 1).nonzero()[-1].item()
            
            # input_ids: 去掉最后一个非padding token
            input_ids = full_input_ids[:last_non_padding_idx]
            # label: 去掉BOS token之后的第一个token，保留BOS，去掉第二个token
            # 即：[BOS, token3, token4, ..., last_token]
            label = torch.cat([full_input_ids[:1], full_input_ids[2:last_non_padding_idx + 1]])
            
            # 4. 使用新tokenizer生成的attention_mask，与IOIDatasetWrapper保持一致
            attention_mask = full_attention_mask[:last_non_padding_idx]
            
            # 直接padding到固定长度
            max_len = 600  # 可以根据需要调整
            pad_token_id = self.target_tokenizer.pad_token_id if self.target_tokenizer.pad_token_id is not None else 0
            
            # Padding input_ids
            input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - len(input_ids))
            
            # Padding label (使用-100作为padding，因为-100在损失计算中会被忽略)
            label_padded = label.tolist() + [-100] * (max_len - len(label))
            
            # Padding attention_mask
            attention_mask_padded = attention_mask.tolist() + [0] * (max_len - len(attention_mask))
            
            # 将padding后的数据存储
            self.converted_data.append({
                "input_ids": torch.tensor(input_ids_padded),
                "attention_mask": torch.tensor(attention_mask_padded),
                "label": torch.tensor(label_padded)
            })
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = InductionDatasetWrapper.__new__(InductionDatasetWrapper)
        new_wrapper.induction_dataset = self.induction_dataset
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.length = len(indices)
        # 直接复制原始数据，让__getitem__方法处理padding
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

# Docstring数据集包装器，用于将docstring数据集转换为与其他数据集兼容的格式
class DocstingDatasetWrapper:
    def __init__(self, docstring_data, target_tokenizer):
        self.docstring_data = docstring_data
        self.target_tokenizer = target_tokenizer
        self.length = len(docstring_data)
        
        # 转换数据
        self.converted_data = []
        for i in range(self.length):
            # 1. 构建原始文本：clean_prompt + correct_answers[0]
            prompt_item = self.docstring_data[i]
            clean_prompt = prompt_item.clean_prompt
            correct_answer = prompt_item.correct_answers[0]
            original_text = clean_prompt + correct_answer
            
            # 2. 使用目标tokenizer重新tokenize
            tokenized = self.target_tokenizer(
                original_text,
                truncation=True,
                padding="max_length",
                max_length=50,
                add_special_tokens=True,
                return_tensors="pt"
            )
            
            # 3. 正确处理input_ids和label的对应关系
            full_input_ids = tokenized.input_ids[0]
            full_attention_mask = tokenized.attention_mask[0]
            
            # 找到最后一个非padding token的位置
            last_non_padding_idx = (full_attention_mask == 1).nonzero()[-1].item()
            
            # input_ids: 去掉最后一个非padding token
            input_ids = full_input_ids[:last_non_padding_idx]
            # label: 去掉BOS token之后的第一个token，保留BOS，去掉第二个token
            # 即：[BOS, token3, token4, ..., last_token]
            label = torch.cat([full_input_ids[:1], full_input_ids[2:last_non_padding_idx + 1]])
            
            # 4. 使用新tokenizer生成的attention_mask
            attention_mask = full_attention_mask[:last_non_padding_idx]
            
            # 直接padding到固定长度
            max_len = 50  # 可以根据需要调整
            pad_token_id = self.target_tokenizer.pad_token_id if self.target_tokenizer.pad_token_id is not None else 0
            
            # Padding input_ids
            input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - len(input_ids))
            
            # Padding label (使用-100作为padding，因为-100在损失计算中会被忽略)
            label_padded = label.tolist() + [-100] * (max_len - len(label))
            
            # Padding attention_mask
            attention_mask_padded = attention_mask.tolist() + [0] * (max_len - len(attention_mask))
            
            # 将padding后的数据存储
            self.converted_data.append({
                "input_ids": torch.tensor(input_ids_padded),
                "attention_mask": torch.tensor(attention_mask_padded),
                "label": torch.tensor(label_padded)
            })
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = DocstingDatasetWrapper.__new__(DocstingDatasetWrapper)
        new_wrapper.docstring_data = self.docstring_data
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.length = len(indices)
        # 直接复制原始数据，让__getitem__方法处理padding
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

# Gender数据集包装器，用于将gender数据集转换为与其他数据集兼容的格式
class GenderDatasetWrapper:
    def __init__(self, gender_data, target_tokenizer):
        self.gender_data = gender_data
        self.target_tokenizer = target_tokenizer
        self.length = len(gender_data)
        
        # 转换数据
        self.converted_data = []
        for i in range(self.length):
            # 1. 构建输入样本：prefix + 空格 + pronoun + "?" + 换行 + corr_prefix + 空格 + corr_pronoun
            data_item = self.gender_data[i]
            original_text = f"{data_item['prefix']}" " " f"{data_item['pronoun']}"
            
            # 2. 使用目标tokenizer重新tokenize
            tokenized = self.target_tokenizer(
                original_text,
                truncation=True,
                padding="max_length",
                max_length=50,
                add_special_tokens=True,
                return_tensors="pt"
            )
            
            # 3. 正确处理input_ids和label的对应关系
            full_input_ids = tokenized.input_ids[0]
            full_attention_mask = tokenized.attention_mask[0]
            
            # 找到最后一个非padding token的位置
            last_non_padding_idx = (full_attention_mask == 1).nonzero()[-1].item()
            
            # input_ids: 去掉最后一个非padding token
            input_ids = full_input_ids[:last_non_padding_idx]
            # label: 去掉BOS token之后的第一个token，保留BOS，去掉第二个token
            # 即：[BOS, token3, token4, ..., last_token]
            label = torch.cat([full_input_ids[:1], full_input_ids[2:last_non_padding_idx + 1]])
            
            # 4. 使用新tokenizer生成的attention_mask
            attention_mask = full_attention_mask[:last_non_padding_idx]
            
            # 直接padding到固定长度
            max_len = 50  # 可以根据需要调整
            pad_token_id = self.target_tokenizer.pad_token_id if self.target_tokenizer.pad_token_id is not None else 0
            
            # Padding input_ids
            input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - len(input_ids))
            
            # Padding label (使用-100作为padding，因为-100在损失计算中会被忽略)
            label_padded = label.tolist() + [-100] * (max_len - len(label))
            
            # Padding attention_mask
            attention_mask_padded = attention_mask.tolist() + [0] * (max_len - len(attention_mask))
            
            # 将padding后的数据存储
            self.converted_data.append({
                "input_ids": torch.tensor(input_ids_padded),
                "attention_mask": torch.tensor(attention_mask_padded),
                "label": torch.tensor(label_padded)
            })
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = GenderDatasetWrapper.__new__(GenderDatasetWrapper)
        new_wrapper.gender_data = self.gender_data
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.length = len(indices)
        # 直接复制原始数据，让__getitem__方法处理padding
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

# Bool数据集包装器，用于将gender数据集转换为与其他数据集兼容的格式
class BoolDatasetWrapper:
    def __init__(self, bool_data, target_tokenizer):
        self.bool_data = bool_data
        self.target_tokenizer = target_tokenizer
        self.length = len(bool_data)
        
        # 转换数据
        self.converted_data = []
        for i in range(self.length):
            # 1. 构建输入样本：prefix + 空格 + pronoun + "." + 换行 + corr_prefix + 空格 + corr_pronoun
            data_item = self.bool_data[i]
            ip = data_item["input"]
            ip = ip[:ip.rfind(" is")]
            original_text = f"[INST] <<SYS>>\nEvaluate the following boolean expression as either 'True' or 'False'.\n<</SYS>>\n\n" + \
                f"{ip} [/INST] "+f"{data_item['target']}"
            #original_text = f"{data_item['input']} {data_item['target']}.\n {data_item['corr_input']} {data_item['corr_target']}"
            
            # 2. 使用目标tokenizer重新tokenize
            tokenized = self.target_tokenizer(
                original_text,
                truncation=True,
                padding="max_length",
                max_length=100,
                add_special_tokens=True,
                return_tensors="pt"
            )
            
            # 3. 正确处理input_ids和label的对应关系
            full_input_ids = tokenized.input_ids[0]
            full_attention_mask = tokenized.attention_mask[0]
            
            # 找到最后一个非padding token的位置
            last_non_padding_idx = (full_attention_mask == 1).nonzero()[-1].item()
            
            # input_ids: 去掉最后一个非padding token
            input_ids = full_input_ids[:last_non_padding_idx]
            # label: 去掉BOS token之后的第一个token，保留BOS，去掉第二个token
            # 即：[BOS, token3, token4, ..., last_token]
            label = torch.cat([full_input_ids[:1], full_input_ids[2:last_non_padding_idx + 1]])
            
            # 4. 使用新tokenizer生成的attention_mask
            attention_mask = full_attention_mask[:last_non_padding_idx]
            
            # 直接padding到固定长度
            max_len = 100  # 可以根据需要调整
            pad_token_id = self.target_tokenizer.pad_token_id if self.target_tokenizer.pad_token_id is not None else 0
            
            # Padding input_ids
            input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - len(input_ids))
            
            # Padding label (使用-100作为padding，因为-100在损失计算中会被忽略)
            label_padded = label.tolist() + [-100] * (max_len - len(label))
            
            # Padding attention_mask
            attention_mask_padded = attention_mask.tolist() + [0] * (max_len - len(attention_mask))
            
            # 将padding后的数据存储
            self.converted_data.append({
                "input_ids": torch.tensor(input_ids_padded),
                "attention_mask": torch.tensor(attention_mask_padded),
                "label": torch.tensor(label_padded)
            })
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = GenderDatasetWrapper.__new__(GenderDatasetWrapper)
        new_wrapper.bool_data = self.bool_data
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.length = len(indices)
        # 直接复制原始数据，让__getitem__方法处理padding
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

def get_validation_data(num_examples=None, seq_len=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]

def get_mask_repeat_candidates(num_examples=None, seq_len=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname)
    mask_repeat_candidates.requires_grad = False

    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]

def get_dataset(
    dataset_names,
    tokenizer,
    dataset_seed,
    forget_ratio,
    self_retain=False,
    if_llama=False,
):
    ### forget dataset & test dataset
    if dataset_names["forget"] == "SafePku":
        dataset = SafePkuDataset("SafePku", if_llama=if_llama)
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "wikitext":
        dataset = wikitext("wikitext")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "HP":
        dataset = HP("HP")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif "Tofu" in dataset_names["forget"]:
        subset = dataset_names["forget"].split("_")[1]
        dataset = ToFU("TOFU", subset=subset, if_llama=if_llama)
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPCyber":
        dataset = WMDPCyber("WMDPCyber", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPBio":
        dataset = WMDPBio("WMDPBio", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPALL":
        dataset = WMDPALL("WMDPALL", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "IOI":
        # 创建IOI数据集作为forget数据集，使用GPT-2 tokenizer生成数据
        ioi_dataset = IOIDataset(
            prompt_type="ABBA",
            N=200,
            nb_templates=1,
            seed=dataset_seed,
            tokenizer=None,  # 使用默认的GPT-2 tokenizer
        )
        
        forget_dataset = IOIDatasetWrapper(ioi_dataset, tokenizer)
        test_dataset = IOIDatasetWrapper(ioi_dataset, tokenizer)  # 使用相同的数据集作为测试集
    elif "forget" not in dataset_names:
        forget_dataset = None
        test_dataset = None
    else:
        raise ValueError("No dataset")

    #### retain dataset
    retain_datasets = {}
    test_datasets = {}
    
    # 检查retain是否是列表（多个数据集）
    if isinstance(dataset_names["retain"], list):
        retain_dataset_names = dataset_names["retain"]
    else:
        retain_dataset_names = [dataset_names["retain"]]
    
    for i, retain_name in enumerate(retain_dataset_names):
        if retain_name == "SafePku":
            dataset = SafePkuDataset("SafePku")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "C4":
            dataset = C4("C4")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "winogrande":
            dataset = Winogrande("winogrande")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "sst2":
            dataset = SST2("sst2")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "wikitext":
            dataset = wikitext("wikitext")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif "Tofu" in retain_name:
            subset = retain_name.split("_")[1]
            dataset = ToFU("TOFU", subset=subset, if_llama=if_llama)
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "WMDPCyber":
            dataset = WMDPCyber("WMDPCyber", subset="retain")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "WMDPBio":
            dataset = WMDPBio("WMDPBio", subset="retain")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "WMDPALL":
            dataset = WMDPALL("WMDPALL", subset="retain")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "IOI":
            # 创建IOI数据集，使用GPT-2 tokenizer生成数据
            ioi_dataset = IOIDataset(
                prompt_type="ABBA",
                N=2400,
                nb_templates=1,
                seed=dataset_seed,
                tokenizer=None,  # 使用默认的GPT-2 tokenizer
            )
            ioi_dataset_test=IOIDataset(
                prompt_type="ABBA",
                N=600,
                nb_templates=1,
                seed=dataset_seed,
                tokenizer=None,  # 使用默认的GPT-2 tokenizer
            )
            
            retain_datasets[f"retain{i+1}"] = IOIDatasetWrapper(ioi_dataset, tokenizer)
            test_datasets[f"test{i+1}"] = IOIDatasetWrapper(ioi_dataset_test, tokenizer)
        elif retain_name == "induction":
            induction_dataset=get_validation_data(num_examples=3000)
            validation_slice = slice(0, 2400)
            test_slice = slice(2400, 3000)
            validation_data = induction_dataset[validation_slice, :].contiguous()
            test_data = induction_dataset[test_slice, :].contiguous()
            retain_datasets[f"retain{i+1}"] = InductionDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = InductionDatasetWrapper(test_data, tokenizer)
        elif retain_name == "docstring":
            docstring_ind_prompt_kwargs = dict(
            n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
        )
            raw_prompts = [
            docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=j)
            for j in range(3000)
        ]
            # 划分数据：前800个作为validation_data，后200个作为test_data
            validation_data = raw_prompts[:2400]
            test_data = raw_prompts[2400:3000]
            retain_datasets[f"retain{i+1}"] = DocstingDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = DocstingDatasetWrapper(test_data, tokenizer)
        elif retain_name == "gender":
            dataset=load_datasets("/home/lthpc/hangc/CSAT/files/data/gp", 3000, 600)
            validation_data=dataset["train_3k"]
            test_data=dataset["test"]
            retain_datasets[f"retain{i+1}"] = GenderDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = GenderDatasetWrapper(test_data, tokenizer)
        elif retain_name == "bool":
            dataset=load_datasets("/home/lthpc/hangc/CSAT/files/data/boolean_expressions", 3000, 600)
            validation_data=dataset["train"]
            test_data=dataset["test"]
            retain_datasets[f"retain{i+1}"] = BoolDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = BoolDatasetWrapper(test_data, tokenizer)
        elif "retain" not in dataset_names:
            retain_datasets[f"retain{i+1}"] = None
            test_datasets[f"test{i+1}"] = None
        else:
            raise ValueError(f"No dataset: {retain_name}")

    # 如果没有retain数据集，设置为None
    if not retain_datasets:
        retain_datasets = {"retain": None}
        test_datasets = {"test": None}
    elif len(retain_datasets) == 1:
        # 如果是单个retain数据集，使用"retain"键名以保持向后兼容性
        single_retain_key = list(retain_datasets.keys())[0]
        single_test_key = list(test_datasets.keys())[0]
        retain_datasets = {"retain": retain_datasets[single_retain_key]}
        test_datasets = {"test": test_datasets[single_test_key]}

    #### downstream datasets
    downstream_datasets = {}
    downstream_dataset_names = ["induction", "IOI", "bool", "gender", "docstring","winogrande","sst2"]
    
    # 获取所有retain数据集名称（包括多个数据集的情况）
    all_retain_names = []
    if isinstance(dataset_names["retain"], list):
        all_retain_names = dataset_names["retain"]
    else:
        all_retain_names = [dataset_names["retain"]]
    
    # 为每个不在retain中的downstream数据集创建test_data
    for downstream_name in downstream_dataset_names:
        if downstream_name not in all_retain_names:
            if downstream_name == "induction":
                induction_dataset = get_validation_data(num_examples=3000)
                test_slice = slice(2400, 3000)
                test_data = induction_dataset[test_slice, :].contiguous()
                downstream_datasets[f"downstream_{downstream_name}"] = InductionDatasetWrapper(test_data, tokenizer)
            elif downstream_name == "IOI":
                ioi_dataset_test = IOIDataset(
                    prompt_type="ABBA",
                    N=600,
                    nb_templates=1,
                    seed=dataset_seed,
                    tokenizer=None,
                )
                downstream_datasets[f"downstream_{downstream_name}"] = IOIDatasetWrapper(ioi_dataset_test, tokenizer)
            elif downstream_name == "bool":
                dataset = load_datasets("/home/lthpc/hangc/CSAT/files/data/boolean_expressions", 3000, 600)
                test_data = dataset["test"]
                downstream_datasets[f"downstream_{downstream_name}"] = BoolDatasetWrapper(test_data, tokenizer)
            elif downstream_name == "gender":
                dataset = load_datasets("/home/lthpc/hangc/CSAT/files/data/gp", 3000, 600)
                test_data = dataset["test"]
                downstream_datasets[f"downstream_{downstream_name}"] = GenderDatasetWrapper(test_data, tokenizer)
            elif downstream_name == "docstring":
                docstring_ind_prompt_kwargs = dict(
                    n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
                )
                raw_prompts = [
                    docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=j)
                    for j in range(3000)
                ]
                test_data = raw_prompts[2400:3000]
                downstream_datasets[f"downstream_{downstream_name}"] = DocstingDatasetWrapper(test_data, tokenizer)

    unlearn_dataset = UnlearnDataset(
        {"forget": forget_dataset, **retain_datasets},
        forget_ratio,
        dataset_seed,
        self_retain,
    )
    unlearn_collator = unlearncollector

    test_collator = default_data_collator

    return unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_names = {"forget": "SafePku", "retain": "BookCorpus"}
    dataset_seed = 8888
    forget_ratio = 0.1
    self_retain = False
    unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets = get_dataset(
        dataset_names, tokenizer, dataset_seed, forget_ratio, self_retain
    )
    print(len(unlearn_dataset))

    print(f"测试数据集数量: {len(test_datasets)}")
    for key, dataset in test_datasets.items():
        if dataset is not None:
            print(f"{key}: {len(dataset)}")
        else:
            print(f"{key}: None")
    import torch

    dataloader = torch.utils.data.DataLoader(
        unlearn_dataset, batch_size=2, collate_fn=unlearn_collator
    )
    for batch in dataloader:
        print(batch)
        break
