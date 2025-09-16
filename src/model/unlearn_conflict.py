import os
import sys

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from peft import  get_peft_model, LoraConfig
from pruner.utils import WrappedGPT, find_layers
from dataset import get_dataset
from metrics import (
    eval_copyright,
    eval_few_shots,
    eval_PII,
    eval_ppl,
    eval_tofu,
    eval_toxic,
    eval_wmdp,
)
from metrics.simple_accuracy import eval_acc
from optim import create_sophia_optimizer
from unlearn import GenerateMask, get_unlearn_method

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # 冲突学习参数
        self.safe_unlearn_method = kwargs["safe_unlearn_method"]
        self.conflict_unlearn_method = kwargs["conflict_unlearn_method"]
        self.safe_mask_path = kwargs["safe_mask_path"]
        self.conflict_mask_path = kwargs["conflict_mask_path"]
        self.alternate_frequency = kwargs.get("alternate_frequency", 1)
        
        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)  # 添加梯度裁剪参数
        self.alpha = kwargs.get("alpha", None)
        self.gamma = kwargs.get("gamma", None)
        self.task_name = kwargs.get("task_name", None)
        self.k = kwargs.get("k", 100)
        self.sophia = kwargs.get("sophia", False)
        self.betas_low = kwargs.get("betas_low", 0.9)
        self.betas_high = kwargs.get("betas_high", 0.95)
        self.betas = (self.betas_low, self.betas_high)
        self.rho = kwargs.get("rho", 0.03)
        self.p = kwargs.get("p", 0.0)
        self.q = kwargs.get("q", 0.0)
        self.if_llama = "llama" in self.model_name
        self.resume_path = kwargs.get("resume_path", None)
        self.max_steps = kwargs.get("max_steps", -1)
        self.use_lora = kwargs.get("use_lora", False)
        self.if_wanda = False
        self.mu = kwargs.get("mu", 1e-3)
    
    def _move_mask_to_device(self, mask, if_wanda, mask_name):
        """将mask移动到正确的设备上"""
        if mask is None:
            return
        
        # 直接使用wanda类型的处理方式（数字索引作为键）
        print(f"处理{mask_name} mask，使用数字索引作为键")
        try:
            layers = self.model.model.layers
        except:
            layers = self.model.model.decoder.layers
        cnt = 0
        with torch.no_grad():
            for layer in layers:
                subset = find_layers(layer)
                for name in subset:
                    if cnt in mask:
                        mask[cnt] = mask[cnt].to(subset[name].weight.device)
                    cnt += 1
    def init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        if self.use_lora:
            peft_config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=["q_proj","v_proj"], 
                lora_dropout=0.05,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())

        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:#如果是llama系列的话用pad，不是的话用eos（这个还得看具体的模型）
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))#因为添加了pad，所以需要保证embedding大小和tokenizer大小相同
        self.tokenizer = tokenizer
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs for DataParallel!")
        #     self.model = torch.nn.DataParallel(self.model)
        # try:
        #     self.device = torch.device("cuda:0")
        #     #self.device = model.hf_device_map["lm_head"]
        # except:
        #     self.device = torch.device("cuda:0")
        # self.model = self.model.to(self.device)

    def init_dataset(self):
        unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_datasets = test_datasets
        self.downstream_datasets = downstream_datasets
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        if self.max_steps == -1:
            self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
            self.steps_per_epoch = len(unlearn_dataset) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
        else:
            self.steps_per_epoch = self.max_steps // self.num_epochs

    def init_unlearner(self, logger):
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        
        # 冲突学习模式：创建两个unlearner
        self._init_conflict_unlearners(logger)
    
    def _init_conflict_unlearners(self, logger):
        """初始化冲突学习的两个unlearner"""
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            output_dir=unlearn_checkpoint,
            optim="adamw_torch",
            save_steps=self.max_steps,
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            save_total_limit=1,
            report_to=[], 
        )
        
        # 创建safe unlearner
        if self.optimizer is not None:
            self.safe_unlearner = get_unlearn_method(
                name=self.safe_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.safe_mask,
                optimizers=(self.optimizer, None),
                if_wanda=True,
            )
        else:
            self.safe_unlearner = get_unlearn_method(
                name=self.safe_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.safe_mask,
                if_wanda=True,
            )
        
        # 创建conflict unlearner
        if self.optimizer is not None:
            self.conflict_unlearner = get_unlearn_method(
                name=self.conflict_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.conflict_mask,
                optimizers=(self.optimizer, None),
                if_wanda=True,
            )
        else:
            self.conflict_unlearner = get_unlearn_method(
                name=self.conflict_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.conflict_mask,
                if_wanda=True,
            )

    def init_mask(self, logger):
        # 初始化两个mask
        self.safe_mask = None
        self.conflict_mask = None
        
        # 加载safe mask /data/zodiark/CSAT/conflict/safe_mask.pt
        if self.safe_mask_path is not None and os.path.exists(self.safe_mask_path):
            print(f"加载safe mask: {self.safe_mask_path}")
            self.safe_mask = torch.load(self.safe_mask_path)
            self._move_mask_to_device(self.safe_mask, None, "safe")
        
        # 加载conflict mask
        if self.conflict_mask_path is not None and os.path.exists(self.conflict_mask_path):
            print(f"加载conflict mask: {self.conflict_mask_path}")
            self.conflict_mask = torch.load(self.conflict_mask_path)
            self._move_mask_to_device(self.conflict_mask, None, "conflict")
        
        # 设置默认mask为safe_mask
        if self.safe_mask is not None:
            self.mask = self.safe_mask
        else:
            self.mask = None
            
        # 如果safe mask文件不存在，需要生成
        if self.safe_mask_path is not None and not os.path.exists(self.safe_mask_path):
            self._generate_mask(self.safe_mask_path, logger, "safe")
        
        # 如果conflict mask文件不存在，需要生成
        if self.conflict_mask_path is not None and not os.path.exists(self.conflict_mask_path):
            self._generate_mask(self.conflict_mask_path, logger, "conflict")
    
    def _generate_mask(self, mask_path, logger, mask_name):
        """生成mask文件"""
        parts = mask_path.split("/")
        score_type = parts[-2]
        if score_type == "wanda":
            if_wanda = True
        else:
            if_wanda = False

        ratio = float(parts[-1].split("_")[-1].split(".p")[0])
        root = logger.get_root()
        mask_dir = mask_path.replace(f"with_{ratio}.pt", "")
        if mask_dir == mask_path:
            mask_dir = mask_path.replace(f"with_{self.p}_{self.q}.pt", "")
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        mask_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            optim="adamw_torch",
            save_steps=self.steps_per_epoch,
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            save_total_limit=3,
            output_dir=mask_dir,
            report_to=[],
        )
        if score_type == "wanda":
            unlearn_dataset,_,_,_,_ = get_dataset(
                self.dataset_names,
                self.tokenizer,
                self.dataset_seed,
                128,
                self.self_retain,
                self.if_llama,
            )
            mask = GenerateMask(
                score_type=score_type,
                ratios=[ratio],
                mask_dir=mask_dir,
                model=self.model,
                data_collator=self.unlearn_collator,
                tokenizer=self.tokenizer,
                train_dataset=unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=mask_args,
                p=self.p,
                q=self.q,
                mu=self.mu,
            ).get_mask()
        else:
            mask = GenerateMask(
                score_type=score_type,
                ratios=[ratio],
                mask_dir=mask_dir,
                model=self.model,
                data_collator=self.unlearn_collator,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=mask_args,
                p=self.p,
                q=self.q,
                mu=self.mu,
            ).get_mask()
        if score_type == "snip_forget_reinit":
            mask = None
            os.system(f"rm -rf {mask_path}")
            return
        
        # 保存生成的mask
        torch.save(mask, mask_path)
        print(f"Generated {mask_name} mask saved to {mask_path}")
    def init_optimizer(self):
        if self.sophia:
            self.optimizer = create_sophia_optimizer(
                self.model,
                lr=self.lr,
                betas=self.betas,
                rho=self.rho,
                weight_decay=self.weight_decay,
            )
        else:
            # 创建标准的AdamW优化器
            from torch.optim import AdamW
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        
        # 确保根目录存在
        import os
        os.makedirs(root, exist_ok=True)
        
        if self.resume_path is not None:
            model_name = self.resume_path
        else:
            # 使用 checkpoints 路径，这是保存微调后模型的地方
            model_name = os.path.join(root, "checkpoints")
        if self.task_name == "downstream":
            if "WMDP" in self.dataset_names["forget"]:
                print("开始评估wmdp数据集...")
                eval_few_shots(model_name=model_name, task_list=["wmdp"], output_path=f"{root}/wmdp.json")
                torch.cuda.empty_cache()
                eval_few_shots(model_name=model_name,  task_list=["mmlu"],output_path=f"{root}/mmlu.json")
            torch.cuda.empty_cache()
            if self.dataset_names["forget"] == "SafePku":
                print("开始评估DETOX数据集...")
                eval_toxic(
                    model_name=model_name, output_dir=root, dataset=self.unlearn_dataset
                )
            torch.cuda.empty_cache()
            # 处理多个test数据集的情况
            if isinstance(self.test_datasets, dict) and len(self.test_datasets) > 1:
                # 多个test数据集
                data_num=0
                for test_key, test_dataset in self.test_datasets.items():
                    if test_dataset is not None:
                        print(f"评估数据集: {self.dataset_names['retain'][data_num]}")
                        data_num+=1
                        # 确保每个test数据集的输出目录存在
                        test_output_dir = f"{root}/{test_key}"
                        os.makedirs(test_output_dir, exist_ok=True)
                        eval_acc(model_name=model_name, retain_dataset=test_dataset, output_dir=test_output_dir, batch_size=8)
                        torch.cuda.empty_cache()
            else:
                # 单个test数据集（向后兼容）
                if isinstance(self.test_datasets, dict):
                    # 单个test数据集，键为"test"
                    test_dataset = self.test_datasets.get("test")
                else:
                    # 直接是数据集对象
                    test_dataset = self.test_datasets
                
                if test_dataset is not None:
                    eval_acc(model_name=model_name, retain_dataset=test_dataset, output_dir=root, batch_size=8)
                    torch.cuda.empty_cache()
            
            # 评估downstream数据集
            if self.downstream_datasets and len(self.downstream_datasets) > 0:
                print("开始评估downstream数据集...")
                for downstream_key, downstream_dataset in self.downstream_datasets.items():
                    if downstream_dataset is not None:
                        # 提取数据集名称（去掉"downstream_"前缀）
                        dataset_name = downstream_key.replace("downstream_", "")
                        print(f"评估downstream数据集: {dataset_name}")
                        
                        # 确保downstream数据集的输出目录存在
                        downstream_output_dir = f"{root}/downstream_{dataset_name}"
                        os.makedirs(downstream_output_dir, exist_ok=True)
                        
                        eval_acc(model_name=model_name, retain_dataset=downstream_dataset, output_dir=downstream_output_dir, batch_size=8)
                        torch.cuda.empty_cache()
            # 执行eval_ppl
            #eval_ppl(model_name=model_name, output_path=f"{root}/ppl.json")
            # 添加准确率评估
            eval_few_shots(model_name=model_name, output_path=f"{root}/few_shots.json")
            torch.cuda.empty_cache()

    def eval_accuracy(self, model_name, output_dir=".", batch_size=8):
        """
        评估模型在retain数据集上的准确率
        
        Args:
            model_name: 模型名称
            output_dir: 输出目录
            batch_size: 批次大小
        
        Returns:
            accuracies: 准确率字典，键为数据集名称，值为准确率（百分比）
        """
        accuracies = {}
        
        # 处理多个test数据集的情况
        if isinstance(self.test_datasets, dict):
            # 多个test数据集
            for test_key, test_dataset in self.test_datasets.items():
                if test_dataset is not None:
                    print(f"评估数据集: {test_key}")
                    accuracy = eval_acc(
                        model_name=model_name,
                        retain_dataset=test_dataset,
                        output_dir=f"{output_dir}/{test_key}",
                        batch_size=batch_size
                    )
                    accuracies[test_key] = accuracy
        else:
            # 单个test数据集（向后兼容）
            accuracy = eval_acc(
                model_name=model_name,
                retain_dataset=self.test_datasets,
                output_dir=output_dir,
                batch_size=batch_size
            )
            accuracies["test"] = accuracy
        
        return accuracies

    def save(self, logger):
        logger.save_ckpt("model", self.model, self.use_lora)
        logger.save_ckpt("tokenizer", self.tokenizer, self.use_lora)

    def run(self, logger):
        if self.resume_path is None:
            self.init_model()
            self.init_optimizer()
            self.init_dataset()
            self.init_mask(logger)
            self.init_unlearner(logger)
            
            # 运行冲突学习训练
            self._run_conflict_training(logger)
                    
            self.save(logger)
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            self.eval(logger)
        else:
            self.init_model()
            self.init_dataset()
            self.eval(logger)
    
    def _run_conflict_training(self, logger):
        """运行冲突学习训练"""
        print(f"开始冲突学习训练，交替频率：{self.alternate_frequency} epochs")
        print(f"Safe unlearn method: {self.safe_unlearn_method}")
        print(f"Conflict unlearn method: {self.conflict_unlearn_method}")
        
        # 计算每个epoch的步数
        steps_per_epoch = len(self.unlearn_dataset) // (
            self.batch_size * self.gradient_accumulation_steps * self.num_devices
        )
        
        # 创建自定义的训练循环
        self._custom_conflict_training_loop(steps_per_epoch, logger)
    
    def _custom_conflict_training_loop(self, steps_per_epoch, logger):
        """自定义冲突学习训练循环"""
        self.model.train()
        
        # 确保优化器已初始化
        if self.optimizer is None:
            self.init_optimizer()
        
        # 计算总步数
        total_steps = self.num_epochs * steps_per_epoch
        
        # 创建数据加载器
        train_dataloader = torch.utils.data.DataLoader(
            self.unlearn_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.unlearn_collator,
        )
        
        current_step = 0
        
        print(f"开始训练，总epoch数：{self.num_epochs}，每epoch步数：{steps_per_epoch}")
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # 决定当前epoch使用哪个unlearner
            if epoch==0:
                current_unlearner = self.safe_unlearner
                current_method = self.safe_unlearn_method
                print(f"使用 Safe unlearner: {current_method}")
            else:
                current_unlearner = self.conflict_unlearner
                current_method = self.conflict_unlearn_method
                print(f"使用 Conflict unlearner: {current_method}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                if current_step >= total_steps:
                    break
                
                # 将batch移动到设备
                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # 计算损失
                with torch.cuda.amp.autocast():
                    loss = current_unlearner.compute_loss(current_unlearner.model, batch)
                
                # 检查损失是否为NaN或无穷大
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：Batch {batch_idx} 损失为 {loss.item()}，跳过此批次")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪 - 防止梯度爆炸
                if self.optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                # 梯度累积
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.optimizer is not None:
                        # 检查梯度是否为NaN
                        has_nan_grad = False
                        for param in self.model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_grad = True
                                break
                        
                        if has_nan_grad:
                            print(f"警告：Batch {batch_idx} 梯度包含NaN，跳过此步骤")
                            self.optimizer.zero_grad()
                        else:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    current_step += 1
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 5 == 0:
                self.save(logger)
        
        print("冲突学习训练完成")


def get(**kwargs):
    return Unlearn(**kwargs)
