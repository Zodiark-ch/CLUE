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


class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Conflict learning parameters
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
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)  # Add gradient clipping parameter
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
        """Move mask to correct device"""
        if mask is None:
            return
        
        # Directly use wanda-type processing (using numeric indices as keys)
        print(f"Processing {mask_name} mask, using numeric indices as keys")
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
            if self.if_llama:# If llama series use pad, otherwise use eos (depends on specific model)
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))# Because pad was added, need to ensure embedding size matches tokenizer size
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
        
        # Conflict learning mode: create two unlearners
        self._init_conflict_unlearners(logger)
    
    def _init_conflict_unlearners(self, logger):
        """Initialize two unlearners for conflict learning"""
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
        
        # Create safe unlearner
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
        
        # Create conflict unlearner
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
        # Initialize two masks
        self.safe_mask = None
        self.conflict_mask = None
        
        # Load safe mask /data/zodiark/CSAT/conflict/safe_mask.pt
        if self.safe_mask_path is not None and os.path.exists(self.safe_mask_path):
            print(f"Loading safe mask: {self.safe_mask_path}")
            self.safe_mask = torch.load(self.safe_mask_path)
            self._move_mask_to_device(self.safe_mask, None, "safe")
        
        # Load conflict mask
        if self.conflict_mask_path is not None and os.path.exists(self.conflict_mask_path):
            print(f"Loading conflict mask: {self.conflict_mask_path}")
            self.conflict_mask = torch.load(self.conflict_mask_path)
            self._move_mask_to_device(self.conflict_mask, None, "conflict")
        
        # Set default mask to safe_mask
        if self.safe_mask is not None:
            self.mask = self.safe_mask
        else:
            self.mask = None
            
        # If safe mask file does not exist, need to generate
        if self.safe_mask_path is not None and not os.path.exists(self.safe_mask_path):
            self._generate_mask(self.safe_mask_path, logger, "safe")
        
        # If conflict mask file does not exist, need to generate
        if self.conflict_mask_path is not None and not os.path.exists(self.conflict_mask_path):
            self._generate_mask(self.conflict_mask_path, logger, "conflict")
    
    def _generate_mask(self, mask_path, logger, mask_name):
        """Generate mask file"""
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
        
        # Save generated mask
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
            # Create standard AdamW optimizer
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
        
        # Ensure root directory exists
        import os
        os.makedirs(root, exist_ok=True)
        
        if self.resume_path is not None:
            model_name = self.resume_path
        else:
            # Use checkpoints path, this is where fine-tuned models are saved
            model_name = os.path.join(root, "checkpoints")
        if self.task_name == "downstream":
            if "WMDP" in self.dataset_names["forget"]:
                print("Starting evaluation of wmdp dataset...")
                eval_few_shots(model_name=model_name, task_list=["wmdp"], output_path=f"{root}/wmdp.json")
                torch.cuda.empty_cache()
                eval_few_shots(model_name=model_name,  task_list=["mmlu"],output_path=f"{root}/mmlu.json")
            torch.cuda.empty_cache()
            if self.dataset_names["forget"] == "SafePku":
                print("Starting evaluation of DETOX dataset...")
                eval_toxic(
                    model_name=model_name, output_dir=root, dataset=self.unlearn_dataset
                )
            torch.cuda.empty_cache()
            # Handle multiple test datasets
            if isinstance(self.test_datasets, dict) and len(self.test_datasets) > 1:
                # Multiple test datasets
                data_num=0
                for test_key, test_dataset in self.test_datasets.items():
                    if test_dataset is not None:
                        print(f"Evaluating dataset: {self.dataset_names['retain'][data_num]}")
                        data_num+=1
                        # Ensure output directory exists for each test dataset
                        test_output_dir = f"{root}/{test_key}"
                        os.makedirs(test_output_dir, exist_ok=True)
                        eval_acc(model_name=model_name, retain_dataset=test_dataset, output_dir=test_output_dir, batch_size=8)
                        torch.cuda.empty_cache()
            else:
                # Single test dataset (backward compatibility)
                if isinstance(self.test_datasets, dict):
                    # Single test dataset, key is "test"
                    test_dataset = self.test_datasets.get("test")
                else:
                    # Directly a dataset object
                    test_dataset = self.test_datasets
                
                if test_dataset is not None:
                    eval_acc(model_name=model_name, retain_dataset=test_dataset, output_dir=root, batch_size=8)
                    torch.cuda.empty_cache()
            
            # Evaluate downstream datasets
            if self.downstream_datasets and len(self.downstream_datasets) > 0:
                print("Starting evaluation of downstream datasets...")
                for downstream_key, downstream_dataset in self.downstream_datasets.items():
                    if downstream_dataset is not None:
                        # Extract dataset name (remove "downstream_" prefix)
                        dataset_name = downstream_key.replace("downstream_", "")
                        print(f"Evaluating downstream dataset: {dataset_name}")
                        
                        # Ensure output directory exists for downstream dataset
                        downstream_output_dir = f"{root}/downstream_{dataset_name}"
                        os.makedirs(downstream_output_dir, exist_ok=True)
                        
                        eval_acc(model_name=model_name, retain_dataset=downstream_dataset, output_dir=downstream_output_dir, batch_size=8)
                        torch.cuda.empty_cache()
            # Execute eval_ppl
            #eval_ppl(model_name=model_name, output_path=f"{root}/ppl.json")
            # Add accuracy evaluation
            eval_few_shots(model_name=model_name, output_path=f"{root}/few_shots.json")
            torch.cuda.empty_cache()

    def eval_accuracy(self, model_name, output_dir=".", batch_size=8):
        """
        Evaluate model accuracy on retain dataset
        
        Args:
            model_name: Model name
            output_dir: Output directory
            batch_size: Batch size
        
        Returns:
            accuracies: Accuracy dictionary, key is dataset name, value is accuracy (percentage)
        """
        accuracies = {}
        
        # Handle multiple test datasets
        if isinstance(self.test_datasets, dict):
            # Multiple test datasets
            for test_key, test_dataset in self.test_datasets.items():
                if test_dataset is not None:
                    print(f"Evaluating dataset: {test_key}")
                    accuracy = eval_acc(
                        model_name=model_name,
                        retain_dataset=test_dataset,
                        output_dir=f"{output_dir}/{test_key}",
                        batch_size=batch_size
                    )
                    accuracies[test_key] = accuracy
        else:
            # Single test dataset (backward compatibility)
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
            
            # Run conflict learning training
            self._run_conflict_training(logger)
                    
            self.save(logger)
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            self.eval(logger)
        else:
            self.init_model()
            self.init_dataset()
            self.eval(logger)
    
    def _run_conflict_training(self, logger):
        """Run conflict learning training"""
        print(f"Starting conflict learning training, alternate frequency: {self.alternate_frequency} epochs")
        print(f"Safe unlearn method: {self.safe_unlearn_method}")
        print(f"Conflict unlearn method: {self.conflict_unlearn_method}")
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.unlearn_dataset) // (
            self.batch_size * self.gradient_accumulation_steps * self.num_devices
        )
        
        # Create custom training loop
        self._custom_conflict_training_loop(steps_per_epoch, logger)
    
    def _custom_conflict_training_loop(self, steps_per_epoch, logger):
        """Custom conflict learning training loop"""
        self.model.train()
        
        # Ensure optimizer is initialized
        if self.optimizer is None:
            self.init_optimizer()
        
        # Calculate total steps
        total_steps = self.num_epochs * steps_per_epoch
        
        # Create data loader
        train_dataloader = torch.utils.data.DataLoader(
            self.unlearn_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.unlearn_collator,
        )
        
        current_step = 0
        
        print(f"Starting training, total epochs: {self.num_epochs}, steps per epoch: {steps_per_epoch}")
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Decide which unlearner to use for current epoch
            if epoch==0:
                current_unlearner = self.safe_unlearner
                current_method = self.safe_unlearn_method
                print(f"Using Safe unlearner: {current_method}")
            else:
                current_unlearner = self.conflict_unlearner
                current_method = self.conflict_unlearn_method
                print(f"Using Conflict unlearner: {current_method}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                if current_step >= total_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # Calculate loss
                with torch.cuda.amp.autocast():
                    loss = current_unlearner.compute_loss(current_unlearner.model, batch)
                
                # Check if loss is NaN or infinite
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Batch {batch_idx} loss is {loss.item()}, skipping this batch")
                    continue
                
                # Backward propagation
                loss.backward()
                
                # Gradient clipping - prevent gradient explosion
                if self.optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.optimizer is not None:
                        # Check if gradients are NaN
                        has_nan_grad = False
                        for param in self.model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_grad = True
                                break
                        
                        if has_nan_grad:
                            print(f"Warning: Batch {batch_idx} gradients contain NaN, skipping this step")
                            self.optimizer.zero_grad()
                        else:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    current_step += 1
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1} completed, average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save(logger)
        
        print("Conflict learning training completed")


def get(**kwargs):
    return Unlearn(**kwargs)
