import json
import torch
import tqdm
from torch.utils.data import DataLoader


def debug_dataset_format(retain_dataset, num_samples=5):
    """
    Debug dataset format, display structure of first few samples
    """
    print("=== Dataset Format Debug ===")
    print(f"Dataset type: {type(retain_dataset)}")
    print(f"Dataset length: {len(retain_dataset)}")
    
    # Check first few samples
    for i in range(min(num_samples, len(retain_dataset))):
        sample = retain_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Type: {type(sample)}")
        if isinstance(sample, dict):
            print(f"  Keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: tensor shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"    {key}: {type(value)} - {value}")
        else:
            print(f"  Value: {sample}")


def eval_acc_debug(
    model_name,
    retain_dataset,
    output_dir=".",
    batch_size=8,
    device="cuda"
):
    """
    Debug version of accuracy evaluation function
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # First debug dataset format
    debug_dataset_format(retain_dataset)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Set pad token
    try:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Create data loader
    def collate_fn(batch):
        print(f"Collate function called, batch size: {len(batch)}")
        print(f"First sample type: {type(batch[0])}")
        
        if isinstance(batch[0], dict):
            print(f"First sample keys: {list(batch[0].keys())}")
            
            # Check if all samples have consistent keys
            all_keys = set()
            for item in batch:
                all_keys.update(item.keys())
            print(f"All sample keys: {all_keys}")
            
            # Try to extract data
            try:
                if 'input_ids' in batch[0]:
                    input_ids = torch.stack([item['input_ids'] for item in batch])
                    attention_mask = torch.stack([item.get('attention_mask', torch.ones_like(item['input_ids'])) for item in batch])
                    labels = torch.stack([item.get('label', item.get('labels', item['input_ids'])) for item in batch])
                    
                    print(f"Successfully extracted data - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
                    
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'label': labels
                    }
                else:
                    print("Error: input_ids key not found")
                    return None
            except Exception as e:
                print(f"Error extracting data: {e}")
                return None
        else:
            print(f"Sample is not in dictionary format: {type(batch[0])}")
            return None
    
    dataloader = DataLoader(
        retain_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    print("Starting accuracy evaluation...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluation progress")):
            if batch is None:
                print(f"Skipping batch {i} (data format issue)")
                continue
                
            print(f"Processing batch {i}: {type(batch)}")
            
            try:
                # Move data to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                print(f"Batch {i} data shape: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, labels={labels.shape}")
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get prediction results
                logits = outputs.logits
                
                # Get predicted tokens for each position
                predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len-1]
                
                # Calculate accuracy (ignore padding and -100 labels)
                valid_mask = (labels != -100) & (attention_mask[:, :-1] == 1)
                
                # Compare predictions with true labels
                correct = (predicted_tokens == labels) & valid_mask
                correct_predictions += correct.sum().item()
                total_predictions += valid_mask.sum().item()
                
                print(f"Batch {i} results: correct={correct.sum().item()}, total={valid_mask.sum().item()}")
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {total_predictions}")
    
    # Save results
    result = {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "model_name": model_name
    }
    
    with open(f"{output_dir}/accuracy_debug.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return accuracy 