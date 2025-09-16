import json
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


def eval_acc(
    model_name,
    retain_dataset,
    output_dir=".",
    batch_size=8,
    device="cuda"
):
    """
    Evaluate model accuracy on retain dataset
    
    Args:
        model_name: Model name
        retain_dataset: Test dataset
        output_dir: Output directory
        batch_size: Batch size
        device: Device
    """
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
    dataloader = DataLoader(
        retain_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'label': torch.stack([item['label'] for item in x])
        }
    )
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    print("Starting accuracy evaluation...")
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluation progress"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get prediction results
            logits = outputs.logits
            
            # For language models, we compare next token predictions
            # Input: [seq_len-1], Labels: [seq_len-1] (starting from second token)
            # Predictions: [seq_len-1, vocab_size]
            
            # Get predicted tokens for each position
            predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len-1]
            
            # Calculate accuracy (ignore padding and -100 labels)
            valid_mask = (labels != -100) & (attention_mask[:, :-1] == 1)
            
            # Compare predictions with true labels
            correct = (predicted_tokens == labels) & valid_mask
            correct_predictions += correct.sum().item()
            total_predictions += valid_mask.sum().item()
    
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
    
    with open(f"{output_dir}/accuracy.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return accuracy


if __name__ == "__main__":
    # Test code
    from src.dataset import get_dataset
    
    # Create test dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset_names = {"forget": "SafePku", "retain": "IOI"}
    dataset_seed = 8888
    forget_ratio = 0.1
    self_retain = False
    
    unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
        dataset_names, tokenizer, dataset_seed, forget_ratio, self_retain
    )
    
    # Test evaluation
    accuracy = eval_acc(
        model_name="gpt2",
        retain_dataset=test_dataset,
        output_dir="./results",
        batch_size=4
    )
    
    print(f"Final accuracy: {accuracy:.2f}%") 