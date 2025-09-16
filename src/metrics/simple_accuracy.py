import json
import torch
import tqdm
from torch.utils.data import DataLoader


def extract_retain_from_unlearn_dataset(unlearn_dataset, num_samples=None):
    """
    Extract retain data from UnlearnDataset
    
    Args:
        unlearn_dataset: UnlearnDataset instance
        num_samples: Number of samples to extract, if None extract all
    
    Returns:
        retain_data: List containing retain data
    """
    retain_data = []
    
    if num_samples is None:
        num_samples = len(unlearn_dataset)
    
    for i in range(min(num_samples, len(unlearn_dataset))):
        item = unlearn_dataset[i]
        if item['retain'] is not None:
            retain_data.append(item['retain'])
    
    return retain_data


def eval_acc(
    model_name,
    retain_dataset,
    output_dir=".",
    batch_size=8,
    device="cuda"
):
    """
    Evaluate model accuracy on retain dataset (simplified version)
    
    Args:
        model_name: Model name
        retain_dataset: Test dataset (could be UnlearnDataset or regular dataset)
        output_dir: Output directory
        batch_size: Batch size
        device: Device
    
    Returns:
        accuracy: Accuracy (percentage)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
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
    
    # Check dataset type and extract retain data
    if hasattr(retain_dataset, 'retain_dataset') and retain_dataset.retain_dataset is not None:
        # If it's UnlearnDataset, directly use its retain_dataset
        print("Detected UnlearnDataset, using its retain_dataset for evaluation")
        actual_dataset = retain_dataset.retain_dataset
    else:
        # If it's a regular dataset, use it directly
        print("Using regular dataset for evaluation")
        actual_dataset = retain_dataset
    
    # Create data loader
    def collate_fn(batch):
        # Process standard format data
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch])
        }
    
    dataloader = DataLoader(
        actual_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
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
            
            # Get predicted tokens for each position
            predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
            
            # Calculate accuracy (ignore padding and -100 labels)
            # Ensure all tensors have consistent size
            valid_mask = (labels != -100) & (attention_mask == 1)
            
            # Only compare the prediction of the last valid token for each sample
            batch_correct = 0
            batch_total = 0
            
            for i in range(len(labels)):
                # Find the position of the last valid token for this sample
                valid_positions = valid_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1]
                    # Check if the prediction of the last valid token is correct
                    if predicted_tokens[i, last_valid_pos] == labels[i, last_valid_pos]:
                        batch_correct += 1
                    batch_total += 1
            
            correct_predictions += batch_correct
            total_predictions += batch_total
    
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
    
    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/accuracy.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return accuracy


# Method used in unlearn model
def eval_acc_in_unlearn(self, model_name, output_dir=".", batch_size=8):
    """
    Accuracy evaluation method used in unlearn model
    
    Args:
        self: unlearn model instance
        model_name: Model name
        output_dir: Output directory
        batch_size: Batch size
    
    Returns:
        accuracy: Accuracy (percentage)
    """
    return eval_acc(
        model_name=model_name,
        retain_dataset=self.test_dataset,
        output_dir=output_dir,
        batch_size=batch_size
    ) 