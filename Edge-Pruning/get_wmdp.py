#!/usr/bin/env python
# coding=utf-8
"""
WMDP dataset loading and processing script
Used to reprocess WMDP dataset, especially the cyber subset
"""

import os
import sys
import torch
from datasets import load_dataset, load_from_disk
from typing import Optional, Dict, Any

def load_wmdp_dataset(
    dataset_name: str = "cais/wmdp",
    subset: str = "wmdp-bio",
    split: str = "test",
    cache_dir: str = "./.cache"
) -> Any:
    """
    Load WMDP dataset
    
    Args:
        dataset_name: Dataset name, default is "cais/wmdp"
        subset: Dataset subset, default is "wmdp-cyber"
        split: Dataset split, default is "test"
        cache_dir: Cache directory, default is "./.cache"
    
    Returns:
        Loaded dataset
    """
    try:
        print(f"Loading dataset: {dataset_name}, subset: {subset}, split: {split}")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load dataset
        dataset = load_dataset(
            dataset_name, 
            subset, 
            cache_dir=cache_dir
        )[split]
        
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        print(f"Dataset features: {dataset.features}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def save_dataset_to_disk(
    dataset: Any,
    save_path: str,
    dataset_name: str = "wmdp-bio"
) -> None:
    """
    Save dataset to local disk for later loading with load_from_disk
    
    Args:
        dataset: Dataset to save
        save_path: Save path
        dataset_name: Dataset name for creating directory
    """
    try:
        # Create save directory
        full_save_path = os.path.join(save_path, dataset_name)
        os.makedirs(full_save_path, exist_ok=True)
        
        print(f"Saving dataset to: {full_save_path}")
        
        # Save dataset
        dataset.save_to_disk(full_save_path)
        
        print(f"Dataset successfully saved to: {full_save_path}")
        print(f"Saved dataset contains {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error saving dataset: {e}")
        raise

def load_dataset_from_disk(
    dataset_path: str,
    dataset_name: str = "wmdp-bio"
) -> Any:
    """
    Load dataset from local disk
    
    Args:
        dataset_path: Dataset root directory path
        dataset_name: Dataset name
    
    Returns:
        Loaded dataset
    """
    try:
        full_path = os.path.join(dataset_path, dataset_name)
        print(f"Loading dataset from local: {full_path}")
        
        # Check if path exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset path does not exist: {full_path}")
        
        # Load dataset from disk
        dataset = load_from_disk(full_path)
        
        print(f"Successfully loaded dataset from local with {len(dataset)} samples")
        print(f"Dataset features: {dataset.features}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset from local: {e}")
        raise

def process_wmdp_dataset(dataset: Any, bio_dataset: Any = None, max_tokens: int = 5000) -> Any:
    """
    Process WMDP dataset, add new metadata, and filter out samples exceeding token length limit
    
    Args:
        dataset: Original dataset
        bio_dataset: Bio dataset for generating choiceE
        max_tokens: Maximum token count limit, samples exceeding this will be filtered out
    
    Returns:
        Processed dataset
    """
    import random
    
    # Load tokenizer for checking token length
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    print(f"Using tokenizer to check token length, maximum limit: {max_tokens}")
    
    def process_sample(sample):
        # Get original data
        answer = sample['answer']
        question = sample['question']
        choices = sample['choices']
        
        # 1. choiceA is the correct answer, get from choices[answer]
        if 0 <= answer < len(choices):
            choiceA = choices[answer]
        else:
            choiceA = choices[0] if len(choices) > 0 else ""
        
        # 2. choiceB randomly selected from bio dataset
        if bio_dataset is not None and len(bio_dataset) > 0:
            random_bio_sample = random.choice(bio_dataset)
            random_bio_choices = random_bio_sample['choices']
            if len(random_bio_choices) > 0:
                choiceB = random.choice(random_bio_choices)
            else:
                choiceB = "No bio choice available"
        else:
            choiceB = "No bio dataset available"
        
        # 3. Build sentence - now only A and B options
        sentence = (
            "Please choose the answer that is most relevant to the question from options A and B. "
            f"The Question: {question} "
            f"The Options: "
            f"A. {choiceA} "
            f"B. {choiceB} "
            f"The answer is: A"
        )
        
        # 4. Build corr_sentence - choose B as wrong answer
        corr_sentence = (
            "Please choose the answer that is least relevant to the question from options A and B. "
            f"The Question: {question} "
            f"The Options: "
            f"A. {choiceA} "
            f"B. {choiceB} "
            f"The answer is: B"
        )
        
        # Check token length
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
        
        sentence_token_count = len(sentence_tokens)
        corr_sentence_token_count = len(corr_sentence_tokens)
        
        # If any text exceeds token limit, return None to filter out
        if sentence_token_count > max_tokens or corr_sentence_token_count > max_tokens:
            return None
        
        # Return sample containing all fields
        return {
            'answer': answer,
            'question': question,
            'choices': choices,
            'choiceA': choiceA,
            'choiceB': choiceB,
            'sentence': sentence,
            'corr_sentence': corr_sentence,
            'sentence_token_count': sentence_token_count,
            'corr_sentence_token_count': corr_sentence_token_count
        }
    
    # Process all samples and filter
    processed_samples = []
    filtered_count = 0
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:  # Print progress every 100 samples
            print(f"Processing sample {i+1}/{len(dataset)}")
        
        processed_sample = process_sample(sample)
        if processed_sample is not None:
            processed_samples.append(processed_sample)
        else:
            filtered_count += 1
    
    # Create new dataset
    from datasets import Dataset
    processed_dataset = Dataset.from_list(processed_samples)
    
    print(f"Dataset processing completed, contains {len(processed_dataset)} samples")
    print(f"Filtered out samples: {filtered_count} (token length exceeds {max_tokens})")
    print(f"Retention rate: {len(processed_dataset)/len(dataset)*100:.2f}%")
    print(f"New dataset features: {processed_dataset.features}")
    
    return processed_dataset

def validate_samples_with_llm(dataset: Any, num_samples: int = None) -> Any:
    """
    Use LLM to validate dataset samples and calculate accuracy
    
    Args:
        dataset: Dataset to validate
        num_samples: Number of samples to validate, if None then validate all samples
    """
    try:
        print(f"Loading LLM model for validation...")
        
        # Import model
        sys.path.append(os.path.join(os.getcwd(), "src/modeling/"))
        from transformers import MistralForCausalLM
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        model = MistralForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Move model to GPU (if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded to {device}")
        
        # Determine number of samples to validate
        total_samples = len(dataset)
        if num_samples is None:
            num_samples = total_samples
        else:
            num_samples = min(num_samples, total_samples)
        
        print(f"Starting validation of {num_samples} samples...")
        
        # Statistical variables
        sentence_correct = 0
        corr_sentence_correct = 0
        total_validated = 0
        
        # Token count statistical variables
        max_sentence_tokens = 0
        max_corr_sentence_tokens = 0
        sentence_token_counts = []
        corr_sentence_token_counts = []
        
        # Save correctly predicted samples
        correct_samples = []
        
        # Validate samples
        for i in range(num_samples):
            sample = dataset[i]
            
            # Get sentence and corr_sentence
            sentence = sample['sentence']
            corr_sentence = sample['corr_sentence']
            
            print(f"\n{'='*60}")
            print(f"Validating sample {i+1}/{num_samples}")
            print(f"{'='*60}")
            
            # Validate sentence
            print(f"\nOriginal sentence:")
            print(f"Complete text: {sentence}")
            
            # Count sentence tokens
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_token_count = len(sentence_tokens)
            sentence_token_counts.append(sentence_token_count)
            if sentence_token_count > max_sentence_tokens:
                max_sentence_tokens = sentence_token_count
            
            print(f"Token count: {sentence_token_count}")
            
            sentence_prediction_correct = False
            # Split input and label
            last_space_idx = sentence.rfind(' ')
            if last_space_idx != -1:
                input_text = sentence[:last_space_idx]
                target_label = sentence[last_space_idx:]  # Include space
                
                print(f"Input part: {input_text}")
                print(f"Target label: {target_label}")
                
                # Use LLM to generate prediction
                with torch.no_grad():
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate next token
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode generated token
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
                    
                    print(f"Model predicted next token: '{predicted_token}'")
                    print(f"Expected label: '{target_label}'")
                    
                    # Check if prediction is correct
                    if predicted_token.strip() == target_label.strip():
                        print("✅ Prediction correct!")
                        sentence_prediction_correct = True
                        sentence_correct += 1
                    else:
                        print("❌ Prediction incorrect!")
            else:
                print("Cannot find last space to split input and label")
            
            # Validate corr_sentence
            print(f"\nInterference sentence:")
            print(f"Complete text: {corr_sentence}")
            
            # Count corr_sentence tokens
            corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
            corr_sentence_token_count = len(corr_sentence_tokens)
            corr_sentence_token_counts.append(corr_sentence_token_count)
            if corr_sentence_token_count > max_corr_sentence_tokens:
                max_corr_sentence_tokens = corr_sentence_token_count
            
            print(f"Token count: {corr_sentence_token_count}")
            
            corr_sentence_prediction_correct = False
            # Split input and label
            last_space_idx = corr_sentence.rfind(' ')
            if last_space_idx != -1:
                input_text = corr_sentence[:last_space_idx]
                target_label = corr_sentence[last_space_idx:]  # Include space
                
                print(f"Input part: {input_text}")
                print(f"Target label: {target_label}")
                
                # Use LLM to generate prediction
                with torch.no_grad():
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate next token
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode generated token
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
                    
                    print(f"Model predicted next token: '{predicted_token}'")
                    print(f"Expected label: '{target_label}'")
                    
                    # Check if prediction is correct
                    if predicted_token.strip() == target_label.strip():
                        print("✅ Prediction correct!")
                        corr_sentence_prediction_correct = True
                        corr_sentence_correct += 1
                    else:
                        print("❌ Prediction incorrect!")
            else:
                print("Cannot find last space to split input and label")
            
            # Update total validation count
            total_validated += 1
            
            # Display current sample validation results
            print(f"\nSample {i+1} validation results:")
            print(f"  sentence prediction: {'✅ Correct' if sentence_prediction_correct else '❌ Incorrect'}")
            print(f"  corr_sentence prediction: {'✅ Correct' if corr_sentence_prediction_correct else '❌ Incorrect'}")
            
            # If both sentence and corr_sentence are predicted correctly, save this sample
            if sentence_prediction_correct and corr_sentence_prediction_correct:
                correct_samples.append(sample)
                print(f"  🎯 This sample has both sentence and corr_sentence predicted correctly, saved")
        
        # Calculate and display final statistical results
        sentence_accuracy = (sentence_correct / total_validated) * 100 if total_validated > 0 else 0
        corr_sentence_accuracy = (corr_sentence_correct / total_validated) * 100 if total_validated > 0 else 0
        
        # Calculate token statistics
        avg_sentence_tokens = sum(sentence_token_counts) / len(sentence_token_counts) if sentence_token_counts else 0
        avg_corr_sentence_tokens = sum(corr_sentence_token_counts) / len(corr_sentence_token_counts) if corr_sentence_token_counts else 0
        
        
        
        # Save correctly predicted samples as new dataset
        if correct_samples:
            try:
                from datasets import Dataset
                correct_dataset = Dataset.from_list(correct_samples)
                
                # Save to local
                save_path = "./data/datasets/wmdpcyber"
                correct_dataset.save_to_disk(save_path)
                
                print(f"\n💾 Correct prediction sample dataset saved to: {save_path}")
                print(f"   Dataset size: {len(correct_dataset)} samples")
                print(f"   Dataset features: {correct_dataset.features}")
                
                return correct_dataset
            except Exception as e:
                print(f"Error occurred while saving correct prediction sample dataset: {e}")
                return None
        else:
            print(f"\n⚠️  No samples found where both sentence and corr_sentence are predicted correctly")
            return None
        
    except Exception as e:
        print(f"Error occurred during LLM validation: {e}")
        import traceback
        traceback.print_exc()

def explore_dataset(dataset: Any, num_samples: int = 5) -> None:
    """
    Explore dataset structure and content
    
    Args:
        dataset: Loaded dataset
        num_samples: Number of samples to display
    """
    print("=" * 50)
    print("Dataset exploration")
    print("=" * 50)
    
    # Display basic dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    
    # Display first few samples
    print(f"\nFirst {num_samples} samples:")
    for i, sample in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        print(f"\nSample {i+1}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")

def main():
    """
    Main function
    """
    try:
        # Load WMDP cyber dataset
        print("="*50)
        print("Loading WMDP cyber dataset")
        print("="*50)
        dataset = load_wmdp_dataset()
        
        # Load WMDP bio dataset for generating choiceE
        print("\n" + "="*50)
        print("Loading WMDP bio dataset")
        print("="*50)
        bio_dataset = load_dataset("cais/wmdp", "wmdp-cyber", cache_dir="./.cache")["test"]
        print(f"Bio dataset contains {len(bio_dataset)} samples")
        
        # Explore original dataset
        print("\n" + "="*50)
        print("Original dataset exploration")
        print("="*50)
        explore_dataset(dataset)
        
        # Process dataset, add new metadata
        print("\n" + "="*50)
        print("Processing dataset, adding new metadata")
        print("="*50)
        processed_dataset = process_wmdp_dataset(dataset, bio_dataset, max_tokens=5000)
        
        # Explore processed dataset
        print("\n" + "="*50)
        print("Processed dataset exploration")
        print("="*50)
        explore_dataset(processed_dataset)
        

        print("\n" + "="*50)
        print("Saving processed dataset")
        print("="*50)
        
        # Split dataset: 1000 samples for training, rest for validation
        total_samples = len(processed_dataset)
        train_size = min(1000, total_samples)
        validation_size = total_samples - train_size
        
        print(f"Total dataset samples: {total_samples}")
        print(f"Training set samples: {train_size}")
        print(f"Validation set samples: {validation_size}")
        
        # Create training and validation sets
        train_dataset = processed_dataset.select(range(train_size))
        validation_dataset = processed_dataset.select(range(train_size, total_samples))
        
        # Create main dataset folder
        import os
        main_dataset_path = "./Edge-Pruning/data/datasets/wmdpbio"
        os.makedirs(main_dataset_path, exist_ok=True)
        
        # Save training set to train subfolder
        print(f"\nSaving training set...")
        train_path = os.path.join(main_dataset_path, "train")
        train_dataset.save_to_disk(train_path)
        print(f"Training set saved to: {train_path}")
        
        # Save validation set to validation subfolder
        if validation_size > 0:
            print(f"Saving validation set...")
            validation_path = os.path.join(main_dataset_path, "validation")
            validation_dataset.save_to_disk(validation_path)
            print(f"Validation set saved to: {validation_path}")
        else:
            print(f"Validation set is empty, skipping save")
        
        # Demonstrate loading processed dataset from local
        print("\n" + "="*50)
        print("Demonstrating loading processed dataset from local")
        print("="*50)
        
        # Load training set
        train_path = "./Edge-Pruning/data/datasets/wmdpbio/train"
        if os.path.exists(train_path):
            from datasets import load_from_disk
            local_train_dataset = load_from_disk(train_path)
            print(f"Training set loaded successfully! Contains {len(local_train_dataset)} samples")
        else:
            print("Training set file does not exist")
            local_train_dataset = None
        
        # Load validation set
        validation_path = "./Edge-Pruning/data/datasets/wmdpbio/validation"
        if os.path.exists(validation_path):
            local_validation_dataset = load_from_disk(validation_path)
            print(f"Validation set loaded successfully! Contains {len(local_validation_dataset)} samples")
        else:
            print("Validation set file does not exist")
            local_validation_dataset = None
        
        # Test token length
        print("\n" + "="*50)
        print("Testing dataset token length")
        print("="*50)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        print("Tokenizer loaded successfully")
        
        def analyze_token_lengths(dataset, dataset_name):
            """Analyze token length of dataset"""
            if dataset is None:
                print(f"{dataset_name} dataset does not exist, skipping analysis")
                return
            
            print(f"\nAnalyzing {dataset_name} dataset...")
            
            # Statistical variables
            sentence_token_lengths = []
            corr_sentence_token_lengths = []
            max_sentence_tokens = 0
            max_corr_sentence_tokens = 0
            max_sentence_sample = None
            max_corr_sentence_sample = None
            over_limit_count = 0
            
            # Analyze each sample
            for i, sample in enumerate(dataset):
                # Analyze sentence
                sentence = sample['sentence']
                sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                sentence_token_count = len(sentence_tokens)
                sentence_token_lengths.append(sentence_token_count)
                
                if sentence_token_count > max_sentence_tokens:
                    max_sentence_tokens = sentence_token_count
                    max_sentence_sample = sample
                
                # Analyze corr_sentence
                corr_sentence = sample['corr_sentence']
                corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
                corr_sentence_token_count = len(corr_sentence_tokens)
                corr_sentence_token_lengths.append(corr_sentence_token_count)
                
                if corr_sentence_token_count > max_corr_sentence_tokens:
                    max_corr_sentence_tokens = corr_sentence_token_count
                    max_corr_sentence_sample = sample
                
                # Check if there are samples exceeding 1000
                if sentence_token_count > 70 or corr_sentence_token_count > 70:
                    over_limit_count += 1
                    print(f"  Warning: Sample {i} exceeds 1000 tokens - sentence: {sentence_token_count}, corr_sentence: {corr_sentence_token_count}")
                
                # Show progress every 100 samples
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} samples")
            
            # Calculate statistics
            avg_sentence_tokens = sum(sentence_token_lengths) / len(sentence_token_lengths)
            avg_corr_sentence_tokens = sum(corr_sentence_token_lengths) / len(corr_sentence_token_lengths)
            min_sentence_tokens = min(sentence_token_lengths)
            min_corr_sentence_tokens = min(corr_sentence_token_lengths)
            
            
            
            if over_limit_count > 0:
                print(f"  ⚠️  Warning: Found {over_limit_count} samples exceeding 1000 tokens limit")
            else:
                print(f"  ✅ All samples are within 1000 tokens limit")
            
            # Display detailed information of longest samples
            if max_sentence_sample:
                print(f"\nLongest sentence sample (token count: {max_sentence_tokens}):")
                print(f"  sentence: {max_sentence_sample['sentence']}")
                print(f"  corr_sentence: {max_sentence_sample['corr_sentence']}")
            
            if max_corr_sentence_sample and max_corr_sentence_sample != max_sentence_sample:
                print(f"\nLongest corr_sentence sample (token count: {max_corr_sentence_tokens}):")
                print(f"  sentence: {max_corr_sentence_sample['sentence']}")
                print(f"  corr_sentence: {max_corr_sentence_sample['corr_sentence']}")
            
            return {
                'max_sentence_tokens': max_sentence_tokens,
                'max_corr_sentence_tokens': max_corr_sentence_tokens,
                'avg_sentence_tokens': avg_sentence_tokens,
                'avg_corr_sentence_tokens': avg_corr_sentence_tokens
            }
        
        # Analyze training set
        train_stats = analyze_token_lengths(local_train_dataset, "Training set")
        
        # Analyze validation set
        validation_stats = analyze_token_lengths(local_validation_dataset, "Validation set")
        
        
        
        # Return processed dataset for further processing
        return processed_dataset
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        return None

if __name__ == "__main__":
    dataset = main()
    if dataset is not None:
        print(f"\nDataset loaded successfully! Contains {len(dataset)} samples")
    else:
        print("Dataset loading failed!")
