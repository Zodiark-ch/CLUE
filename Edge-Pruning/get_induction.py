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
import huggingface_hub
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

def get_validation_data(num_examples=None, seq_len=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]


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
        
        # Return sample with all fields
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
    
    print(f"Dataset processing completed with {len(processed_dataset)} samples")
    print(f"Filtered samples: {filtered_count} (token length exceeds {max_tokens})")
    print(f"Retention rate: {len(processed_dataset)/len(dataset)*100:.2f}%")
    print(f"New dataset features: {processed_dataset.features}")
    
    return processed_dataset

def validate_samples_with_llm(dataset: Any, num_samples: int = None) -> Any:
    """
    Use LLM to verify if dataset sample inputs are correct and calculate accuracy
    
    Args:
        dataset: Dataset to verify
        num_samples: Number of samples to verify, if None verify all samples
    """
    try:
        print(f"Loading LLM model for verification...")
        
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
        
        # Determine number of samples to verify
        total_samples = len(dataset)
        if num_samples is None:
            num_samples = total_samples
        else:
            num_samples = min(num_samples, total_samples)
        
        print(f"Starting verification of {num_samples} samples...")
        
        # Statistics variables
        sentence_correct = 0
        corr_sentence_correct = 0
        total_validated = 0
        
        # Token count statistics variables
        max_sentence_tokens = 0
        max_corr_sentence_tokens = 0
        sentence_token_counts = []
        corr_sentence_token_counts = []
        
        # Save correctly predicted samples
        correct_samples = []
        
        # Verify samples
        for i in range(num_samples):
            sample = dataset[i]
            
            # Get sentence and corr_sentence
            sentence = sample['sentence']
            corr_sentence = sample['corr_sentence']
            
            print(f"\n{'='*60}")
            print(f"Verifying sample {i+1}/{num_samples}")
            print(f"{'='*60}")
            
            # Verify sentence
            print(f"\nOriginal sentence:")
            print(f"Complete text: {sentence}")
            
            # Count tokens in sentence
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
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=5000)
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
            
            # Count tokens in corr_sentence
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
                print(f"  🎯 This sample's sentence and corr_sentence are both predicted correctly, saved")
        
        # Calculate and display final statistics
        sentence_accuracy = (sentence_correct / total_validated) * 100 if total_validated > 0 else 0
        corr_sentence_accuracy = (corr_sentence_correct / total_validated) * 100 if total_validated > 0 else 0
        
        # Calculate token statistics
        avg_sentence_tokens = sum(sentence_token_counts) / len(sentence_token_counts) if sentence_token_counts else 0
        avg_corr_sentence_tokens = sum(corr_sentence_token_counts) / len(corr_sentence_token_counts) if corr_sentence_token_counts else 0
        
        print(f"\n{'='*60}")
        print(f"LLM validation completed, validated {total_validated} samples in total")
        print(f"{'='*60}")
        print(f"📊 Final statistical results:")
        print(f"  sentence accuracy: {sentence_correct}/{total_validated} = {sentence_accuracy:.2f}%")
        print(f"  corr_sentence accuracy: {corr_sentence_correct}/{total_validated} = {corr_sentence_accuracy:.2f}%")
        print(f"  sentence correct predictions: {sentence_correct}")
        print(f"  corr_sentence correct predictions: {corr_sentence_correct}")
        print(f"  corr_sentence incorrect predictions: {total_validated - corr_sentence_correct}")
        print(f"\n🔢 Token count statistics:")
        print(f"  sentence max token count: {max_sentence_tokens}")
        print(f"  corr_sentence max token count: {max_corr_sentence_tokens}")
        print(f"  sentence average token count: {avg_sentence_tokens:.2f}")
        print(f"  corr_sentence average token count: {avg_corr_sentence_tokens:.2f}")
        print(f"  sentence token count range: {min(sentence_token_counts)} - {max_sentence_tokens}")
        print(f"  corr_sentence token count range: {min(corr_sentence_token_counts)} - {max_corr_sentence_tokens}")
        print(f"\n📁 Correct prediction sample statistics:")
        print(f"  Number of samples with both sentence and corr_sentence correct: {len(correct_samples)}")
        print(f"  {len(correct_samples)}/{total_validated} = {(len(correct_samples)/total_validated)*100:.2f}%")
        print(f"{'='*60}")
        
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
        print("Loading induction dataset")
        print("="*50)
        induction_dataset=get_validation_data(num_examples=3000)
        print(f"Induction dataset shape: {induction_dataset.shape}")
        
        from transformers import GPT2TokenizerFast
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('ArthurConmy/redwood_tokenizer')
        
        # Convert tensor to text
        print("\n" + "="*50)
        print("Converting tensor to text")
        print("="*50)
        
        def tensor_to_text(tensor_data, tokenizer):
            """Convert tensor data to text list"""
            text_list = []
            
            for i in range(tensor_data.shape[0]):
                # Get token sequence for the i-th sample
                token_sequence = tensor_data[i].tolist()
                
                # Use tokenizer to decode
                try:
                    text = tokenizer.decode(token_sequence, skip_special_tokens=True)
                    text_list.append(text)
                except Exception as e:
                    print(f"Error decoding sample {i}: {e}")
                    text_list.append("")  # Use empty string if decoding fails
                
                # Show progress every 100 samples
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{tensor_data.shape[0]} samples")
            
            return text_list
        
        # Convert tensor to text
        induction_texts = tensor_to_text(induction_dataset, gpt2_tokenizer)
        
        print(f"Conversion completed, generated {len(induction_texts)} text samples")
        
        # Display conversion results of first few samples
        print(f"\nConversion results of first 5 samples:")
        for i in range(min(5, len(induction_texts))):
            print(f"Sample {i+1}:")
            print(f"  Original tensor length: {len(induction_dataset[i])}")
            print(f"  Converted text: {repr(induction_texts[i])}")
            print(f"  Text length: {len(induction_texts[i])}")
            print()
        
        # Create dataset containing text
        from datasets import Dataset
        induction_text_dataset = Dataset.from_dict({
            'text': induction_texts,
            'original_tensor_length': [len(tensor) for tensor in induction_dataset]
        })
        
        print(f"Text dataset creation completed, contains {len(induction_text_dataset)} samples")
        print(f"Dataset features: {induction_text_dataset.features}")
        
        # Process induction text data
        print("\n" + "="*50)
        print("Processing induction text data")
        print("="*50)
        
                
            
        
        # Process all text
        processed_samples = []
        filtered_count = 0
        
        for i, text in enumerate(induction_texts):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{len(induction_texts)}")
            
            processed_sample = text
            if processed_sample is not None:
                processed_samples.append(processed_sample)
            else:
                filtered_count += 1
        
        print(f"Text processing completed, retained {len(processed_samples)} samples, filtered {filtered_count} samples")
        
        # Create processed dataset
        processed_induction_dataset = Dataset.from_list(processed_samples)
        print(f"Processed dataset features: {processed_induction_dataset.features}")
        
        # Display first few processed samples
        print(f"\nFirst 3 processed samples:")
        for i in range(min(3, len(processed_samples))):
            print(f"Sample {i+1}:")
            print(f"  sentence: {processed_samples[i]['sentence']}")
            print(f"  corr_sentence: {processed_samples[i]['corr_sentence']}")
            print(f"  object: {processed_samples[i]['object']}")
            print()
        
        # Token length detection and filtering
        print("\n" + "="*50)
        print("Token length detection and filtering")
        print("="*50)
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        print("Tokenizer loaded successfully")
        
        def check_token_length(sample, max_tokens=5000):
            """Check token length of samples"""
            sentence_tokens = tokenizer.encode(sample['sentence'], add_special_tokens=False)
            corr_sentence_tokens = tokenizer.encode(sample['corr_sentence'], add_special_tokens=False)
            
            sentence_token_count = len(sentence_tokens)
            corr_sentence_token_count = len(corr_sentence_tokens)
            
            return sentence_token_count <= max_tokens and corr_sentence_token_count <= max_tokens
        
        token_filtered_samples = []
        token_filtered_count = 0
        
        for i, sample in enumerate(processed_samples):
            if check_token_length(sample, max_tokens=5000):
                token_filtered_samples.append(sample)
            else:
                token_filtered_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Checked {i + 1}/{len(processed_samples)} samples")
        
        print(f"Token length filtering completed, retained {len(token_filtered_samples)} samples, filtered {token_filtered_count} samples")
        
        # Create final dataset
        final_induction_dataset = Dataset.from_list(token_filtered_samples)
        print(f"Final dataset size: {len(final_induction_dataset)} samples")
        
        # Split dataset: 1000 for training, 600 for validation
        total_final_samples = len(final_induction_dataset)
        train_size = min(1000, total_final_samples)
        validation_size = min(600, total_final_samples - train_size)
        
        print(f"\nDataset split:")
        print(f"  Total samples: {total_final_samples}")
        print(f"  Training set: {train_size}")
        print(f"  Validation set: {validation_size}")
        
        # Create training and validation sets
        train_induction_dataset = final_induction_dataset.select(range(train_size))
        validation_induction_dataset = final_induction_dataset.select(range(train_size, train_size + validation_size))
        
        # Save dataset
        print("\n" + "="*50)
        print("Saving induction dataset")
        print("="*50)
        
        # Create main dataset folder
        import os
        main_dataset_path = "./Edge-Pruning/data/datasets/induction"
        os.makedirs(main_dataset_path, exist_ok=True)
        
        # Save training set
        print(f"Saving training set...")
        train_path = os.path.join(main_dataset_path, "train")
        train_induction_dataset.save_to_disk(train_path)
        print(f"Training set saved to: {train_path}")
        
        # Save validation set
        if validation_size > 0:
            print(f"Saving validation set...")
            validation_path = os.path.join(main_dataset_path, "validation")
            validation_induction_dataset.save_to_disk(validation_path)
            print(f"Validation set saved to: {validation_path}")
        else:
            print(f"Validation set is empty, skipping save")
        
        print(f"Induction dataset processing completed!")
        
        # Validation function: load induction dataset from local and count tokens
        print("\n" + "="*50)
        print("Validating induction dataset")
        print("="*50)
        
        def validate_induction_dataset():
            """Validate induction dataset and count tokens"""
            try:
                # Load training set
                train_path = "./Edge-Pruning/data/datasets/induction/train"
                if os.path.exists(train_path):
                    from datasets import load_from_disk
                    train_dataset = load_from_disk(train_path)
                    print(f"Training set loaded successfully! Contains {len(train_dataset)} samples")
                else:
                    print("Training set file does not exist")
                    return
                
                # Load validation set
                validation_path = "./Edge-Pruning/data/datasets/induction/validation"
                if os.path.exists(validation_path):
                    validation_dataset = load_from_disk(validation_path)
                    print(f"Validation set loaded successfully! Contains {len(validation_dataset)} samples")
                else:
                    print("Validation set file does not exist")
                    return
                
                # Load tokenizer
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
                print("Tokenizer loaded successfully")
                
                def analyze_dataset_tokens(dataset, dataset_name):
                    """Analyze token count of dataset"""
                    print(f"\nAnalyzing {dataset_name} dataset...")
                    
                    sentence_token_counts = []
                    corr_sentence_token_counts = []
                    
                    for i, sample in enumerate(dataset):
                        # Analyze sentence
                        sentence = sample['sentence']
                        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                        sentence_token_counts.append(len(sentence_tokens))
                        
                        # Analyze corr_sentence
                        corr_sentence = sample['corr_sentence']
                        corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
                        corr_sentence_token_counts.append(len(corr_sentence_tokens))
                        
                        # Show progress every 50 samples
                        if (i + 1) % 50 == 0:
                            print(f"  Processed {i + 1}/{len(dataset)} samples")
                    
                    # Calculate statistics
                    sentence_max = max(sentence_token_counts)
                    sentence_min = min(sentence_token_counts)
                    sentence_mean = sum(sentence_token_counts) / len(sentence_token_counts)
                    
                    corr_sentence_max = max(corr_sentence_token_counts)
                    corr_sentence_min = min(corr_sentence_token_counts)
                    corr_sentence_mean = sum(corr_sentence_token_counts) / len(corr_sentence_token_counts)
                    
                    # Display statistical results
                    print(f"\n{dataset_name} dataset token statistics:")
                    print(f"  sentence:")
                    print(f"    Maximum: {sentence_max}")
                    print(f"    Minimum: {sentence_min}")
                    print(f"    Mean: {sentence_mean:.2f}")
                    
                    print(f"  corr_sentence:")
                    print(f"    Maximum: {corr_sentence_max}")
                    print(f"    Minimum: {corr_sentence_min}")
                    print(f"    Mean: {corr_sentence_mean:.2f}")
                    
                    return {
                        'sentence_max': sentence_max,
                        'sentence_min': sentence_min,
                        'sentence_mean': sentence_mean,
                        'corr_sentence_max': corr_sentence_max,
                        'corr_sentence_min': corr_sentence_min,
                        'corr_sentence_mean': corr_sentence_mean
                    }
                
                # Analyze training set
                train_stats = analyze_dataset_tokens(train_dataset, "Training set")
                
                # Analyze validation set
                validation_stats = analyze_dataset_tokens(validation_dataset, "Validation set")
                
                # Comprehensive statistics
                print(f"\n{'='*60}")
                print("Comprehensive statistical results:")
                print(f"{'='*60}")
                print(f"Training set + Validation set token statistics:")
                print(f"  sentence:")
                print(f"    Maximum: {max(train_stats['sentence_max'], validation_stats['sentence_max'])}")
                print(f"    Minimum: {min(train_stats['sentence_min'], validation_stats['sentence_min'])}")
                print(f"    Mean: {(train_stats['sentence_mean'] + validation_stats['sentence_mean']) / 2:.2f}")
                
                print(f"  corr_sentence:")
                print(f"    Maximum: {max(train_stats['corr_sentence_max'], validation_stats['corr_sentence_max'])}")
                print(f"    Minimum: {min(train_stats['corr_sentence_min'], validation_stats['corr_sentence_min'])}")
                print(f"    Mean: {(train_stats['corr_sentence_mean'] + validation_stats['corr_sentence_mean']) / 2:.2f}")
                
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"Error occurred during validation: {e}")
                import traceback
                traceback.print_exc()
        
        # Execute validation
        validate_induction_dataset()
        
        # Return processed dataset for further processing
        return final_induction_dataset
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        return None

if __name__ == "__main__":
    dataset = main()
    if dataset is not None:
        print(f"\nDataset loaded successfully! Contains {len(dataset)} samples")
    else:
        print("Dataset loading failed!")
