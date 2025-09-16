import os
import json
import argparse
import torch
import sys

def parse_args():
    """
    Parse command line arguments
    Returns:
        argparse.Namespace: Parsed argument object
    """
    parser = argparse.ArgumentParser(description="Mask generation tool")
    
    # mask path parameters
    parser.add_argument(
        "-mask", "--mask_path", 
        type=str, 
        default="/data/zodiark/CSAT/wanda/zephyr/with_0.2.pt",
        help="Path to the mask pt file"
    )
    
    # conflict node parameters
    parser.add_argument(
        "-conflict", "--conflict_node", 
        type=str, 
        default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/masks/conflict_node_list_cyber_winogrande_sst2_bool_ioi_gp.json",
        help="Path to the conflict node list JSON file"
    )
    
    # false node parameters
    parser.add_argument(
        "-false", "--false_node", 
        type=str, 
        default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/masks/false_node_list_cyber_winogrande_sst2_bool_ioi_gp.json",
        help="Path to the false node list JSON file"
    )
    
    # output parameters
    parser.add_argument(
        "-output", "--output", 
        type=str, 
        default="/data/zodiark/CSAT/conflict",
        help="Output directory path (if empty, uses mask path directory)"
    )
    
    args = parser.parse_args()
    
    # If output is empty, set to folder containing mask path
    if not args.output:
        args.output = os.path.dirname(args.mask_path)
    
    return args

def load_mask_file(mask_path):
    """
    Load mask pt file
    Args:
        mask_path: Path to mask file
    Returns:
        torch.Tensor or dict: Loaded mask data
    """
    try:
        print(f"Loading mask file: {mask_path}")
        mask_data = torch.load(mask_path, map_location='cpu')
        print(f"Successfully loaded mask file")
        print(f"Data type: {type(mask_data)}")
        
        if isinstance(mask_data, torch.Tensor):
            print(f"Tensor shape: {mask_data.shape}")
            print(f"Tensor data type: {mask_data.dtype}")
        elif isinstance(mask_data, dict):
            print(f"Dictionary keys: {list(mask_data.keys())}")
            for key, value in mask_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape {value.shape}, type {value.dtype}")
                else:
                    print(f"  {key}: type {type(value)}")
        
        return mask_data
        
    except FileNotFoundError:
        print(f"Error: Cannot find mask file {mask_path}")
        return None
    except Exception as e:
        print(f"Error occurred while loading mask file: {e}")
        return None

def load_node_lists(conflict_path, false_path):
    """
    Load node list files
    Args:
        conflict_path: Path to conflict node list file
        false_path: Path to false node list file
    Returns:
        tuple: (conflict_nodes, false_nodes) Two node lists
    """
    conflict_nodes = []
    false_nodes = []
    
    # Load conflict node list
    try:
        if os.path.exists(conflict_path):
            with open(conflict_path, 'r') as f:
                conflict_nodes = json.load(f)
            print(f"Successfully loaded conflict node list: {len(conflict_nodes)} nodes")
        else:
            print(f"Warning: conflict node file does not exist: {conflict_path}")
    except Exception as e:
        print(f"Error occurred while loading conflict node file: {e}")
    
    # Load false node list
    try:
        if os.path.exists(false_path):
            with open(false_path, 'r') as f:
                false_nodes = json.load(f)
            print(f"Successfully loaded false node list: {len(false_nodes)} nodes")
        else:
            print(f"Warning: false node file does not exist: {false_path}")
    except Exception as e:
        print(f"Error occurred while loading false node file: {e}")
    
    return conflict_nodes, false_nodes

def count_true_false_percentage(data):
    """
    Count the percentage of True and False in data
    Args:
        data: Data to be counted (can be Tensor or dict)
    Returns:
        tuple: (true_count, false_count, total_count, true_percentage, false_percentage)
    """
    true_count = 0
    false_count = 0
    total_count = 0
    
    if isinstance(data, torch.Tensor):
        # If it's a tensor, count directly
        true_count = torch.sum(data == True).item()
        false_count = torch.sum(data == False).item()
        total_count = data.numel()
    elif isinstance(data, dict):
        # If it's a dictionary, iterate through all values
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                true_count += torch.sum(value == True).item()
                false_count += torch.sum(value == False).item()
                total_count += value.numel()
            elif isinstance(value, bool):
                if value:
                    true_count += 1
                else:
                    false_count += 1
                total_count += 1
    else:
        print(f"Warning: Unsupported data type {type(data)}")
        return 0, 0, 0, 0.0, 0.0
    
    # Calculate percentage
    true_percentage = (true_count / total_count * 100) if total_count > 0 else 0.0
    false_percentage = (false_count / total_count * 100) if total_count > 0 else 0.0
    
    return true_count, false_count, total_count, true_percentage, false_percentage

def create_mask_structure(mask_data):
    """
    Create a mask with the same structure as mask_data
    Args:
        mask_data: Original mask data, used to get structure
    Returns:
        dict: Mask with the same structure as mask_data, initial value is False
    """
    mask_structure =mask_data.copy()
    mask_structure_ture={}
    # Copy the structure of mask_data, but set all values to False
    for key, tensor in mask_data.items():
        #mask_structure[key] = torch.zeros_like(tensor, dtype=torch.bool)
        mask_structure_ture[key] = torch.ones_like(tensor, dtype=torch.bool)
    
    return mask_structure,mask_structure_ture

def parse_node_name(node_name):
    """
    Parse node name and map to key 0-224
    Args:
        node_name: Node name string
    Returns:
        dict: Parse result, containing key, param_type, slice_info and other information
    """
    if node_name.startswith('m'):
        # Handle nodes starting with 'm'
        try:
            # Find the position of the first '.', if there's no '.' then take the entire string
            dot_pos = node_name.find('.')
            if dot_pos == -1:
                layer_str = node_name[1:]  # Take all characters after 'm'
            else:
                layer_str = node_name[1:dot_pos]  # Take all characters from after 'm' to before the first '.'
            
            layer_num = int(layer_str)
            
            # Calculate corresponding key: 7 parameters per layer, gate, up, down correspond to keys 4, 5, 6 respectively
            base_key = layer_num * 7
            keys = [base_key + 4, base_key + 5, base_key + 6]  # gate, up, down
            
            return {
                'type': 'm',
                'layer': layer_num,
                'keys': keys,
                'param_type': ['gate', 'up', 'down']  # Affects three parameters
            }
        except (ValueError, IndexError):
            print(f"Warning: Cannot parse node name {node_name}")
            return None
    
    elif node_name.startswith('a'):
        # Handle nodes starting with 'a'
        try:
            parts = node_name.split('.')
            if len(parts) != 3:
                print(f"Warning: Node name format error {node_name}")
                return None
            
            # Parse layer number - take all characters after 'a'
            layer_str = parts[0][1:]  # Remove 'a', take all characters after
            layer_num = int(layer_str)
            
            # Parse h/H part - take all characters after 'h' or 'H'
            h_part = parts[1]
            if h_part.startswith('h'):
                h_num = int(h_part[1:])  # Take all characters after 'h'
                slice_start = h_num * 128
                slice_end = slice_start + 128
                slice_info = (slice_start, slice_end)
            elif h_part.startswith('H'):
                h_num = int(h_part[1:])  # Take all characters after 'H'
                slice_start = h_num * 128
                slice_end = slice_start + 128
                slice_info = (slice_start, slice_end)
            else:
                print(f"Warning: Cannot parse h/H part {h_part}")
                return None
            
            # Parse parameter type
            param_type = parts[2]
            if param_type not in ['q', 'k', 'v', 'o']:
                print(f"Warning: Unknown parameter type {param_type}")
                return None
            
            # Calculate corresponding key: 7 parameters per layer, q,k,v,o correspond to keys 0,1,2,3 respectively
            param_to_key = {'q': 0, 'k': 1, 'v': 2, 'o': 3}
            key = layer_num * 7 + param_to_key[param_type]
            
            return {
                'type': 'a',
                'layer': layer_num,
                'key': key,
                'param_type': param_type,
                'slice_info': slice_info,
                'h_part': h_part
            }
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing node name {node_name}: {e}")
            return None
    
    else:
        print(f"Warning: Unsupported node name format {node_name}")
        return None

def apply_node_to_mask(mask_structure, node_name, mask_data):
    """
    Apply node to mask structure
    Args:
        mask_structure: Mask data structure
        node_name: Node name
        mask_data: Original mask data
    """
    parsed = parse_node_name(node_name)
    if parsed is None:
        return
    
    if parsed['type'] == 'm':
        # For gate, up, down parameters of layer N, copy corresponding values from mask_data
        for key in parsed['keys']:
            key_str = int(key)
            mask_structure[key_str] = mask_data[key_str].clone()
            print(f"  Copy key {key_str} (layer {parsed['layer']} {parsed['param_type'][parsed['keys'].index(key)]}) from mask_data (m type)")
    
    elif parsed['type'] == 'a':
        # For specific slices of specific parameter matrices, copy corresponding values from mask_data
        key_str = int(parsed['key'])
        slice_start, slice_end = parsed['slice_info']
        
        # Copy values from corresponding slices in mask_data as True
        mask_structure[key_str][slice_start:slice_end, :] = mask_data[key_str][slice_start:slice_end, :]
            print(f"  Copy key {key_str} (layer {parsed['layer']} {parsed['param_type']}) [{slice_start}:{slice_end}, :] from mask_data (a type)")

def count_mask_percentage(mask_structure, mask_name):
    """
    Count the percentage of True and False in mask structure
    Args:
        mask_structure: Mask data structure
        mask_name: Mask name
    """
    true_count = 0
    false_count = 0
    total_count = 0
    
    for key, tensor in mask_structure.items():
        if isinstance(tensor, torch.Tensor):
            true_count += torch.sum(tensor).item()
            false_count += torch.sum(~tensor).item()
            total_count += tensor.numel()
    
    true_percentage = (true_count / total_count * 100) if total_count > 0 else 0.0
    false_percentage = (false_count / total_count * 100) if total_count > 0 else 0.0
    
    print(f"\n{mask_name} statistics:")
    print(f"  Total element count: {total_count}")
    print(f"  True count: {true_count} ({true_percentage:.2f}%)")
    print(f"  False count: {false_count} ({false_percentage:.2f}%)")
    
    return true_count, false_count, total_count, true_percentage, false_percentage

def save_mask_structure(mask_structure, file_path, mask_name):
    """
    Save mask structure to file
    Args:
        mask_structure: Mask data structure
        file_path: Save path
        mask_name: Mask name
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save mask structure
        torch.save(mask_structure, file_path)
        print(f"  Successfully saved {mask_name} to: {file_path}")
        
    except Exception as e:
        print(f"  Error occurred while saving {mask_name}: {e}")

def main():
    """
    Main function
    """
    print("=== Mask Generation Tool ===")
    
    # Parse command line arguments
    args = parse_args()
    
    print(f"\nParameter settings:")
    print(f"  mask_path: {args.mask_path}")
    print(f"  conflict_node: {args.conflict_node}")
    print(f"  false_node: {args.false_node}")
    print(f"  output: {args.output}")
    
    # Check if file exists
    if not os.path.exists(args.mask_path):
        print(f"Error: mask file does not exist: {args.mask_path}")
        return
    
    # Load mask file
    mask_data = load_mask_file(args.mask_path)
    if mask_data is None:
        print("Cannot load mask file, program exiting")
        return
    
    # Load node lists
    conflict_nodes, false_nodes = load_node_lists(args.conflict_node, args.false_node)
    
    print(f"\n=== Loading completed ===")
    print(f"mask: {type(mask_data)}")
    print(f"conflict nodes: {len(conflict_nodes)} nodes")
    print(f"false nodes: {len(false_nodes)} nodes")
    
    # Count the percentage of True and False in mask_data
    print(f"\n=== Count percentage of True and False in mask_data ===")
    
    # Count mask_data
    true_count, false_count, total_count, true_percentage, false_percentage = count_true_false_percentage(mask_data)
    
    print(f"Total element count: {total_count}")
    print(f"True count: {true_count} ({true_percentage:.2f}%)")
    print(f"False count: {false_count} ({false_percentage:.2f}%)")
    
    # Verify percentage sum
    total_percentage = true_percentage + false_percentage
    print(f"Percentage sum: {total_percentage:.2f}%")
    
    if abs(total_percentage - 100.0) > 0.01:
        print(f"Warning: Percentage sum is not 100%, may contain non-boolean values")
    
    # Create safe_mask and conflict_mask
    print(f"\n=== Create safe_mask and conflict_mask ===")
    
    # Create mask structure
    safe_mask,safe_mask_ture = create_mask_structure(mask_data)
    conflict_mask,conflict_mask_ture = create_mask_structure(mask_data)
    
    print(f"Created mask with the same structure as mask_data")
    
    # Apply false_node and conflict_node to safe_mask
    print(f"\n=== Apply false_node and conflict_node to safe_mask ===")
    for node in false_nodes:
        apply_node_to_mask(safe_mask, node, safe_mask_ture)
    
    for node in conflict_nodes:
        apply_node_to_mask(safe_mask, node, safe_mask_ture)
    
    # Apply conflict_node to conflict_mask
    print(f"\n=== Apply conflict_node to conflict_mask ===")
    for node in conflict_nodes:
        apply_node_to_mask(conflict_mask, node, conflict_mask_ture)
    
    # Count True/False ratio of safe_mask and conflict_mask
    print(f"\n=== Count True/False ratio of safe_mask and conflict_mask ===")
    
    # Count two masks
    safe_stats = count_mask_percentage(safe_mask, "safe_mask")
    conflict_stats = count_mask_percentage(conflict_mask, "conflict_mask")
    
    # Save safe_mask and conflict_mask to output address
    print(f"\n=== Save mask files ===")
    
    # Generate save paths
    safe_mask_path = os.path.join(args.output, "safe_mask.pt")
    conflict_mask_path = os.path.join(args.output, "conflict_mask.pt")
    
    # Save two masks
    save_mask_structure(safe_mask, safe_mask_path, "safe_mask")
    save_mask_structure(conflict_mask, conflict_mask_path, "conflict_mask")
    
    print(f"\nAll mask files have been saved to: {args.output}")
    print(f"Ready, waiting for subsequent processing logic...")

if __name__ == "__main__":
    main()
