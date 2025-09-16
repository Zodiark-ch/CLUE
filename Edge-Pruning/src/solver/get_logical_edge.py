import os
import json
import argparse
import sys
import re
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/solver/"
    )
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-edge1", "--edge1", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/winogrande_edges/edges_13290_winogrande.json")
    parser.add_argument("-edge2", "--edge2", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/winogrande_edges/edges_35526_winogrande_or.json")
    parser.add_argument("-w", "--with_embedding_nodes", action="store_true", default=True)
    parser.add_argument("-o", "--output", type=str, default="")
    
    args = parser.parse_args()
    
    if args.output == "":
        
        args.output = os.path.join("/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/winogrande_edges", "edges_bool.json")
    else:
        raise ValueError("Output file must be specified when comparing two models") 
    
    return args

def add_o_suffix_to_nodes(edges):
    """
    For nodes starting with "a", if the last character is a digit instead of one of 'q', 'k', 'v', add suffix character '.o'
    Then add additional edge processing logic
    """
    modified_edges = []
    for edge in edges:
        from_node, to_node = edge
        
        # Process from_node
        if from_node.startswith("a") and from_node[-1].isdigit() and from_node[-1] not in ['q', 'k', 'v', 'o']:
            from_node = from_node + ".o"
        
        # Process to_node
        if to_node.startswith("a") and to_node[-1].isdigit() and to_node[-1] not in ['q', 'k', 'v', 'o']:
            to_node = to_node + ".o"
        
        modified_edges.append((from_node, to_node))
    
    # Additional edge processing logic
    additional_edges = []
    
    # Build set of all edges for fast lookup
    all_edges_set = set(modified_edges)
    
    # Build set of from_nodes to check if a node exists as from_node
    from_nodes_set = set(edge[0] for edge in modified_edges)
    
    for edge in modified_edges:
        from_node, to_node = edge
        
        # Check condition 1: to_node starts with "a" and last character is one of 'q', 'k', 'v'
        if (to_node.startswith("a") and 
            to_node[-1] in ['q', 'k', 'v']):
            
            # Check condition 2: this to_node does not exist as from_node in other edges
            if to_node not in from_nodes_set:
                
                # Check condition 3: this to_node with '.o' replacing 'q', 'k', 'v' (named o_node) exists as from_node in other edges
                o_node = to_node[:-1] + 'o'  # Replace last character with 'o'
                
                if o_node in from_nodes_set:
                    # All conditions met, add new edge [to_node, o_node]
                    additional_edges.append((to_node, o_node))
    
    # Add additional edges to result
    modified_edges.extend(additional_edges)
    
    return modified_edges

def classify_h_numbers(edges):
    """
    For nodes starting with "a", if last character is one of 'k', 'v', perform h number classification
    Example: 'a14.h6.q' -> 'a14.H1.q' (because h6 belongs to H1 group)
    """
    modified_edges = []
    for edge in edges:
        from_node, to_node,type = edge
        
        # Process from_node
        if from_node.startswith("a") and from_node[-1] in ['k', 'v']:
            from_node = process_h_classification(from_node)
        
        # Process to_node
        if to_node.startswith("a") and to_node[-1] in ['k', 'v']:
            to_node = process_h_classification(to_node)
        
        modified_edges.append((from_node, to_node,type))
    
    return modified_edges

def process_h_classification(node_name):
    """
    Process h number classification for single node
    Example: 'a14.h6.k' -> 'a14.H1.k'
    """
    if not node_name.startswith("a") or node_name[-1] not in ['k', 'v']:
        return node_name
    
    # Use regex to split node name
    # Match pattern: a number.h number.letter
    pattern = r'^([a-z]\d+)\.(h\d+)\.([qkv])$'
    match = re.match(pattern, node_name)
    
    if match:
        prefix, h_part, suffix = match.groups()
        
        # Extract number after h
        h_number = int(h_part[1:])  # Remove 'h', get number
        
        # Calculate H group number: every 4 numbers form a group
        h_group = h_number // 4
        
        # Build new node name
        new_node_name = f"{prefix}.H{h_group}.{suffix}"
        return new_node_name
    
    return node_name

def find_OR_gate_edges(edges_ns, edges_dn):
    """
    Find logical gate sub_edges that satisfy the following conditions:
    1. All exist in edges_dn but not in edges_ns
    2. to_node does not end with ".o"
    3. For any to_node, the same to_node should be found in another element of sub_edges
    """
    # Convert edges to set for fast lookup
    edges_ns_set = set(edges_ns)
    edges_dn_set = set(edges_dn)
    
    # Condition 1: Find edges that exist in edges_dn but not in edges_ns
    candidate_edges = edges_dn_set - edges_ns_set
    
    # Condition 2: Filter out edges where to_node ends with ".o"
    filtered_edges = []
    for edge in candidate_edges:
        from_node, to_node = edge
        if not to_node.endswith(".o"):
            filtered_edges.append(edge)
    
    # Condition 3: Ensure each to_node appears at least twice in sub_edges
    to_node_count = {}
    for edge in filtered_edges:
        to_node = edge[1]
        to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # Check edges in edges_ns_set, if a to_node appears only once as to_node, also add 1 to its count
    ns_to_node_count = {}
    for edge in edges_ns_set:
        to_node = edge[1]
        ns_to_node_count[to_node] = ns_to_node_count.get(to_node, 0) + 1
    
    # Add count of to_nodes that appear only once to to_node_count
    for to_node, count in ns_to_node_count.items():
        if count == 1:
            to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # Only keep edges where to_node appears >=2 times
    sub_edges = []
    for edge in filtered_edges:
        to_node = edge[1]
        if to_node_count[to_node] >= 2:
            sub_edges.append(edge)
    
    return sub_edges

def find_AND_gate_edges(edges_ns, edges_dn):
    """
    Find logical gate sub_edges that satisfy the following conditions:
    1. All exist in edges_ns but not in edges_dn
    2. to_node does not end with ".o"
    3. For any to_node, the same to_node should be found in another element of sub_edges
    """
    # Convert edges to set for fast lookup
    edges_ns_set = set(edges_ns)
    edges_dn_set = set(edges_dn)
    
    # Condition 1: Find edges that exist in edges_dn but not in edges_ns
    candidate_edges = edges_ns_set-edges_dn_set 
    
    # Condition 2: Filter out edges where to_node ends with ".o"
    filtered_edges = []
    for edge in candidate_edges:
        from_node, to_node = edge
        if not to_node.endswith(".o"):
            filtered_edges.append(edge)
    
    # Condition 3: Ensure each to_node appears at least twice in sub_edges
    to_node_count = {}
    for edge in filtered_edges:
        to_node = edge[1]
        to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # Check edges in edges_ns_set, if a to_node appears only once as to_node, also add 1 to its count
    dn_to_node_count = {}
    for edge in edges_dn_set:
        to_node = edge[1]
        dn_to_node_count[to_node] = dn_to_node_count.get(to_node, 0) + 1
    
    # Add count of to_nodes that appear only once to to_node_count
    for to_node, count in dn_to_node_count.items():
        if count == 1:
            to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # Only keep edges where to_node appears >=2 times
    sub_edges = []
    for edge in filtered_edges:
        to_node = edge[1]
        if to_node_count[to_node] >= 2:
            sub_edges.append(edge)
    
    return sub_edges

def create_bool_edges(edges_ns, edges_dn, or_edges, and_edges):
    """
    Create bool_edges, including all edges and adding labels:
    - If edge exists in sub_edges, label is "OR"
    - Otherwise label is "AND"
    - Each element expands from [from_node, to_node] to [from_node, to_node, label]
    """
    # Merge edges_ns and edges_dn, remove duplicates
    all_edges = list(set(edges_ns + edges_dn))
    
    # Convert sub_edges to set for fast lookup
    or_edges_set = set(or_edges)
    and_edges_set = set(and_edges)
    
    # Create bool_edges, add labels
    bool_edges = []
    for edge in all_edges:
        from_node, to_node = edge
        
        # Determine label
        if edge in or_edges_set:
            label = "OR"
        elif edge in and_edges_set:
            label = "AND"
        else:
            label = "ADDER"
        
        # Expand to triple
        bool_edges.append([from_node, to_node, label])
    
    return bool_edges

def validate_paths_to_resid_post(bool_edges):
    """
    Validate that all from_nodes in bool_edges can recursively find a path to "resid_post"
    If a node is found not in the adjacency table, edges will be dynamically added and re-validated
    """
    # Create a copy of bool_edges for dynamic modification
    current_bool_edges = bool_edges.copy()
    

    # Build adjacency table for easy lookup
    adjacency_list = {}
    for edge in current_bool_edges:
        from_node, to_node, label = edge
        if from_node not in adjacency_list:
            adjacency_list[from_node] = []
        adjacency_list[from_node].append(to_node)
    
    # Get all unique from_nodes
    all_from_nodes = set(edge[0] for edge in current_bool_edges)
    
    # Validate each from_node
    failed_nodes = []
    successful_nodes = []
    added_edges = []
    
    for from_node in all_from_nodes:
        success, failed_node = can_reach_resid_post(from_node, adjacency_list, set(),current_bool_edges)
        if success:
            successful_nodes.append(from_node)
        else:
            failed_nodes.append(failed_node)
            # If the failed node is not in the adjacency table, add an edge to resid_post
                
    
    # Return validation result
    result = {
        "total_from_nodes": len(all_from_nodes),
        "successful_nodes": len(successful_nodes),
        "failed_nodes": len(failed_nodes),
        "failed_node_list": failed_nodes,
        "is_valid": len(failed_nodes) == 0,
        "updated_bool_edges": current_bool_edges
    }
    
    return current_bool_edges

def can_reach_resid_post(node, adjacency_list, visited,current_bool_edges):
    """
    Recursively check if a node can reach resid_post
    Use depth-first search to avoid cycles
    Return tuple (success, failed_node), where failed_node is the node that caused the failure
    """
    # If this node has been visited, there is a cycle, return False and current node
    if node in visited:
        return False, node
    visited.add(node)
    
    # If current node is resid_post, return True and None
    if node == "resid_post":
        return True, None
    
    # If node is not in adjacency table, there are no outgoing edges, return False and current node
    if node not in adjacency_list:
        new_edge = [node, "resid_post", "ADDER"]
        current_bool_edges.append(new_edge)
        return False, node
    
    # Mark current node as visited
    
    # Recursively check all neighbor nodes
    for neighbor in adjacency_list[node]:
        success, node = can_reach_resid_post(neighbor, adjacency_list, visited.copy(),current_bool_edges)
        if success:
            return True, None
    
    # If all neighbors cannot reach resid_post, return False and current node
    return False, node

def save_bool_edges(bool_edges, output_path):
    """
    Save bool_edges to the specified output path
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as JSON format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bool_edges, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(bool_edges)} edges to {output_path}")

def sanitize_edges(edges):
    # First, add all q,k,v -> o edges
    new_edges_ = set()
    for edge in edges:
        if edge[0][0] == "a" and edge[0][-1] not in ["q", "k", "v"]:
            new_edges_.add(edge[0])
    for to in new_edges_:
        for suffix in [".q", ".k", ".v"]:
            from_ = to +  suffix
            edges.append((from_, to))
    while True:
        orig_len = len(edges)
        # Find all nodes that are destinations but not sources
        froms = set()
        tos = set()
        for edge in edges:
            froms.add(edge[0])
            if edge[1] != "resid_post":
                tos.add(edge[1])
        banned_tos = tos.difference(froms)
        edges = [e for e in edges if e[1] not in banned_tos]

        # Find qkv nodes that have no incoming edges, and remove the q -> o edge for them
        qkv_nodes = set()
        for edge in edges:
            if edge[1].endswith(".q"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".k"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".v"):
                qkv_nodes.add(edge[1])

        edges = [
            e for e in edges if not (
                (e[0].endswith(".q") and e[0] not in qkv_nodes) or
                (e[0].endswith(".k") and e[0] not in qkv_nodes) or
                (e[0].endswith(".v") and e[0] not in qkv_nodes)
            )
        ]
        if orig_len == len(edges):
           break

    return edges

def main():
    args = parse_args()
    edges_ns = json.load(open(args.edge1))
    edges_dn = json.load(open(args.edge2))
    print("Original edge count:", len(edges_ns), len(edges_dn))
    
    # if strict, uncomment the following
    
    # # Original sanitize_edges processing
    # edges_ns = sanitize_edges(edges_ns)
    # edges_dn = sanitize_edges(edges_dn)
    # print("After sanitize:", len(edges_ns), len(edges_dn))
    
    # Apply first function: add .o suffix to a nodes ending with numbers
    edges_ns = add_o_suffix_to_nodes(edges_ns)
    edges_dn = add_o_suffix_to_nodes(edges_dn)
    print("After adding .o suffix:", len(edges_ns), len(edges_dn))
    
    # Apply second function: classify h numbers for nodes ending with k,v
    # edges_ns = classify_h_numbers(edges_ns)
    # edges_dn = classify_h_numbers(edges_dn)
    # print("After h number classification:", len(edges_ns), len(edges_dn))
    
    # Remove duplicate edges
    edges_ns = list(set(edges_ns))
    edges_dn = list(set(edges_dn))
    print("After deduplication:", len(edges_ns), len(edges_dn))
    
    # Find logical gate sub_edges
    OR_edges = find_OR_gate_edges(edges_ns, edges_dn)
    AND_edges = find_AND_gate_edges(edges_ns, edges_dn)
    print("OR gate edge count:", len(OR_edges))
    print("AND gate edge count:", len(AND_edges))
    
    # Create bool_edges, merge all edges and add labels
    bool_edges = create_bool_edges(edges_ns, edges_dn, OR_edges,AND_edges)
    print("bool_edges count:", len(bool_edges))
    
    # Validate that all from_nodes can reach resid_post
    validation_result = validate_paths_to_resid_post(bool_edges)
    print("Path validation result:", len(validation_result))
    validation_result=classify_h_numbers(validation_result)
    validation_result=list(set(tuple(edge) for edge in validation_result))
    print("After h number classification:", len(validation_result))
    
    # Save bool_edges to output file
    save_bool_edges(validation_result, args.output)
    print(f"bool_edges saved to: {args.output}")
    
    
    
    
if __name__ == '__main__':
    main()