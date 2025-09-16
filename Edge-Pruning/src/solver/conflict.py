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

class Circuit:
    def __init__(self, json_file_path, leaf_nodes=None):
        """
        Initialize Circuit class
        Args:
            json_file_path: JSON file path containing circuit data
            leaf_nodes: List of leaf nodes for topological sorting to build circuit
        """
        self.circuit = {}
        self.NOR = False  # Default NOR parameter
        self.load_circuit(json_file_path, leaf_nodes)
    
    def load_circuit(self, json_file_path, leaf_nodes=None):
        """
        Load circuit data from JSON file and reorganize into dictionary structure by topological sorting
        Args:
            json_file_path: JSON file path
            leaf_nodes: List of leaf nodes for topological sorting to build circuit
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Initialize known_node_list starting from leaf nodes
            known_node_list = set(leaf_nodes) if leaf_nodes else set()
            
            # Create mapping of all to_nodes for checking dependencies
            to_node_dependencies = {}
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    if to_node not in to_node_dependencies:
                        to_node_dependencies[to_node] = []
                    to_node_dependencies[to_node].append((from_node, node_type))
            
            # Topological sorting to build circuit
            while to_node_dependencies:
                # Find all to_nodes whose dependencies are satisfied
                ready_to_process = []
                for to_node, dependencies in to_node_dependencies.items():
                    all_dependencies_met = True
                    for from_node, _ in dependencies:
                        if from_node not in known_node_list:
                            all_dependencies_met = False
                            break
                    
                    if all_dependencies_met:
                        ready_to_process.append(to_node)
                
                # If no processable nodes found, circular dependency exists
                if not ready_to_process:
                    print(f"Warning: Circular dependency exists, cannot complete topological sorting")
                    # Add remaining nodes directly to circuit
                    # for to_node, dependencies in to_node_dependencies.items():
                    #     if to_node not in self.circuit:
                    #         self.circuit[to_node] = {}
                    #     for from_node, node_type in dependencies:
                    #         if node_type not in self.circuit[to_node]:
                    #             self.circuit[to_node][node_type] = []
                    #         self.circuit[to_node][node_type].append(from_node)
                    break
                
                # Process all processable to_nodes
                for to_node in ready_to_process:
                    dependencies = to_node_dependencies[to_node]
                    
                    # Build circuit structure
                    if to_node not in self.circuit:
                        self.circuit[to_node] = {}
                    
                    for from_node, node_type in dependencies:
                        if node_type not in self.circuit[to_node]:
                            self.circuit[to_node][node_type] = []
                        self.circuit[to_node][node_type].append(from_node)
                    
                    # Add to_node to known_node_list
                    known_node_list.add(to_node)
                    
                    # Remove from pending list
                    del to_node_dependencies[to_node]
                    
        except FileNotFoundError:
            print(f"Error: File not found {json_file_path}")
        except json.JSONDecodeError:
            print(f"Error: JSON file format incorrect {json_file_path}")
        except Exception as e:
            print(f"Error loading circuit data: {e}")
    
    def bool_validation(self, VALUE_retain, VALUE_forget, input_retain=None, input_forget=None, NOR=None):
        """
        Boolean validation of node assignments in circuit
        Args:
            VALUE_retain: List containing [node, bool_value] for non-NOR mode
            VALUE_forget: List containing [node, bool_value] for NOR mode
            input_retain: List containing leaf nodes [node, bool_value] for non-NOR mode
            input_forget: List containing leaf nodes [node, bool_value] for NOR mode
            NOR: Whether to use NOR mode (if None, use instance NOR attribute)
        Returns:
            bool: Returns True if validation passes, False otherwise
            list: Returns list of [node, numeric_value] if validation passes, empty list otherwise
        """
        # If NOR parameter is None, use instance NOR attribute
        if NOR is None:
            NOR = self.NOR
        
        # Create real-time value list containing all node values in current circuit
        real_time_values = {}
        
        # Collect all nodes in circuit
        all_nodes = set()
        for to_node, type_groups in self.circuit.items():
            all_nodes.add(to_node)
            for from_nodes in type_groups.values():
                all_nodes.update(from_nodes)
        
        # Initialize real-time value list
        for node in all_nodes:
            node_value = None
            
            if NOR:
                # If NOR mode, reference VALUE_forget
                for n, value in VALUE_forget:
                    if n == node:
                        node_value = value
                        break
            else:
                # If not NOR mode, reference VALUE_retain
                for n, value in VALUE_retain:
                    if n == node:
                        node_value = value
                        break
            
            real_time_values[node] = node_value
        
        # Update leaf node values in input_retain and input_forget
        if NOR:
            # NOR mode: use input_forget to update leaf node values
            for node, value in input_forget:
                if node in real_time_values:
                    real_time_values[node] = value
        else:
            # Non-NOR mode: use input_retain to update leaf node values
            if input_retain is not None:
                for node, value in input_retain:
                    if node in real_time_values:
                        real_time_values[node] = value
        
        
        # Perform bool calculation for each key in self.circuit
        for to_node, type_groups in self.circuit.items():
            for node_type, from_nodes in type_groups.items():
                # Get from_node value
                from_node_values = []
                for from_node in from_nodes:
                    from_node_value = real_time_values.get(from_node)
                    if from_node_value is None:
                        # If from_node value is None, cannot perform calculation
                        return False, []
                    from_node_values.append(from_node_value)
                
                # Perform bool calculation based on NOR mode and node type
                if NOR:
                    # NOR mode: all from_nodes are negated first
                    inverted_from_values = [not val for val in from_node_values]
                    
                    if node_type in ["OR", "ADDER"]:
                        # OR/ADDER: compute_result = from_node1 and from_node2 and from_node3 ...
                        compute_result = all(inverted_from_values)
                    elif node_type == "AND":
                        # AND: compute_result = from_node1 or from_node2 or from_node3 ...
                        compute_result = any(inverted_from_values)
                    else:
                        # Unknown node type
                        return False, []
                else:
                    # Non-NOR mode: from_node values used directly
                    if node_type in ["AND", "ADDER"]:
                        # AND/ADDER: compute_result = from_node1 and from_node2 and from_node3 ...
                        compute_result = all(from_node_values)
                    elif node_type == "OR":
                        # OR: compute_result = from_node1 or from_node2 or from_node3 ...
                        compute_result = any(from_node_values)
                    else:
                        # Unknown node type
                        return False, []
                
                # Check to_node value
                to_node_value = real_time_values.get(to_node)
                
                if to_node_value is None:
                    # If to_node value is None, assign compute_result to to_node
                    if NOR:
                        real_time_values[to_node] = not compute_result
                    else:
                        real_time_values[to_node] = compute_result
                else:
                    # If to_node is not None, check if compute_result and to_node are equal, since to_node_value needs negation, equal means not equal
                    if NOR:
                        if compute_result == to_node_value:
                            return False, []
                    else:
                        if compute_result != to_node_value:
                            return False, []
        
        # Validation passed, return list of each node paired with value from real-time value list
        
        return True, real_time_values

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-circuit1", "--circuit1", type=str, default="")
    parser.add_argument("-circuit2", "--circuit2", type=str, default="")
    parser.add_argument("-circuit3", "--circuit3", type=str, default="")
    parser.add_argument("-circuit4", "--circuit4", type=str, default="")
    parser.add_argument("-circuit5", "--circuit5", type=str, default="")
    parser.add_argument("-circuit6", "--circuit6", type=str, default="")
    parser.add_argument("-circuit7", "--circuit7", type=str, default="")
    parser.add_argument("-w", "--with_embedding_nodes", action="store_true", default=True)
    
    args = parser.parse_args()
    
    return args




def initialize_values(args, leaf_nodes_by_circuit):
    """
    Initialize VALUE_retain, VALUE_forget, input_retain and input_forget based on circuit files in args
    Args:
        args: Parameter object containing circuit file paths
        leaf_nodes_by_circuit: Dictionary of leaf nodes for each circuit
    Returns:
        tuple: (VALUE_retain, VALUE_forget, input_retain, input_forget) Four lists containing [node, bool_value]
    """
    VALUE_retain = []
    VALUE_forget = []
    input_retain = []
    input_forget = []
    
    # Get all circuit file paths
    circuit_files = []
    for i in range(1, 7):  # Check circuit1 to circuit6
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_files.append((i, circuit_path))
    
    if not circuit_files:
        print("Error: No valid circuit files found!")
        return VALUE_retain, VALUE_forget, input_retain, input_forget
    
    print(f"Found {len(circuit_files)} circuit files")
    
    # Process each circuit file
    for circuit_num, circuit_path in circuit_files:
        try:
            # Load JSON data directly to get all nodes, without relying on Circuit class
            with open(circuit_path, 'r') as f:
                data = json.load(f)
            
            # Collect all nodes in this circuit
            circuit_nodes = set()
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    circuit_nodes.add(from_node)
                    circuit_nodes.add(to_node)
            
            circuit_nodes_list = list(circuit_nodes)
            print(f"Circuit{circuit_num} ({circuit_path}) contains {len(circuit_nodes_list)} nodes")
            
            # Assign nodes based on circuit number
            if circuit_num == 1:
                # Add all nodes from circuit1 to VALUE_forget with value False
                for node in circuit_nodes_list:
                    VALUE_forget.append([node, False])
                print(f"  → Added to VALUE_forget (value: False)")
            else:
                # Add all nodes from other circuits to VALUE_retain with value True
                for node in circuit_nodes_list:
                    VALUE_retain.append([node, True])
                print(f"  → Added to VALUE_retain (value: True)")
                
        except Exception as e:
            print(f"Error occurred while processing circuit{circuit_num}: {e}")
            continue
    
    # Initialize input_forget and input_retain
    print("\nInitializing input_forget and input_retain...")
    
    # input_forget nodes are leaf_nodes_by_circuit[1] nodes, value defaults to False
    if 1 in leaf_nodes_by_circuit:
        circuit1_leaf_nodes = leaf_nodes_by_circuit[1]
        for node in circuit1_leaf_nodes:
            input_forget.append([node, False])
        print(f"  input_forget: {len(input_forget)} nodes (from Circuit1 leaf nodes, value defaults to False)")
    else:
        print("  Warning: No Circuit1 leaf nodes found")
    
    # input_retain nodes are the set of all nodes from leaf_nodes_by_circuit[2:], value defaults to True
    all_other_leaf_nodes = set()
    for circuit_num in range(2, 7):  # circuit2 to circuit6
        if circuit_num in leaf_nodes_by_circuit:
            all_other_leaf_nodes.update(leaf_nodes_by_circuit[circuit_num])
    
    for node in all_other_leaf_nodes:
        input_retain.append([node, True])
    print(f"  input_retain: {len(input_retain)} nodes (from Circuit2-6 leaf nodes, value defaults to True)")
    
    # Deduplicate all four lists
    #print("\nPerforming deduplication...")
    
    # Deduplicate VALUE_retain
    seen_nodes = set()
    unique_VALUE_retain = []
    for node, value in VALUE_retain:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_VALUE_retain.append([node, value])
    VALUE_retain = unique_VALUE_retain
    #print(f"  VALUE_retain after deduplication: {len(VALUE_retain)} nodes")
    
    # Deduplicate VALUE_forget
    seen_nodes = set()
    unique_VALUE_forget = []
    for node, value in VALUE_forget:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_VALUE_forget.append([node, value])
    VALUE_forget = unique_VALUE_forget
    #print(f"  VALUE_forget after deduplication: {len(VALUE_forget)} nodes")
    
    # Deduplicate input_retain
    seen_nodes = set()
    unique_input_retain = []
    for node, value in input_retain:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_input_retain.append([node, value])
    input_retain = unique_input_retain
    #print(f"  input_retain after deduplication: {len(input_retain)} nodes")
    
    # Deduplicate input_forget
    seen_nodes = set()
    unique_input_forget = []
    for node, value in input_forget:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_input_forget.append([node, value])
    input_forget = unique_input_forget
    #print(f"  input_forget after deduplication: {len(input_forget)} nodes")
    
    return VALUE_retain, VALUE_forget, input_retain, input_forget



def modify(VALUE_retain, VALUE_forget, input_retain, input_forget, circuits):
    """
    Modify values of VALUE_retain, VALUE_forget, input_retain, input_forget
    Args:
        VALUE_retain: Retain node list [[node, bool_value], ...]
        VALUE_forget: Forget node list [[node, bool_value], ...]
        input_retain: Input retain node list [[node, bool_value], ...]
        input_forget: Input forget node list [[node, bool_value], ...]
        circuits: Dictionary of all circuits {circuit_num: circuit}
    Returns:
        tuple: (VALUE_retain, VALUE_forget, input_retain, input_forget) Modified four lists
    """
    import copy
    
    # Create a copy to avoid modifying original data
    VALUE_retain = copy.deepcopy(VALUE_retain)
    VALUE_forget = copy.deepcopy(VALUE_forget)
    input_retain = copy.deepcopy(input_retain)
    input_forget = copy.deepcopy(input_forget)
    
    print("\nStarting modify function...")
    
    # 1. Set all node values in VALUE_forget and VALUE_retain to None
    print("1. Setting all VALUE_forget and VALUE_retain values to None...")
    for i in range(len(VALUE_forget)):
        VALUE_forget[i][1] = None
    for i in range(len(VALUE_retain)):
        VALUE_retain[i][1] = None
    
    # 2. Find nodes that appear only in circuit1 and not in other circuits
    print("2. Finding nodes that appear only in circuit1...")
    circuit1_nodes = set()
    other_circuit_nodes = set()
    
    # Get all nodes of circuit1
    if 1 in circuits:
        circuit1 = circuits[1]
        for to_node, type_groups in circuit1.circuit.items():
            circuit1_nodes.add(to_node)
            for from_nodes in type_groups.values():
                circuit1_nodes.update(from_nodes)
    
    # Get all nodes of other circuits
    for circuit_num, circuit in circuits.items():
        if circuit_num != 1:
            for to_node, type_groups in circuit.circuit.items():
                other_circuit_nodes.add(to_node)
                for from_nodes in type_groups.values():
                    other_circuit_nodes.update(from_nodes)
    
    # # Nodes appearing only in circuit1
    # only_in_circuit1 = circuit1_nodes - other_circuit_nodes
    # for node in only_in_circuit1:
    #     # Set value to False in VALUE_forget
    #     for i, (n, _) in enumerate(VALUE_forget):
    #         if n == node:
    #             VALUE_forget[i][1] = False
    #             break
    # print(f"  Found {len(only_in_circuit1)} nodes appearing only in circuit1, set to False")
    
    # # 3. Find nodes that appear only in other circuits and not in circuit1
    # # print("3. Finding nodes that appear only in other circuits...")
    # only_in_other_circuits = other_circuit_nodes - circuit1_nodes
    # for node in only_in_other_circuits:
    #     # Set value to True in VALUE_retain
    #     for i, (n, _) in enumerate(VALUE_retain):
    #         if n == node:
    #             VALUE_retain[i][1] = True
    #             break
    # # print(f"  Found {len(only_in_other_circuits)} nodes appearing only in other circuits, set to True")
    
    # 4. Starting from resid_post in circuit1, recursively process OR and ADDER type from_nodes
    print("4. Processing OR and ADDER type nodes in circuit1...")
    if 1 in circuits:
        circuit1 = circuits[1]
        processed_nodes = set()
        
        def process_circuit1_node(node):
            """Recursively process nodes in circuit1"""
            if node in processed_nodes:
                return
            
            processed_nodes.add(node)
            
            if node in circuit1.circuit:
                for node_type, from_nodes in circuit1.circuit[node].items():
                    if node_type in ["OR", "ADDER"]:
                        for from_node in from_nodes:
                            # Check from_node value in VALUE_forget
                            from_node_value = None
                            for i, (n, v) in enumerate(VALUE_forget):
                                if n == from_node:
                                    from_node_value = v
                                    break
                            
                            if from_node_value is None:
                                # Set to False
                                for i, (n, v) in enumerate(VALUE_forget):
                                    if n == from_node:
                                        VALUE_forget[i][1] = False
                                        break
                                #print(f"    Node {from_node} set to False")
                            elif from_node_value != False:
                                # Error
                                raise ValueError(f"Node {from_node} value is not False, but {from_node_value}")
                            
                            # Recursively process this from_node
                            process_circuit1_node(from_node)
        
        # Start processing from resid_post
        resid_post_nodes = []
        for to_node in circuit1.circuit.keys():
            if "resid_post" in to_node:
                for i, (n, v) in enumerate(VALUE_forget):
                                    if n == 'resid_post':
                                        VALUE_forget[i][1] = False
                                        break
                resid_post_nodes.append(to_node)
        
        for resid_post in resid_post_nodes:
            print(f"  Starting processing from {resid_post}...")
            process_circuit1_node(resid_post)
    
    # 5. Starting from resid_post in other circuits, recursively process AND and ADDER type from_nodes
    print("5. Processing AND and ADDER type nodes in other circuits...")
    for circuit_num, circuit in circuits.items():
        if circuit_num != 1:
            processed_nodes = set()
            
            def process_other_circuit_node(node):
                """Recursively process nodes in other circuits"""
                if node in processed_nodes:
                    return
                
                processed_nodes.add(node)
                
                if node in circuit.circuit:
                    for node_type, from_nodes in circuit.circuit[node].items():
                        if node_type in ["AND", "ADDER"]:
                            for from_node in from_nodes:
                                # Check from_node value in VALUE_retain
                                from_node_value = None
                                for i, (n, v) in enumerate(VALUE_retain):
                                    if n == from_node:
                                        from_node_value = v
                                        break
                                
                                if from_node_value is None:
                                    # Set to True
                                    for i, (n, v) in enumerate(VALUE_retain):
                                        if n == from_node:
                                            VALUE_retain[i][1] = True
                                            break
                                    #print(f"    Node {from_node} set to True")
                                elif from_node_value != True:
                                    # Error
                                    raise ValueError(f"Node {from_node} value is not True, but {from_node_value}")
                                
                                # Recursively process this from_node
                                process_other_circuit_node(from_node)
            
            # Start processing from resid_post
            resid_post_nodes = []
            for to_node in circuit.circuit.keys():
                if "resid_post" in to_node:
                    for i, (n, v) in enumerate(VALUE_retain):
                                        if n == "resid_post":
                                            VALUE_retain[i][1] = True
                                            break
                    resid_post_nodes.append(to_node)
            
            for resid_post in resid_post_nodes:
                print(f"  Starting processing from Circuit{circuit_num}'s {resid_post}...")
                process_other_circuit_node(resid_post)
    

    
    # 8. Update input_retain values
    print("8. Updating input_retain values...")
    # Set all node values in input_retain to None
    for i in range(len(input_retain)):
        input_retain[i][1] = None
    
    # Iterate VALUE_retain, update values in input_retain
    for node, value in VALUE_retain:
        if value is not None:
            for i, (n, v) in enumerate(input_retain):
                if n == node:
                    input_retain[i][1] = value
                    print(f"    Node {node} in input_retain updated to {value}")
                    break
    
    # 9. Update input_forget values
    print("9. Updating input_forget values...")
    # Set all node values in input_forget to None
    for i in range(len(input_forget)):
        input_forget[i][1] = None
    
    # Iterate VALUE_forget, update values in input_forget
    for node, value in VALUE_forget:
        if value is not None:
            for i, (n, v) in enumerate(input_forget):
                if n == node:
                    input_forget[i][1] = value
                    print(f"    Node {node} in input_forget updated to {value}")
                    break
    
    # 10. Count nodes with value=None
    print("10. Counting nodes with value=None...")
    none_count_retain = sum(1 for _, v in VALUE_retain if v is None)
    none_count_forget = sum(1 for _, v in VALUE_forget if v is None)
    none_count_input_retain = sum(1 for _, v in input_retain if v is None)
    none_count_input_forget = sum(1 for _, v in input_forget if v is None)
    
    print(f"  Nodes with value=None in VALUE_retain: {none_count_retain}")
    print(f"  Nodes with value=None in VALUE_forget: {none_count_forget}")
    print(f"  Nodes with value=None in input_retain: {none_count_input_retain}")
    print(f"  Nodes with value=None in input_forget: {none_count_input_forget}")
    
    # 11. Count nodes with None in input_retain and input_forget to form uncertain_node_list
    print("11. Counting uncertain node list...")
    uncertain_node_list_retain = []
    uncertain_node_list_forget = []
    
    for node, value in input_retain:
        if value is None:
            uncertain_node_list_retain.append(node)
    
    for node, value in input_forget:
        if value is None:
            uncertain_node_list_forget.append(node)
    
    print(f"  uncertain_node_list_retain: {len(uncertain_node_list_retain)} nodes")
    if uncertain_node_list_retain:
        print(f"    Node list: {uncertain_node_list_retain}")
    
    print(f"  uncertain_node_list_forget: {len(uncertain_node_list_forget)} nodes")
    if uncertain_node_list_forget:
        print(f"    Node list: {uncertain_node_list_forget}")
    
    print("modify function execution completed")
    return VALUE_retain, VALUE_forget, input_retain, input_forget, uncertain_node_list_retain, uncertain_node_list_forget


def investigate_leaf_nodes(circuit_files):
    """
    Investigate leaf nodes of each circuit
    Definition of leaf node: a node that appears only in from_node, not in to_node
    Args:
        circuit_files: List of (circuit_num, circuit_path) tuples
    Returns:
        dict: {circuit_num: leaf_nodes_list} Leaf node list for each circuit
    """
    leaf_nodes_by_circuit = {}
    
    for circuit_num, circuit_path in circuit_files:
        try:
            print(f"\nInvestigating leaf nodes of Circuit{circuit_num}...")
            
            # Load JSON data directly, without relying on Circuit class
            with open(circuit_path, 'r') as f:
                data = json.load(f)
            
            # Collect all to_nodes and from_nodes
            to_nodes = set()
            from_nodes = set()
            
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    to_nodes.add(to_node)
                    from_nodes.add(from_node)
            
            # Leaf nodes: appear only in from_node, not in to_node
            leaf_nodes = from_nodes - to_nodes
            leaf_nodes_list = list(leaf_nodes)
            
            leaf_nodes_by_circuit[circuit_num] = leaf_nodes_list
            
            print(f"  Circuit{circuit_num} contains {len(leaf_nodes_list)} leaf nodes")
            print(f"  Leaf node list: {leaf_nodes_list}")
            
        except Exception as e:
            print(f"Error occurred while investigating leaf nodes of Circuit{circuit_num}: {e}")
            leaf_nodes_by_circuit[circuit_num] = []
    
    return leaf_nodes_by_circuit


def main():
    """
    Main function: build circuit class, initialize VALUE1 and VALUE0
    """
    # Parse command line arguments
    args = parse_args()
    
    # Build all circuit classes
    circuits = {}
    circuit_files = []
    
    # Get all valid circuit files
    for i in range(1, 7):  # Check circuit1 to circuit6
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_files.append((i, circuit_path))
    
    if not circuit_files:
        print("Error: No valid circuit files found!")
        return
    
    print(f"Found {len(circuit_files)} circuit files")
    
    # Investigate leaf nodes
    print("\nStarting leaf node investigation...")
    leaf_nodes_by_circuit = investigate_leaf_nodes(circuit_files)
    
    # Build circuit class and set NOR parameters
    for circuit_num, circuit_path in circuit_files:
        try:
            print(f"\nBuilding Circuit{circuit_num}...")
            
            # Get leaf nodes for this circuit
            leaf_nodes = leaf_nodes_by_circuit.get(circuit_num, [])
            print(f"  Using {len(leaf_nodes)} leaf nodes for topological sorting")
            
            # Create Circuit instance, pass leaf nodes
            circuit = Circuit(circuit_path, leaf_nodes)
            
            # Set NOR parameter: True for circuit1, False for others
            if circuit_num == 1:
                circuit.NOR = True
                print(f"  Circuit{circuit_num} set NOR=True")
            else:
                circuit.NOR = False
                print(f"  Circuit{circuit_num} set NOR=False")
            
            circuits[circuit_num] = circuit
            print(f"  Circuit{circuit_num} built successfully, contains {len(circuit.circuit)} to_nodes")
            
        except Exception as e:
            print(f"Error occurred while building Circuit{circuit_num}: {e}")
            continue
    
    if not circuits:
        print("Error: No circuits built successfully!")
        return
    
    print(f"\nSuccessfully built {len(circuits)} circuits")
    
    # Initialize VALUE_retain, VALUE_forget, input_retain and input_forget
    print("\nStarting initialization of VALUE_retain, VALUE_forget, input_retain and input_forget...")
    VALUE_retain, VALUE_forget, input_retain, input_forget = initialize_values(args, leaf_nodes_by_circuit)
    
    print(f"\nInitialization complete:")
    print(f"VALUE_retain (Retain nodes): {len(VALUE_retain)} nodes")
    print(f"VALUE_forget (Forget nodes): {len(VALUE_forget)} nodes")
    print(f"input_retain (Input retain nodes): {len(input_retain)} nodes")
    print(f"input_forget (Input forget nodes): {len(input_forget)} nodes")
    
    
    # Check boolean validation of current VALUE_forget and VALUE_retain assignments
    print("\nChecking boolean validation of current assignments...")
    initial_validations = {}
    has_validation_failures = False
    
    for circuit_num, circuit in circuits.items():
        is_valid, _ = circuit.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
        initial_validations[circuit_num] = is_valid
        if not is_valid:
            has_validation_failures = True
            print(f"  Circuit{circuit_num} Boolean validation failed")
        else:
            print(f"  Circuit{circuit_num} Boolean validation passed")
    
    if has_validation_failures:
        print("Warning: Current assignment boolean validation failed, may affect enumeration algorithm effectiveness")
    else:
        print("✓ Current assignment boolean validation passed")
    
    # Call modify function to modify VALUE_retain, VALUE_forget, input_retain, input_forget
    print("\nStarting call to modify function...")
    VALUE_retain, VALUE_forget, input_retain, input_forget, uncertain_node_list_retain, uncertain_node_list_forget = modify(
        VALUE_retain, VALUE_forget, input_retain, input_forget, circuits
    )
    
    print(f"\nmodify function execution completed:")
    print(f"Modified VALUE_retain node count: {len(VALUE_retain)}")
    print(f"Modified VALUE_forget node count: {len(VALUE_forget)}")
    print(f"Modified input_retain node count: {len(input_retain)}")
    print(f"Modified input_forget node count: {len(input_forget)}")
    
    # Process combination validation of uncertain nodes
    print("\nStarting processing of uncertain node combination validation...")
    import itertools
    import copy
    
    # Initialize all_result_lists as a dictionary
    all_result_lists = {
        'circuit1': [],
        'circuit2': [],
        'circuit3': [],
        'circuit4': [],
        'circuit5': [],
        'circuit6': [],
        'circuit7': []
    }
    
    # Process combinations for uncertain_node_list_forget
    if uncertain_node_list_forget:
        print(f"\nProcessing {len(uncertain_node_list_forget)} nodes in uncertain_node_list_forget...")
        N = len(uncertain_node_list_forget)
        total_combinations = 2 ** N
        print(f"  Total {total_combinations} combinations to validate")
        
        for i, combination in enumerate(itertools.product([0, 1], repeat=N)):
            print(f"  Validating combination {i+1}/{total_combinations}: {combination}")
            
            # Create temporary input_forget copy
            temp_input_forget = copy.deepcopy(input_forget)
            
            # Assign combination values to corresponding nodes
            for j, node in enumerate(uncertain_node_list_forget):
                for k, (n, v) in enumerate(temp_input_forget):
                    if n == node:
                        temp_input_forget[k][1] = bool(combination[j])
                        break
            
            # Validate circuit1
            if 1 in circuits:
                circuit1 = circuits[1]
                is_valid, result_list = circuit1.bool_validation(VALUE_retain, VALUE_forget, input_retain, temp_input_forget)
                
                if is_valid:
                    print(f"    ✓ Circuit1 validation passed, saving result_list")
                    all_result_lists['circuit1'].append({
                        'type': 'forget',
                        'combination': combination,
                        'nodes': uncertain_node_list_forget,
                        'result_lists': result_list
                    })
                else:
                    print(f"    ✗ Circuit1 validation failed")
            else:
                print(f"    Warning: Circuit1 not found")
    else:
        print(f"\nuncertain_node_list_forget is empty, directly validating original input_forget...")
        
        if 1 in circuits:
            circuit1 = circuits[1]
            is_valid, result_list = circuit1.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
            
            if is_valid:
                print(f"  ✓ Circuit1 validation passed, saving result_list")
                all_result_lists['circuit1'].append({
                    'type': 'forget',
                    'combination': (),  # Empty combination
                    'nodes': [],  # Empty node list
                    'result_lists': result_list
                })
            else:
                print(f"  ✗ Circuit1 validation failed")
        else:
            print(f"  Warning: Circuit1 not found")
    
    # Process combinations for uncertain_node_list_retain
    if uncertain_node_list_retain:
        print(f"\nProcessing {len(uncertain_node_list_retain)} nodes in uncertain_node_list_retain...")
        M = len(uncertain_node_list_retain)
        total_combinations = 2 ** M
        print(f"  Total {total_combinations} combinations to validate")
        
        for i, combination in enumerate(itertools.product([0, 1], repeat=M)):
            print(f"  Validating combination {i+1}/{total_combinations}: {combination}")
            
            # Create temporary input_retain copy
            temp_input_retain = copy.deepcopy(input_retain)
            
            # Assign combination values to corresponding nodes
            for j, node in enumerate(uncertain_node_list_retain):
                for k, (n, v) in enumerate(temp_input_retain):
                    if n == node:
                        temp_input_retain[k][1] = bool(combination[j])
                        break
            
            # Validate other circuits except circuit1
            all_other_circuits_valid = True
            result_lists = {}
            
            for circuit_num, circuit in circuits.items():
                if circuit_num != 1:  # Skip circuit1
                    is_valid, result_list = circuit.bool_validation(VALUE_retain, VALUE_forget, temp_input_retain, input_forget)
                    
                    if is_valid:
                        result_lists[circuit_num] = result_list
                    else:
                        all_other_circuits_valid = False
                        print(f"    ✗ Circuit{circuit_num} validation failed")
                        break
            
            if all_other_circuits_valid:
                print(f"    ✓ All other circuits validation passed, saving result_lists")
                # Save results to corresponding circuits
                for circuit_num, result_list in result_lists.items():
                    circuit_key = f'circuit{circuit_num}'
                    all_result_lists[circuit_key].append({
                        'type': 'retain',
                        'combination': combination,
                        'nodes': uncertain_node_list_retain,
                        'result_lists': result_list
                    })
    else:
        print(f"\nuncertain_node_list_retain is empty, directly validating original input_retain...")
        # Validate other circuits except circuit1
        all_other_circuits_valid = True
        result_lists = {}
        
        for circuit_num, circuit in circuits.items():
            if circuit_num != 1:  # Skip circuit1
                is_valid, result_list = circuit.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
                
                if is_valid:
                    result_lists[circuit_num] = result_list
                else:
                    all_other_circuits_valid = False
                    print(f"  ✗ Circuit{circuit_num} validation failed")
                    break
        
        if all_other_circuits_valid:
            print(f"  ✓ All other circuits validation passed, saving result_lists")
            # Save results to corresponding circuits
            for circuit_num, result_list in result_lists.items():
                circuit_key = f'circuit{circuit_num}'
                all_result_lists[circuit_key].append({
                    'type': 'retain',
                    'combination': (),  # Empty combination
                    'nodes': [],  # Empty node list
                    'result_lists': result_list
                })
    
    print(f"\nCombination validation completed, valid combination count for each circuit:")
    total_combinations = 0
    for circuit_key, combinations in all_result_lists.items():
        print(f"  {circuit_key}: {len(combinations)} valid combinations")
        total_combinations += len(combinations)
    print(f"  Total: {total_combinations} valid combinations")
    
    # Generate forget_all_node and retain_all_node
    print(f"\nStarting generation of forget_all_node and retain_all_node...")
    
    def merge_result_lists(result_lists_dict):
        """
        Merge result_lists from multiple circuits
        Args:
            result_lists_dict: {circuit_num: result_list} dictionary
        Returns:
            dict: Merged result_list
        """
        merged_result = {}
        all_nodes = set()
        
        # Collect all nodes
        for result_list in result_lists_dict.values():
            all_nodes.update(result_list.keys())
        
        # Merge each node
        for node in all_nodes:
            values = []
            for circuit_num, result_list in result_lists_dict.items():
                if node in result_list:
                    values.append(result_list[node])
                else:
                    values.append(None)
            
            # Check value validity
            non_none_values = [v for v in values if v is not None]
            
            if len(non_none_values) == 0:
                # All values are None, error
                raise ValueError(f"Node {node} has None values in all circuits")
            
            if len(non_none_values) < len(values):
                # Has None values and bool values, check if bool values are the same
                unique_values = set(non_none_values)
                if len(unique_values) > 1:
                    raise ValueError(f"Node {node} has inconsistent values across circuits: {values}")
                
                # All non-None values are the same, take this value
                merged_result[node] = non_none_values[0]
            else:
                # All values are not None, check if they are the same
                unique_values = set(values)
                if len(unique_values) > 1:
                    raise ValueError(f"Node {node} has inconsistent values across circuits: {values}")
                
                merged_result[node] = values[0]
        
        return merged_result
    
    # Generate forget_all_node (all combinations for circuit1)
    forget_all_node = []
    for combo in all_result_lists['circuit1']:
        forget_all_node.append(combo['result_lists'])
    
    print(f"  forget_all_node: {len(forget_all_node)} combinations")
    
    # Generate retain_all_node (merge combinations from other circuits)
    retain_all_node = []
    
    # Get combinations from other circuits
    other_circuit_combinations = {}
    for circuit_key, combinations in all_result_lists.items():
        if circuit_key != 'circuit1':
            circuit_num = int(circuit_key.replace('circuit', ''))
            other_circuit_combinations[circuit_num] = combinations
    
    if other_circuit_combinations:
        # Filter out circuits without valid combinations
        valid_circuit_combinations = {}
        for circuit_num, combinations in other_circuit_combinations.items():
            if len(combinations) > 0:
                valid_circuit_combinations[circuit_num] = combinations
        
        if valid_circuit_combinations:
            # Get all possible combination indices
            circuit_nums = sorted(valid_circuit_combinations.keys())
            combination_counts = [len(valid_circuit_combinations[circuit_num]) for circuit_num in circuit_nums]
            
            print(f"    Valid circuits: {circuit_nums}")
            print(f"    Combination counts per circuit: {combination_counts}")
            
            # Generate all possible combinations
            for combo_indices in itertools.product(*[range(count) for count in combination_counts]):
                try:
                    # Collect all result_lists for this combination
                    result_lists_dict = {}
                    for i, circuit_num in enumerate(circuit_nums):
                        combo_idx = combo_indices[i]
                        if combo_idx < len(valid_circuit_combinations[circuit_num]):
                            result_lists_dict[circuit_num] = valid_circuit_combinations[circuit_num][combo_idx]['result_lists']
                    
                    # Merge result_lists
                    merged_result = merge_result_lists(result_lists_dict)
                    retain_all_node.append(merged_result)
                    
                except ValueError as e:
                    print(f"    Skipping invalid combination {combo_indices}: {e}")
                    continue
        else:
            print("    Warning: No valid other circuit combinations found")
    else:
        print("    Warning: No other circuit combinations found")
    
    print(f"  retain_all_node: {len(retain_all_node)} combinations")
    
    # Check if there are enough combinations for comparison
    if len(forget_all_node) == 0:
        print("Error: forget_all_node is empty, cannot perform comparison")
        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, [], [], None, None, [], [], []
    
    if len(retain_all_node) == 0:
        print("Error: retain_all_node is empty, cannot perform comparison")
        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, [], None, None, [], [], []
    
    # Calculate Hamming distance
    print(f"\nStarting calculation of Hamming distance between forget_all_node and retain_all_node...")
    
    def calculate_hamming_distance(result_list1, result_list2):
        """
        Calculate Hamming distance between two result_lists
        Args:
            result_list1: First result_list (dictionary format)
            result_list2: Second result_list (dictionary format)
        Returns:
            int: Hamming distance
        """
        # Find nodes that exist in both result_lists
        nodes1 = set(result_list1.keys())
        nodes2 = set(result_list2.keys())
        common_nodes = nodes1.intersection(nodes2)
        
        hamming_distance = 0
        for node in common_nodes:
            # If values are different, add 1 point
            if result_list1[node] != result_list2[node]:
                hamming_distance += 1
        
        return hamming_distance
    
    # Calculate Hamming distance between all forget and retain combinations
    hamming_distances = {}
    best_forget_idx = None
    best_retain_idx = None
    min_distance = float('inf')
    
    for forget_idx, forget_result in enumerate(forget_all_node):
        for retain_idx, retain_result in enumerate(retain_all_node):
            distance = calculate_hamming_distance(forget_result, retain_result)
            hamming_distances[(forget_idx, retain_idx)] = distance
            
            if distance < min_distance:
                min_distance = distance
                best_forget_idx = forget_idx
                best_retain_idx = retain_idx
            
            print(f"  forget[{forget_idx}] vs retain[{retain_idx}]: Hamming distance = {distance}")
    
    print(f"\n=== Hamming Distance Analysis Results ===")
    print(f"Best combination: forget[{best_forget_idx}] vs retain[{best_retain_idx}]")
    print(f"Minimum Hamming distance: {min_distance}")
    
    # Check if best combination was found
    if best_forget_idx is None or best_retain_idx is None:
        print("Error: Could not find best combination")
        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, None, None, [], [], []
    
    # Analyze nodes of best combination
    print(f"\nStarting analysis of best combination nodes...")
    
    best_forget_result = forget_all_node[best_forget_idx]
    best_retain_result = retain_all_node[best_retain_idx]
    
    # Generate two node sets
    forget_nodes = set(best_forget_result.keys())
    retain_nodes = set(best_retain_result.keys())
    
    common_nodes = forget_nodes.intersection(retain_nodes)
    non_common_nodes = forget_nodes.symmetric_difference(retain_nodes)
    
    print(f"  Common nodes: {len(common_nodes)} nodes")
    print(f"  Non-common nodes: {len(non_common_nodes)} nodes")
    
    # Analyze common nodes
    conflict_node_list = []
    same_value_nodes = []
    
    for node in common_nodes:
        if best_forget_result[node] != best_retain_result[node]:
            conflict_node_list.append(node)
        else:
            same_value_nodes.append(node)
    
    # Add nodes with same values to non-common nodes set
    non_common_nodes.update(same_value_nodes)
    
    # Analyze non-common nodes
    false_node_list = []
    true_node_list = []
    
    for node in non_common_nodes:
        if node in best_forget_result:
            value = best_forget_result[node]
        else:
            value = best_retain_result[node]
        
        if value is False:
            false_node_list.append(node)
        elif value is True:
            true_node_list.append(node)
    
    # Ensure no duplicate nodes in the three lists
    conflict_set = set(conflict_node_list)
    false_set = set(false_node_list)
    true_set = set(true_node_list)
    
    # Check for duplicates
    all_nodes = conflict_set.union(false_set).union(true_set)
    if len(all_nodes) != len(conflict_node_list) + len(false_node_list) + len(true_node_list):
        print("  Warning: Duplicate nodes found in the three lists, deduplicating...")
        conflict_node_list = list(conflict_set)
        false_node_list = list(false_set)
        true_node_list = list(true_set)
    
    print(f"\n=== Node Analysis Results ===")
    print(f"conflict_node_list: {len(conflict_node_list)} nodes")
    if conflict_node_list:
        print(f"  Node list: {conflict_node_list}")
    
    print(f"false_node_list: {len(false_node_list)} nodes")
    if false_node_list:
        print(f"  Node list: {false_node_list}")
    
    print(f"true_node_list: {len(true_node_list)} nodes")
    if true_node_list:
        print(f"  Node list: {true_node_list}")
    
    # Save results to JSON files
    print(f"\nStarting to save results to JSON files...")
    
    def extract_circuit_name(circuit_path):
        """
        Extract circuit name from circuit path
        Args:
            circuit_path: Complete path of circuit file
        Returns:
            str: Extracted circuit name
        """
        # Split path
        parts = circuit_path.split('/')
        # Find part containing circuit name
        for part in parts:
            if '_edges' in part:
                # Extract part before underscore
                return part.split('_')[0]
        return "unknown"
    
    # Get names of all circuits
    circuit_names = []
    for i in range(1, 7):  # circuit1 to circuit4
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_name = extract_circuit_name(circuit_path)
                circuit_names.append(circuit_name)
    
    # Generate filename suffix
    suffix = "_".join(circuit_names)
    
    # Save directory
    save_dir = ""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save conflict_node_list
    conflict_file = os.path.join(save_dir, f"conflict_node_list_{suffix}.json")
    with open(conflict_file, 'w') as f:
        json.dump(conflict_node_list, f, indent=2)
    print(f"  Saved conflict_node_list to: {conflict_file}")
    
    # Save false_node_list
    false_file = os.path.join(save_dir, f"false_node_list_{suffix}.json")
    with open(false_file, 'w') as f:
        json.dump(false_node_list, f, indent=2)
    print(f"  Saved false_node_list to: {false_file}")
    
    # Save true_node_list
    true_file = os.path.join(save_dir, f"true_node_list_{suffix}.json")
    with open(true_file, 'w') as f:
        json.dump(true_node_list, f, indent=2)
    print(f"  Saved true_node_list to: {true_file}")
    
    print(f"All results saved to: {save_dir}")
    
    return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, best_forget_idx, best_retain_idx, conflict_node_list, false_node_list, true_node_list

if __name__ == "__main__":
    # Run main function
    circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, best_forget_idx, best_retain_idx, conflict_node_list, false_node_list, true_node_list = main()
    
    # Print leaf node investigation results
    print(f"\n=== Leaf Node Investigation Results ===")
    for circuit_num, leaf_nodes in leaf_nodes_by_circuit.items():
        print(f"Circuit{circuit_num}: {len(leaf_nodes)} leaf nodes")
        if leaf_nodes:
            print(f"  Leaf nodes: {leaf_nodes}")
        print()
    
    # Print input node information
    print(f"\n=== Input Node Information ===")
    print(f"input_retain: {len(input_retain)} nodes")
    if input_retain:
        print(f"  Node list: {input_retain}")
    print(f"input_forget: {len(input_forget)} nodes")
    if input_forget:
        print(f"  Node list: {input_forget}")
    print()
    
    # Print combination validation results
    print(f"\n=== Combination Validation Results ===")
    total_combinations = sum(len(combinations) for combinations in all_result_lists.values())
    print(f"Found {total_combinations} valid combinations")
    
    for circuit_key, combinations in all_result_lists.items():
        if combinations:
            print(f"\n{circuit_key}: {len(combinations)} valid combinations")
            for i, result_info in enumerate(combinations):
                print(f"  Combination {i+1}:")
                print(f"    Type: {result_info['type']}")
                print(f"    Node combination: {result_info['combination']}")
                print(f"    Corresponding nodes: {result_info['nodes']}")
                print(f"    result_lists length: {len(result_info['result_lists'])}")
                print(f"    result_lists first 10 items: {list(result_info['result_lists'].items())[:10]}")
    print()
    
    # Print Hamming distance analysis results
    print(f"\n=== Hamming Distance Analysis Results ===")
    if best_forget_idx is not None and best_retain_idx is not None:
        print(f"Best combination: forget[{best_forget_idx}] vs retain[{best_retain_idx}]")
    else:
        print("Could not find best combination")
    
    print(f"\nforget_all_node: {len(forget_all_node)} combinations")
    print(f"retain_all_node: {len(retain_all_node)} combinations")
    
    print(f"\n=== Node Analysis Results ===")
    print(f"conflict_node_list: {len(conflict_node_list)} nodes")
    if conflict_node_list:
        print(f"  Node list: {conflict_node_list}")
    
    print(f"false_node_list: {len(false_node_list)} nodes")
    if false_node_list:
        print(f"  Node list: {false_node_list}")
    
    print(f"true_node_list: {len(true_node_list)} nodes")
    if true_node_list:
        print(f"  Node list: {true_node_list}")
    print()

