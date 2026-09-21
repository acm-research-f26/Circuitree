import networkx as nx  # Importing the NetworkX library


def parse_bench_file(file_path):
    # Initialize a directed graph since circuit signals flow from inputs to gates/outputs
    circuit_graph = nx.DiGraph()

    gate_counts = {}  # Dictionary to keep track of gate types (e.g., NAND)

    # Open and read the .bench file line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip empty lines or comment lines starting with '#'
            if not line or line.startswith('#'):
                continue

            # Process gate lines, for example: 10 = NAND(1, 3)
            if '=' in line:
                # Split into output node (left side) and the gate operation (right side)
                left_side, right_side = line.split('=')
                output_node = left_side.strip()

                # Extract gate type (e.g., NAND) and inputs (e.g., 1, 3)
                gate_type = right_side.split('(')[0].strip()
                inputs_str = right_side.split('(')[1].rstrip(')')
                input_nodes = [inp.strip() for inp in inputs_str.split(',')]

                # Add output node to the networkx graph
                circuit_graph.add_node(output_node)

                # Add edges from each input node into the output node (gate)
                for input_node in input_nodes:
                    circuit_graph.add_edge(input_node, output_node)

                # Count the occurrences of each gate type
                gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1

    return circuit_graph, gate_counts


if __name__ == "__main__":
    # Path to your c17.bench file in the project folder
    bench_file_path = "c17.bench"

    graph, gates = parse_bench_file(bench_file_path)

    # Print out results based on the task requirements (Target: 11 nodes, 12 edges, 6 NANDs)
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print("Gate counts:")
    for gate, count in gates.items():
        print(f"  - {gate}: {count}")