import torch
import networkx as nx
from torch_geometric.data import Data

def is_aig(G_raw) -> bool:

    allowed_gates = {"INPUT", "AND"}
    for _, node_data in G_raw.nodes(data=True):
        gate_type = node_data.get("gate", "INPUT")
        if gate_type not in allowed_gates:
            return False
    return True

def compute_node_levels(G_raw) -> dict:
    node_levels = {}
    for level, generation in enumerate(nx.topological_generations(G_raw)):
        for node in generation: 
            node_levels[node] = level
    return node_levels

def load_bench(path):
    G_raw = load_bench(path)

    if not is_aig(G_raw):
        print(f"[REJECTED] {path}: The gate provided has been detected to be of a Non-AIG gate format. The file logged is pending for ABC conversion.")
        return None

    node_levels = compute_node_levels(G_raw)
    
    nodes_list = list(G_raw.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}

    GATE_TYPES = ['INPUT', 'AND', 'NAND', 'OR', 'NOR']

    x = []

    for node in nodes_list:
        node_data = G_raw.nodes[node]

        fan_in = G_raw.in_degree(node)
        fan_out = G_raw.out_degree(node)
        level = node_levels[node]

        g_type = node_data.get('gate', 'INPUT')
        type_vec = [0.0] * 5 
        if g_type in GATE_TYPES:
            type_vec[GATE_TYPES.index(g_type)] =  1.0

        node_feature = type_vec + [float(fan_in), float(fan_out), float(level)]
        x.append(node_feature)

    edge_index_src = []
    edge_index_dst = []
    edge_type = []

    for u, v, edge_data in G_raw.edges(data=True):
        edge_index_src.append(node_to_idx[u])
        edge_index_dst.append(node_to_idx[v])

        is_inverted = edge_data.get("inverted", False)
        edge_type.append(1 if is_inverted else 0)

    x_tensor = torch.tensor(x, dtype = torch.float)
    edge_index_tensor = torch.tensor([edge_index_src, edge_index_dst], dtype = torch.long)
    edge_type_tensor = torch.tensor(edge_type, dtype = torch.long)

    data = Data(x=x_tensor, edge_index = edge_index_tensor, edge_type = edge_type_tensor)

    return data