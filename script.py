data = bench_to_pyg("c17.bench")

if data is not None:
    model = CircuitRGCN(
        in_channels=8, 
        hidden_channels=32, 
        out_channels=64, 
        num_layers=3
    )
    model.eval()

    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index, data.edge_type)

    print("Input Node Feature Shape: ", data.x.shape)   
    print("Output Node Embedding Shape:", node_embeddings.shape) 