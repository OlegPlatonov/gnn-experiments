import torch
import orcastr


def compute_graphlet_degree_vectors(graph, max_graphlet_size=5):
    if max_graphlet_size not in [4, 5]:
        raise ValueError('max_graphlet_size should be either 4 or 5.')

    source_nodes, target_nodes = graph.edges()
    source_nodes = source_nodes.cpu().numpy()
    target_nodes = target_nodes.cpu().numpy()

    edges = set()
    for u, v in zip(source_nodes, target_nodes):
        if u == v:
            continue
        if u > v:
            u, v = v, u

        edges.add((u, v))

    n = len(graph.nodes())
    m = len(edges)
    lines = [f'{n} {m}\n']
    for u, v in edges:
        lines.append(f'{u} {v}\n')

    orca_input_string = ''.join(lines)
    orca_output_string = orcastr.motif_counts_str('node', max_graphlet_size, orca_input_string)
    graphlet_degree_vectors = [[int(num) for num in line.split()] for line in orca_output_string.splitlines()]
    graphlet_degree_vectors = torch.tensor(graphlet_degree_vectors)

    return graphlet_degree_vectors


def transform_graphlet_degree_vectors_to_binary_features(graphlet_degree_vectors):
    bounds = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400,
        500, 750, 1000, 1250, 1500, 2000, 2500, 3500, 5000, 7500, 10000, 15000, 20000, 25000, 32000, 40000, 50000,
        70000, 100000, 150000, 200000, 250000
    ]

    graphlet_features = []
    for i in range(graphlet_degree_vectors.shape[1]):
        counts = graphlet_degree_vectors[:, i]
        for j in range(len(bounds) - 1):
            cur_graphlet_features = (counts >= bounds[j]) & (counts < bounds[j + 1])
            graphlet_features.append(cur_graphlet_features)

        cur_graphlet_features = (counts >= bounds[-1])
        graphlet_features.append(cur_graphlet_features)

    graphlet_features = torch.stack(graphlet_features).T.float()

    return graphlet_features
