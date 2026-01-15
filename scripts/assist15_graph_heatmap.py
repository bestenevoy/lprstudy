import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[1]
    graph_path = project_root / "data" / "dataProcess" / "ASSISTments2015" / "nxgraph.pkl"
    output_path = project_root / "data" / "dataProcess" / "ASSISTments2015" / "graph_heatmap.png"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_path}")

    with graph_path.open("rb") as f:
        graph = pickle.load(f)

    nodes = sorted(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    adj = np.zeros((size, size), dtype=np.float32)
    for src, dst in graph.edges():
        adj[index[src], index[dst]] = 1.0

    plt.figure(figsize=(8, 7))
    plt.imshow(adj, cmap="viridis", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Edge")
    plt.title("KES-ASSIST15 Knowledge Graph (Adjacency Heatmap)")
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
