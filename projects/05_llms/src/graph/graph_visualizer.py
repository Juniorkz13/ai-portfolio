import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(graph, output_path="graph.png"):
    """
    Save a visualization of the knowledge graph to a PNG file.
    """
    G = graph.graph

    if G.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.")
        return

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000,
        node_color="#AED6F1",
        edgecolors="black",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=20,
        edge_color="#555555",
    )

    edge_labels = {
        (u, v): d["relation"]
        for u, v, d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="red",
        font_size=9,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Graph saved to {output_path}")
