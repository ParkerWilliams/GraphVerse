import itertools as it

import matplotlib.pyplot as plt
import networkx as nx


def _get_n_colors_from_cmap(n_samples, cmap_name="viridis", vmin=None, vmax=None):
    """
    A stupid little wrapper to help me remember how to pull samples off a cmap
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in np.linspace(vmin or 0, vmax or 1, n_samples)]


def visualize_graph(G, walk=None):
    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Generate a layout for the nodes
    # pos = nx.spring_layout(G)
    pos = nx.shell_layout(G)

    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")

    # Draw the edges with varying thickness based on probability
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=1,  # data["probability"] * 5,
            alpha=0.7,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
        )

    # Draw the node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # Draw the edge labels (probabilities)
    # edge_labels = nx.get_edge_attributes(G, "probability")
    # edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    if walk is not None:
        edge_list = [(walk[i], walk[i + 1]) for i in range(len(walk) - 1)]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_list,
            edge_color=_get_n_colors_from_cmap(
                len(walk), cmap_name="magma", vmin=0.3, vmax=0.8
            ),
            width=4,
            arrows=True,
            arrowsize=20,
            # connectionstyle=connectionstyle
        )

    # Remove axis
    plt.axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()
    return
