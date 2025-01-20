import matplotlib.pyplot as plt
import networkx as nx

# Draw the graph


def visualize_graph(G):
    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Generate a layout for the nodes
    pos = nx.spring_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")

    # Draw the edges with varying thickness based on probability
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=data["probability"] * 5,
            alpha=0.7,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
        )

    # Draw the node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # Draw the edge labels (probabilities)
    edge_labels = nx.get_edge_attributes(G, "probability")
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Remove axis
    plt.axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()
    return
