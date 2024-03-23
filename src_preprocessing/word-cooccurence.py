import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer


def load_comments_from_json(filename):
    """
    Loads comments from a JSON file.

    Args:
        filename: The name of the JSON file.

    Returns:
        A list of comments, where each comment is represented as a string.
    """
    with open(filename, 'r') as json_file:
        comments = json.load(json_file)
    return comments


def create_co_occurrence_matrix(comments, top_n=40):
    """
    Creates a co-occurrence matrix from a list of comments.

    Args:
        comments: A list of comments, where each comment is represented as a string.
        top_n: Number of top words to consider.

    Returns:
        A co-occurrence matrix.
    """
    # Convert comments to a list of strings
    comments_strings = [' '.join(comment) for comment in comments]

    # Use CountVectorizer to create a term-document matrix
    vectorizer = CountVectorizer(max_features=top_n)
    X = vectorizer.fit_transform(comments_strings)

    # Compute the co-occurrence matrix
    co_occurrence_matrix = X.T.dot(X)

    # Convert co-occurrence matrix to dense array
    co_occurrence_matrix_dense = co_occurrence_matrix.toarray()

    # Set diagonal elements to zero to avoid self-co-occurrence
    np.fill_diagonal(co_occurrence_matrix_dense, 0)

    return co_occurrence_matrix_dense, vectorizer.get_feature_names_out()


def visualize_co_occurrence_matrix(co_occurrence_matrix, words):
    """
    Visualizes a co-occurrence matrix.

    Args:
        co_occurrence_matrix: A co-occurrence matrix.
        words: A list of words.
    """
    df = pd.DataFrame(co_occurrence_matrix, index=words, columns=words)

    plt.figure(figsize=(12, 8))
    plt.matshow(df, fignum=1, cmap='Blues')
    plt.title('Co-occurrence Matrix')
    plt.xticks(range(len(words)), words, rotation=90)
    plt.yticks(range(len(words)), words)
    plt.colorbar()
    plt.savefig("../images/co_occurrence_matrix.png")
    plt.show()


def visualize_co_occurrence_graph(co_occurrence_graph, threshold):
    """
    Visualizes a co-occurrence graph with differentiated node size and edge thickness,
    only displaying edges with weights above a specified threshold.

    Args:
        co_occurrence_graph: A co-occurrence graph.
        threshold: The threshold value for edge weights.
    """
    # Filter edges based on the threshold
    edges_to_display = [(u, v) for u, v, d in co_occurrence_graph.edges(data=True) if d['weight'] > threshold]

    # Compute node sizes based on node degrees
    node_degrees = dict(co_occurrence_graph.degree())
    max_degree = max(node_degrees.values())
    min_degree = min(node_degrees.values())
    node_sizes = [100 * (node_degrees[node] - min_degree) / (max_degree - min_degree) for node in co_occurrence_graph.nodes()]

    # Compute edge thicknesses based on edge weights
    edge_weights = [co_occurrence_graph[u][v]['weight'] for u, v in edges_to_display]
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    edge_thicknesses = [0.5 + (weight - min_weight) / (max_weight - min_weight) * 2 for weight in edge_weights]

    plt.figure(figsize=(12, 8))
    # Use spring layout algorithm to position nodes
    pos = nx.spring_layout(co_occurrence_graph, k=10)
    # pos = nx.kamada_kawai_layout(co_occurrence_graph)

    # Draw edges with varying thickness
    nx.draw_networkx_edges(co_occurrence_graph, pos, edgelist=edges_to_display, width=edge_thicknesses, alpha=0.7)

    # Draw nodes with varying size
    nx.draw_networkx_nodes(co_occurrence_graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)

    # Draw node labels
    nx.draw_networkx_labels(co_occurrence_graph, pos, font_size=12)

    plt.title('Co-occurrence Graph')
    plt.tight_layout()
    plt.savefig("../images/co_occurrence_graph.png")
    plt.show()




def main():
    # Load comments from JSON
    comments = load_comments_from_json("../preprocessed_data/tokenized_comments.json")

    # Create co-occurrence matrix
    co_occurrence_matrix, words = create_co_occurrence_matrix(comments)

    # Create co-occurrence graph
    co_occurrence_graph = nx.Graph()

    # Add nodes to the graph
    for word in words:
        co_occurrence_graph.add_node(word)

    # Add weighted edges to the graph
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i < j and co_occurrence_matrix[i, j] > 0:
                co_occurrence_graph.add_edge(word1, word2, weight=co_occurrence_matrix[i, j])

    # Visualize co-occurrence graph
    visualize_co_occurrence_graph(co_occurrence_graph, threshold=50)

    # Visualize co-occurrence matrix
    visualize_co_occurrence_matrix(co_occurrence_matrix, words)


if __name__ == "__main__":
    main()
