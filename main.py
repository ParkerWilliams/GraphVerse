import networkx as nx
import matplotlib.pyplot as plt
import torch
import pandas as pd
import math
import random

from graphverse.graph.graph_generation import generate_random_graph, calculate_edge_density
from graphverse.graph.rules import AscenderRule, DescenderRule, EvenRule, OddRule, RepeaterRule
from graphverse.graph.rules import define_all_rules
from graphverse.data.preparation import prepare_training_data
from graphverse.llm.training import train_model
from graphverse.llm.evaluation import evaluate_model

# Define graph parameters
n = 1000  # Number of vertices
num_walks = 100_000  # Number of walks to generate
min_walk_length = 10  # Minimum walk length
max_walk_length = 50  # Maximum walk length

# Define rule sets
print('Selecting vertices with rules')
ascenders, descenders, evens, odds, repeaters = define_all_rules(n, 3, 10, 100)

# Create rule tuple
rules = (
    AscenderRule(ascenders),
    DescenderRule(descenders),
    EvenRule(evens),
    OddRule(odds),
    RepeaterRule(repeaters)
)

# Generate graph
print('Generating graph')
G = generate_random_graph(n, rules, num_walks, min_walk_length, max_walk_length, verbose=True)

print(f'Graph created')

# Print some information about the graph
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
print(f"Is weakly connected: {nx.is_weakly_connected(G)}")

print(f'Now preparing training data')

# Prepare training data
training_data, vocab = prepare_training_data(G, num_samples=num_walks, min_length=min_walk_length, max_length=max_walk_length, rules=rules, verbose=True)
print(f'Training data prepared')
print(f'Vocab size: {len(vocab)}')
print(f'Training data shape: {training_data.shape}')
print(f'Begin training')

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train_model(training_data, vocab, epochs=1, batch_size=32, learning_rate=0.001, device=device, verbose=True)

# Evaluate the model
max_corpus_length = training_data.size(1)  # Get the maximum sequence length
evaluation_results = evaluate_model(model, G, vocab, num_samples=10000,
                                    min_start_length=1, max_start_length=int(0.1 * max_corpus_length), rules=rules, verbose=True)

# Analyze results
df_results = pd.DataFrame(evaluation_results)

# Calculate average rule violations per rule type
rule_violations_per_type = df_results['rule_violations'].apply(pd.Series).stack().apply(pd.Series)['rule_type'].value_counts()
total_violations = rule_violations_per_type.sum()
avg_violations_per_type = rule_violations_per_type / total_violations
