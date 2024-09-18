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
G = generate_random_graph(n, rules, num_walks, min_walk_length, max_walk_length)

print(f'Graph created')

# Print some information about the graph
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
print(f"Is weakly connected: {nx.is_weakly_connected(G)}")

print(f'Now preparing training data')

# Prepare training data
training_data, vocab = prepare_training_data(G, num_samples=num_walks, min_length=min_walk_length, max_length=max_walk_length, rules=rules)
print(f'Training data prepared')
print(f'Vocab size: {len(vocab)}')
print(f'Training data shape: {training_data.shape}')
print(f'Begin training')

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train_model(training_data, vocab, epochs=1, batch_size=32, learning_rate=0.001, device=device)

# Evaluate the model
max_corpus_length = training_data.size(1)  # Get the maximum sequence length
evaluation_results = evaluate_model(model, G, vocab, num_samples=10000,
                                    min_start_length=1, max_start_length=int(0.1 * max_corpus_length), rules=rules)

# Analyze results
df_results = pd.DataFrame(evaluation_results)

# Calculate average rule violations per rule type
rule_violations_per_type = df_results['rule_violations'].apply(pd.Series).stack().apply(pd.Series)['rule_type'].value_counts()
total_violations = rule_violations_per_type.sum()
avg_violations_per_type = rule_violations_per_type / total_violations

print("\nEvaluation Results:")
print(f"Total rule violations: {total_violations}")
print("Average rule violations per rule type:")
for rule_type, avg_violations in avg_violations_per_type.items():
    print(f"{rule_type}: {avg_violations:.2%}")

# Calculate average walk length and violation position
df_violations = pd.json_normalize(df_results['rule_violations'].explode())
avg_walk_length = df_violations['walk_length'].mean()
avg_violation_position = df_violations['violation_position'].mean()

print(f"\nAverage walk length: {avg_walk_length:.2f}")
print(f"Average violation position: {avg_violation_position:.2f}")

# Analyze rule violations by walk length
walk_length_bins = [0, 10, 20, 30, 40, 50, math.inf]
walk_length_labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '50+']
df_violations['walk_length_bin'] = pd.cut(df_violations['walk_length'], bins=walk_length_bins, labels=walk_length_labels)
violations_by_walk_length = df_violations.groupby('walk_length_bin').size()

print("\nRule violations by walk length:")
for walk_length_bin, count in violations_by_walk_length.items():
    print(f"{walk_length_bin}: {count}")

# Analyze rule violations by violation position
position_bins = [0, 10, 20, 30, 40, 50, math.inf]
position_labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '50+']
df_violations['position_bin'] = pd.cut(df_violations['violation_position'], bins=position_bins, labels=position_labels)
violations_by_position = df_violations.groupby('position_bin').size()

print("\nRule violations by violation position:")
for position_bin, count in violations_by_position.items():
    print(f"{position_bin}: {count}")

# Optional: Save results to a CSV file
df_results.to_csv('evaluation_results.csv', index=False)
print("\nDetailed results saved to 'evaluation_results.csv'")
