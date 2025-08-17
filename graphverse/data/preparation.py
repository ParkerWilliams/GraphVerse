import torch
import numpy as np
import random
from tqdm import tqdm

from ..graph.walk import generate_multiple_walks, generate_valid_walk
from ..analysis.metadata import TrainingCorpusMetadata


class WalkVocabulary:
    """
    Vocabulary for walks.
    """

    def __init__(self, walks):
        max_node = np.max(np.concatenate(walks))
        nodes = range(0, max_node + 1)
        self.token2idx = {
            **{str(x): x for x in nodes},
            **{"<PAD>": max_node + 1, "<START>": max_node + 2, "<END>": max_node + 3}
        }
        self.idx2token = {
            **{x: str(x) for x in nodes},
            **{max_node + 1: "<PAD>", max_node + 2: "<START>", max_node + 3: "<END>"}
        }
        # self. = {}
        # self.build_vocab(walks)

    # def build_vocab(self, walks):
    #     for walk in walks:
    #         for token in walk:
    #             if str(token) not in self.token2idx:
    #                 idx = len(self.token2idx) - 3
    #                 self.token2idx[str(token)] = idx
    #                 self.idx2token[idx] = str(token)

    def __len__(self):
        return len(self.token2idx)


def prepare_training_data(
    graph, num_walks, min_length, max_length, rules, verbose=False
):
    """
    Prepare training data for the model.
    
    Args:
        graph: The graph to generate walks on
        num_walks: Number of random walks to generate
        min_length: Minimum walk length
        max_length: Maximum walk length
        rules: Set of rules to follow
        verbose: Whether to print progress
        
    Returns:
        training_data: Tensor of shape (N, max_seq_len) containing input sequences
        vocab: WalkVocabulary object mapping node indices to tokens
    """
    # Generate walks
    walks = generate_multiple_walks(
        graph, num_walks, min_length, max_length, rules, verbose=verbose
    )
    
    if verbose:
        print(f"\n  Generating a walk starting from each node ({graph.n} nodes)...")
    
    per_node_walks = []
    node_iterator = tqdm(range(graph.n), desc="Per-node walks", unit="node") if verbose else range(graph.n)
    
    for node in node_iterator:
        walk = generate_valid_walk(
            graph, node, min_length, max_length, rules, verbose=False
        )
        if walk:
            per_node_walks.append(walk)
            if verbose:
                node_iterator.set_postfix({"walks": len(per_node_walks), "success_rate": f"{len(per_node_walks)/(node+1):.1%}"})
    
    # Combine all walks
    all_walks = walks + per_node_walks
    
    if verbose:
        print(f"\n  Creating vocabulary and preparing training sequences...")
    
    # Create vocabulary using WalkVocabulary class
    vocab = WalkVocabulary(all_walks)
    
    # Create training data
    max_seq_len = max(len(walk) for walk in all_walks)
    training_sequences = []
    
    if verbose:
        print(f"    Maximum sequence length: {max_seq_len}")
        print(f"    Processing {len(all_walks)} total walks...")
    
    walk_iterator = tqdm(all_walks, desc="Processing walks", unit="walk") if verbose else all_walks
    
    for walk in walk_iterator:
        # Convert walk to indices, including START and END tokens
        walk_indices = [vocab.token2idx["<START>"]] + [vocab.token2idx[str(node)] for node in walk] + [vocab.token2idx["<END>"]]
        # Pad sequence to max_seq_len + 2 (+2 for START and END tokens)
        padded = walk_indices + [vocab.token2idx["<PAD>"]] * (max_seq_len + 2 - len(walk_indices))
        training_sequences.append(padded)
    
    # Convert to tensor
    training_data = torch.tensor(training_sequences, dtype=torch.long)
    
    if verbose:
        print(f"  ✓ Data preparation complete")
    
    # Create comprehensive corpus metadata
    if verbose:
        print(f"  Generating training corpus metadata...")
    
    corpus_metadata = TrainingCorpusMetadata(all_walks, rules, vocab)
    
    if verbose:
        summary = corpus_metadata.get_summary()
        print(f"  Corpus metadata summary:")
        print(f"    Total walks: {summary['basic_stats']['total_walks']}")
        print(f"    Unique sequences: {summary['basic_stats']['unique_sequences']}")
        print(f"    Sequence diversity: {summary['basic_stats']['sequence_diversity']:.3f}")
        if 'rule_exposure' in summary:
            print(f"    Rule exposure:")
            for rule_type, percent in summary['rule_exposure']['exposure_percentages'].items():
                print(f"      {rule_type}: {percent:.1f}%")
    
    return training_data, vocab, corpus_metadata

def prepare_density_controlled_training_data(
    graph, 
    num_walks, 
    min_length, 
    max_length, 
    rules, 
    repeater_densities=None,
    verbose=False
):
    """
    Prepare training data with controlled repeater node density.
    
    Args:
        graph: The graph to generate walks on
        num_walks: Base number of random walks to generate
        min_length: Minimum walk length
        max_length: Maximum walk length
        rules: Set of rules to follow
        repeater_densities: Dict mapping repeater_node -> additional_walk_count
                           Controls how many extra walks include each repeater node
        verbose: Whether to print progress
        
    Returns:
        training_data: Tensor of shape (N, max_seq_len) containing input sequences
        vocab: WalkVocabulary object mapping node indices to tokens
        density_stats: Dictionary with statistics about repeater exposure
    """
    from ..graph.walk import generate_multiple_walks, generate_valid_walk
    
    # Generate base walks
    if verbose:
        print(f"Generating {num_walks} base random walks...")
    base_walks = generate_multiple_walks(
        graph, num_walks, min_length, max_length, rules, verbose=verbose
    )
    
    # Generate per-node walks (one from each node)
    if verbose:
        print(f"Generating walks starting from each node...")
    per_node_walks = []
    for node in range(graph.n):
        walk = generate_valid_walk(
            graph, node, min_length, max_length, rules, verbose=verbose
        )
        if walk:
            per_node_walks.append(walk)
    
    # Generate additional walks for specific repeater nodes if requested
    density_walks = []
    repeater_exposure_counts = {}
    
    if repeater_densities:
        # Find repeater nodes from rules
        repeater_nodes = set()
        for rule in rules:
            if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                repeater_nodes.update(rule.member_nodes)
        
        if verbose:
            print(f"Generating additional walks for repeater density control...")
            print(f"Repeater nodes: {repeater_nodes}")
            print(f"Density targets: {repeater_densities}")
        
        for repeater_node, additional_count in repeater_densities.items():
            if repeater_node in repeater_nodes:
                node_walks = []
                attempts = 0
                max_attempts = additional_count * 10  # Allow more attempts
                
                while len(node_walks) < additional_count and attempts < max_attempts:
                    # Try starting from the repeater node
                    walk = generate_valid_walk(
                        graph, repeater_node, min_length, max_length, rules, verbose=False
                    )
                    if walk and repeater_node in walk:
                        node_walks.append(walk)
                    
                    # Also try random starts that might pass through the repeater
                    if len(node_walks) < additional_count:
                        start_node = random.choice(list(range(graph.n)))
                        walk = generate_valid_walk(
                            graph, start_node, min_length, max_length, rules, verbose=False
                        )
                        if walk and repeater_node in walk:
                            node_walks.append(walk)
                    
                    attempts += 1
                
                density_walks.extend(node_walks)
                if verbose:
                    print(f"Generated {len(node_walks)} additional walks for repeater {repeater_node}")
    
    # Combine all walks
    all_walks = base_walks + per_node_walks + density_walks
    
    # Calculate exposure statistics
    if repeater_densities:
        for repeater_node in repeater_densities.keys():
            count = sum(1 for walk in all_walks if repeater_node in walk)
            repeater_exposure_counts[repeater_node] = count
    
    # Create vocabulary
    vocab = WalkVocabulary(all_walks)
    
    # Create training data
    max_seq_len = max(len(walk) for walk in all_walks)
    training_sequences = []
    
    for walk in all_walks:
        # Convert walk to indices, including START and END tokens
        walk_indices = [vocab.token2idx["<START>"]] + [vocab.token2idx[str(node)] for node in walk] + [vocab.token2idx["<END>"]]
        # Pad sequence to max_seq_len + 2 (+2 for START and END tokens)
        padded = walk_indices + [vocab.token2idx["<PAD>"]] * (max_seq_len + 2 - len(walk_indices))
        training_sequences.append(padded)
    
    # Convert to tensor
    training_data = torch.tensor(training_sequences, dtype=torch.long)
    
    # Prepare density statistics
    density_stats = {
        "total_walks": len(all_walks),
        "base_walks": len(base_walks),
        "per_node_walks": len(per_node_walks),
        "density_walks": len(density_walks),
        "repeater_exposure_counts": repeater_exposure_counts,
        "repeater_densities_requested": repeater_densities or {}
    }
    
    if verbose:
        print(f"Training data prepared:")
        print(f"  Total walks: {len(all_walks)}")
        print(f"  Repeater exposures: {repeater_exposure_counts}")
    
    return training_data, vocab, density_stats


def analyze_rule_exposure_in_corpus_legacy(walks, rules, verbose=False):
    """
    Analyze what portion of the training corpus includes each rule type.
    
    Args:
        walks: List of walks in the training corpus
        rules: List of rule instances
        verbose: Whether to print detailed statistics
        
    Returns:
        dict: Statistics about rule exposure in the corpus
    """
    total_walks = len(walks)
    if total_walks == 0:
        return {}
    
    # Initialize counters
    rule_exposure = {
        'ascender_walks': 0,
        'even_walks': 0, 
        'repeater_walks': 0,
        'no_rule_walks': 0,
        'multiple_rule_walks': 0,
        'total_walks': total_walks,
        'rule_node_frequencies': {},
        'rule_type_details': {}
    }
    
    # Find rule instances
    ascender_rule = None
    even_rule = None
    repeater_rule = None
    
    for rule in rules:
        if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
            ascender_rule = rule
        elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
            even_rule = rule
        elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            repeater_rule = rule
    
    # Collect all rule nodes
    all_rule_nodes = set()
    if ascender_rule:
        all_rule_nodes.update(ascender_rule.member_nodes)
        rule_exposure['rule_type_details']['ascender'] = {
            'nodes': list(ascender_rule.member_nodes),
            'count': len(ascender_rule.member_nodes),
            'walk_exposure': 0
        }
    
    if even_rule:
        all_rule_nodes.update(even_rule.member_nodes)
        rule_exposure['rule_type_details']['even'] = {
            'nodes': list(even_rule.member_nodes),
            'count': len(even_rule.member_nodes),
            'walk_exposure': 0
        }
    
    if repeater_rule:
        all_rule_nodes.update(repeater_rule.member_nodes)
        rule_exposure['rule_type_details']['repeater'] = {
            'nodes': list(repeater_rule.member_nodes),
            'k_values': dict(repeater_rule.members_nodes_dict),
            'count': len(repeater_rule.member_nodes),
            'walk_exposure': 0
        }
    
    # Initialize node frequency tracking
    for node in all_rule_nodes:
        rule_exposure['rule_node_frequencies'][node] = 0
    
    if verbose:
        print(f"\n  Analyzing rule exposure in {total_walks} walks...")
        print(f"    Total rule nodes: {len(all_rule_nodes)}")
        if ascender_rule:
            print(f"    Ascender nodes: {len(ascender_rule.member_nodes)}")
        if even_rule:
            print(f"    Even rule nodes: {len(even_rule.member_nodes)}")
        if repeater_rule:
            print(f"    Repeater nodes: {len(repeater_rule.member_nodes)}")
    
    # Analyze each walk
    for walk in walks:
        walk_rule_types = set()
        walk_has_rule = False
        
        # Check which rule nodes appear in this walk
        for node in walk:
            if node in all_rule_nodes:
                walk_has_rule = True
                rule_exposure['rule_node_frequencies'][node] += 1
                
                # Determine rule type
                if ascender_rule and node in ascender_rule.member_nodes:
                    walk_rule_types.add('ascender')
                if even_rule and node in even_rule.member_nodes:
                    walk_rule_types.add('even')
                if repeater_rule and node in repeater_rule.member_nodes:
                    walk_rule_types.add('repeater')
        
        # Count walk types
        if not walk_has_rule:
            rule_exposure['no_rule_walks'] += 1
        else:
            if 'ascender' in walk_rule_types:
                rule_exposure['ascender_walks'] += 1
                rule_exposure['rule_type_details']['ascender']['walk_exposure'] += 1
            if 'even' in walk_rule_types:
                rule_exposure['even_walks'] += 1
                rule_exposure['rule_type_details']['even']['walk_exposure'] += 1
            if 'repeater' in walk_rule_types:
                rule_exposure['repeater_walks'] += 1
                rule_exposure['rule_type_details']['repeater']['walk_exposure'] += 1
            
            if len(walk_rule_types) > 1:
                rule_exposure['multiple_rule_walks'] += 1
    
    # Calculate percentages
    rule_exposure['percentages'] = {
        'ascender_walks': (rule_exposure['ascender_walks'] / total_walks) * 100,
        'even_walks': (rule_exposure['even_walks'] / total_walks) * 100,
        'repeater_walks': (rule_exposure['repeater_walks'] / total_walks) * 100,
        'no_rule_walks': (rule_exposure['no_rule_walks'] / total_walks) * 100,
        'multiple_rule_walks': (rule_exposure['multiple_rule_walks'] / total_walks) * 100,
        'any_rule_walks': ((total_walks - rule_exposure['no_rule_walks']) / total_walks) * 100
    }
    
    if verbose:
        print(f"\n  Rule exposure analysis:")
        print(f"    Walks with ascender nodes: {rule_exposure['ascender_walks']} ({rule_exposure['percentages']['ascender_walks']:.1f}%)")
        print(f"    Walks with even rule nodes: {rule_exposure['even_walks']} ({rule_exposure['percentages']['even_walks']:.1f}%)")
        print(f"    Walks with repeater nodes: {rule_exposure['repeater_walks']} ({rule_exposure['percentages']['repeater_walks']:.1f}%)")
        print(f"    Walks with no rule nodes: {rule_exposure['no_rule_walks']} ({rule_exposure['percentages']['no_rule_walks']:.1f}%)")
        print(f"    Walks with multiple rule types: {rule_exposure['multiple_rule_walks']} ({rule_exposure['percentages']['multiple_rule_walks']:.1f}%)")
        print(f"    Total walks with any rules: {total_walks - rule_exposure['no_rule_walks']} ({rule_exposure['percentages']['any_rule_walks']:.1f}%)")
        
        # Show most/least frequent rule nodes
        if rule_exposure['rule_node_frequencies']:
            sorted_frequencies = sorted(rule_exposure['rule_node_frequencies'].items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Most frequent rule nodes:")
            for node, freq in sorted_frequencies[:5]:
                percent = (freq / total_walks) * 100
                print(f"    Node {node}: {freq} walks ({percent:.1f}%)")
    
    return rule_exposure
