import random

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ..graph.walk import check_rule_compliance, generate_valid_walk
from ..analysis.metadata import WalkTrajectoryMetadata, EvaluationTrajectoryMetadata


def evaluate_model(
    model,
    graph,
    vocab,
    num_walks,
    min_start_length,
    max_start_length,
    rules,
    verbose=False,
    track_token_details=True,
    trajectory_sampling_config=None,  # NEW: Configuration for trajectory sampling
    config=None,  # NEW: General configuration for evaluation settings
    fast_mode=False,  # NEW: Skip expensive distribution computations
):
    model.eval()
    
    # NEW: Set up trajectory sampling configuration FIRST
    if trajectory_sampling_config is None:
        trajectory_sampling_config = {
            "enabled": False,
            "sample_rate": 1.0,  # Store all trajectories by default
            "stratified": False,
            "min_samples_per_outcome": 100,
            "store_full_distributions": True
        }
    
    trajectory_sampling_enabled = trajectory_sampling_config.get("enabled", False)
    sample_rate = trajectory_sampling_config.get("sample_rate", 1.0)
    stratified_sampling = trajectory_sampling_config.get("stratified", False)
    store_full_distributions = trajectory_sampling_config.get("store_full_distributions", True)
    
    # Override track_token_details in fast mode
    if fast_mode and track_token_details:
        if verbose:
            print("  ⚡ Fast mode enabled - disabling token detail tracking for performance")
        track_token_details = False
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        print(f"  Number of walks to evaluate: {num_walks}")
        print(f"  Start walk length: {min_start_length}-{max_start_length}")
        print(f"  Graph size: {graph.n} nodes")
        print(f"  Number of rules: {len(rules)}")
        print(f"  Token detail tracking: {track_token_details}")
        print(f"  Fast mode: {fast_mode}")
        if trajectory_sampling_enabled:
            print(f"  Trajectory sampling enabled: {sample_rate:.1%} rate")
            print(f"  Stratified sampling: {stratified_sampling}")
            print(f"  Store full distributions: {store_full_distributions}")
        print("="*60 + "\n")

    evaluation_results = []
    repeater_errors = 0
    ascender_errors = 0
    even_errors = 0
    broken_graph_errors = 0
    total_steps = 0

    # Enhanced token-by-token tracking
    kl_divergence_series = []
    token_level_data = []  # Detailed token-by-token information
    walk_trajectories = []  # NEW: Trajectory metadata for each walk
    
    # Track outcome counts for stratified sampling
    outcome_counts = {"completed": 0, "invalid_edge": 0, "end_token": 0, "invalid_token": 0} if stratified_sampling else None
    
    # Create progress bar
    if verbose:
        print("Starting evaluation...")
        walk_iterator = tqdm(range(num_walks), desc="Evaluating walks", unit="walk")
    else:
        walk_iterator = range(num_walks)

    for sample_idx in walk_iterator:

        # Generate starting walk
        start_node = random.choice(range(graph.n))
        start_walk = generate_valid_walk(
            graph, start_node, min_start_length, max_start_length, rules
        )
        
        if verbose and sample_idx == 0:
            print(f"\n  First walk starting from node {start_node}: {start_walk[:10]}..." if len(start_walk) > 10 else f"\n  First walk: {start_walk}")
            if fast_mode:
                print("  [FAST MODE: Skipping expensive distribution computations]")

        # Get device from model
        device = next(model.parameters()).device
        
        input_tensor = torch.tensor(
            [vocab.token2idx[str(node)] for node in start_walk], dtype=torch.long
        ).unsqueeze(0).to(device)

        generated_walk = start_walk[:]
        current_vertex = start_walk[-1]
        
        # NEW: Create trajectory metadata for this walk
        walk_trajectory = WalkTrajectoryMetadata(sample_idx, start_walk)
        
        # NEW: Determine if we should store full trajectory data for this walk
        should_store_full_trajectory = True
        if trajectory_sampling_enabled:
            if stratified_sampling:
                # For stratified sampling, store based on current outcome distribution
                should_store_full_trajectory = True  # Will be updated later when outcome is known
            else:
                # Simple random sampling
                should_store_full_trajectory = random.random() < sample_rate

        counter = 0
        kl_series = []  # KL values for this walk
        walk_token_data = []  # Detailed data for this walk

        while current_vertex in range(graph.n):
            logits = model(input_tensor)
            prediction_probs = F.softmax(logits[0, -1], dim=-1)
            next_vertex_idx = torch.argmax(logits[0, -1]).item()
            next_vertex = vocab.idx2token[next_vertex_idx]

            # Enhanced KL divergence computation with multiple reference distributions
            vocab_size = logits.shape[-1]
            
            # Skip expensive computations in fast mode
            if not fast_mode:
                # Multiple reference distributions for comparison (ensure they're on the same device)
                ref_distributions = {
                    'uniform_random': uniform_random_baseline(vocab_size).to(device),
                    'valid_neighbors': valid_neighbors_baseline(current_vertex, graph, vocab).to(device),
                    'degree_weighted': degree_weighted_baseline(current_vertex, graph, vocab).to(device),
                    'graph_structure': graph_edge_distribution(current_vertex, graph, vocab).to(device),
                    'exponential_mle': fit_exponential_mle(prediction_probs)  # Already on correct device
                }
            else:
                # In fast mode, only compute the most basic distribution
                ref_distributions = {
                    'uniform_random': uniform_random_baseline(vocab_size).to(device)
                }
            
            # Enhanced three-way comparison: LLM vs Graph vs Uniform vs Exponential
            # Make this conditional based on configuration to save memory
            core_comparison = None
            if not fast_mode and getattr(config, 'distribution_analysis', {}).get('enabled', True):
                core_comparison = compute_core_distribution_comparison(
                    prediction_probs, current_vertex, graph, vocab, device, 
                    rules=rules, walk_history=generated_walk
                )
            
            # NEW: Compute all distribution distances (KL, KS, JS, Wasserstein)
            if not fast_mode:
                distances = compute_distribution_distances(prediction_probs, ref_distributions, vocab)
            else:
                # In fast mode, only compute basic KL divergence
                distances = {'kl_divergences': {'uniform_random': 0.0}}
            
            # Keep backward compatibility with old kl_values variable
            kl_values = distances['kl_divergences']
            kl_series.append(kl_values['uniform_random'])  # Use uniform random as primary KL reference
            
            # NEW: Store full probability distribution for this step (conditionally)
            if should_store_full_trajectory and store_full_distributions:
                walk_trajectory.probability_distributions.append(prediction_probs.cpu().detach().numpy().copy())
            
            # NEW: Analyze probability distribution
            if not fast_mode:
                dist_analysis = analyze_probability_distribution(prediction_probs, vocab, graph, current_vertex)
            else:
                dist_analysis = {}  # Skip in fast mode
            
            # Store detailed token information with rule violation analysis
            # In fast mode, skip token details regardless of track_token_details setting
            if track_token_details and not fast_mode:
                # Analyze potential rule violations for this prediction
                predicted_walk = generated_walk + [int(next_vertex)] if next_vertex.isdigit() else generated_walk
                rule_violations = analyze_rule_violations_for_token(
                    current_vertex, next_vertex, predicted_walk, graph, rules, vocab, prediction_probs
                )
                
                # Compute additional uncertainty metrics
                top_k_probs, top_k_indices = torch.topk(prediction_probs, k=min(10, vocab_size))
                gini_coeff = compute_gini_coefficient(prediction_probs)
                effective_vocab = torch.sum(prediction_probs > 0.01).item()
                
                # Check if this is near a repeater deadline
                repeater_context = analyze_repeater_context(current_vertex, generated_walk, rules)
                
                # Calculate normalized metrics comparing model to baselines
                normalized_metrics = calculate_normalized_metrics(prediction_probs, ref_distributions, next_vertex_idx)
                
                token_info = {
                    'walk_idx': sample_idx,
                    'step_idx': counter,
                    'context_length': input_tensor.size(1),
                    'current_vertex': current_vertex,
                    'predicted_vertex': next_vertex,
                    'predicted_idx': next_vertex_idx,
                    'prediction_confidence': prediction_probs[next_vertex_idx].item(),
                    'entropy': entropy_from_logits(logits[0, -1]),
                    'kl_divergences': kl_values,
                    'normalized_metrics': normalized_metrics,  # New normalized baseline comparisons
                    'core_distribution_comparison': core_comparison if core_comparison else {},  # Enhanced 3-way distribution comparison
                    'top_5_predictions': get_top_k_predictions(logits[0, -1], vocab, k=5),
                    'top_5_prob_sum': torch.sum(top_k_probs[:5]).item(),
                    'top_10_prob_sum': torch.sum(top_k_probs).item(),
                    'gini_coefficient': gini_coeff,
                    'effective_vocab_size': effective_vocab,
                    'prob_variance': torch.var(prediction_probs).item(),
                    'prob_std': torch.std(prediction_probs).item(),
                    'context_tokens': [vocab.idx2token[idx] for idx in input_tensor[0].cpu().tolist()],
                    'is_valid_edge': graph.has_edge(current_vertex, int(next_vertex)) if next_vertex.isdigit() else False,
                    'rule_violations': rule_violations,
                    'repeater_context': repeater_context,
                    'walk_so_far': generated_walk.copy()
                }
                walk_token_data.append(token_info)
            
            # NEW: Add metrics to trajectory
            if not fast_mode:
                trajectory_metrics = {
                    'entropy': entropy_from_logits(logits[0, -1]),
                    'confidence': prediction_probs[next_vertex_idx].item(),
                    'perplexity': torch.exp(-torch.log(prediction_probs[next_vertex_idx])).item(),
                    'kl_divergences': distances['kl_divergences'],
                    'ks_distances': distances.get('ks_distances', {}),
                    'distribution_stats': {
                        'top_k_mass': [dist_analysis.get('top_1_mass', 0), dist_analysis.get('top_5_mass', 0), dist_analysis.get('top_10_mass', 0)],
                        'effective_support_size': dist_analysis.get('effective_support_size', 0),
                        'mode_probability': dist_analysis.get('mode_probability', 0),
                        'tail_mass': dist_analysis.get('tail_mass', 0),
                        'valid_neighbor_mass': dist_analysis.get('valid_neighbor_mass', 0),
                        'invalid_edge_mass': dist_analysis.get('invalid_edge_mass', 0)
                    },
                    # Add core distribution comparison metrics to trajectory (if enabled)
                    'core_distribution_metrics': {
                        'kl_from_graph': core_comparison['distribution_distances']['graph_structure']['kl_divergence'] if core_comparison else 0,
                        'kl_from_uniform': core_comparison['distribution_distances']['uniform_valid']['kl_divergence'] if core_comparison else 0,
                        'kl_from_exponential': core_comparison['distribution_distances']['exponential_fitted']['kl_divergence'] if core_comparison else 0,
                        'structural_awareness': core_comparison['prediction_quality_scores']['structural_awareness'] if core_comparison else 0,
                        'neighbor_prioritization': core_comparison['prediction_quality_scores']['neighbor_prioritization'] if core_comparison else 0,
                        'overall_quality': core_comparison['prediction_quality_scores']['overall_quality'] if core_comparison else 0,
                        'graph_structure_overlap': core_comparison['distribution_overlap_analysis']['graph_structure']['overlap_coefficient'] if core_comparison else 0,
                        'top1_agreement_with_graph': core_comparison['distribution_overlap_analysis']['graph_structure']['agreement_on_top_k']['top_1'] if core_comparison else 0,
                        'top5_agreement_with_graph': core_comparison['distribution_overlap_analysis']['graph_structure']['agreement_on_top_k']['top_5'] if core_comparison else 0
                    } if core_comparison else {}
                }
            else:
                # Minimal metrics in fast mode
                trajectory_metrics = {
                    'entropy': entropy_from_logits(logits[0, -1]),
                    'confidence': prediction_probs[next_vertex_idx].item(),
                    'perplexity': torch.exp(-torch.log(prediction_probs[next_vertex_idx])).item(),
                    'kl_divergences': distances['kl_divergences'],
                    'ks_distances': {},
                    'distribution_stats': {},
                    'core_distribution_metrics': {
                        'kl_from_graph': 0,
                        'kl_from_uniform': 0,
                        'kl_from_exponential': 0,
                        'structural_awareness': 0,
                        'neighbor_prioritization': 0,
                        'overall_quality': 0,
                        'graph_structure_overlap': 0,
                        'top1_agreement_with_graph': 0,
                        'top5_agreement_with_graph': 0
                    }
                }
            walk_trajectory.add_step_metrics(counter, trajectory_metrics)
            
            # NEW: Detect critical points
            walk_trajectory.detect_critical_point(counter)

            if next_vertex == "<END>":
                walk_trajectory.termination_reason = "end_token"
                walk_trajectory.termination_step = counter
                break

            try:
                next_vertex_int = int(next_vertex)
            except ValueError:
                walk_trajectory.termination_reason = "invalid_token"
                walk_trajectory.termination_step = counter
                break  # Not a valid node

            # Broken graph connection check
            if not graph.has_edge(current_vertex, next_vertex_int):
                broken_graph_errors += 1
                walk_trajectory.termination_reason = "invalid_edge"
                walk_trajectory.termination_step = counter
                break

            generated_walk.append(next_vertex_int)
            input_tensor = torch.cat(
                (input_tensor, torch.tensor([[next_vertex_idx]], dtype=torch.long).to(device)),
                dim=1,
            )

            current_vertex = next_vertex_int
            counter += 1
            total_steps += 1

        # NEW: Finalize trajectory metadata
        walk_trajectory.generated_walk = generated_walk
        walk_trajectory.final_length = len(generated_walk)
        
        # If walk ended naturally (reached max length without other termination)
        if walk_trajectory.termination_reason is None:
            walk_trajectory.termination_reason = "completed"
            walk_trajectory.termination_step = counter
        
        # NEW: Update stratified sampling decision now that we know the outcome
        if trajectory_sampling_enabled and stratified_sampling:
            outcome = walk_trajectory.termination_reason
            min_samples = trajectory_sampling_config.get("min_samples_per_outcome", 100)
            
            # Store if we haven't reached minimum samples for this outcome
            if outcome_counts[outcome] < min_samples:
                should_store_full_trajectory = True
                outcome_counts[outcome] += 1
            else:
                # Use random sampling for additional samples
                should_store_full_trajectory = random.random() < (sample_rate * 0.1)  # Reduced rate after minimum
        
        # Rule violation tracking
        walk_violations = []
        for rule in rules:
            if hasattr(rule, "is_repeater_rule") and rule.is_repeater_rule:
                if not rule.is_satisfied_by(generated_walk, graph):
                    repeater_errors += 1
                    walk_violations.append("repeater")
                    walk_trajectory.violation_timeline.append((counter, "repeater", "walk_end"))
            elif hasattr(rule, "is_ascender_rule") and rule.is_ascender_rule:
                if not rule.is_satisfied_by(generated_walk, graph):
                    ascender_errors += 1
                    walk_violations.append("ascender")
                    walk_trajectory.violation_timeline.append((counter, "ascender", "walk_end"))
            elif hasattr(rule, "is_even_rule") and rule.is_even_rule:
                if not rule.is_satisfied_by(generated_walk, graph):
                    even_errors += 1
                    walk_violations.append("even")
                    walk_trajectory.violation_timeline.append((counter, "even", "walk_end"))
        
        # NEW: Compute trajectory statistics and store trajectory (conditionally)
        walk_trajectory.compute_statistics()
        
        # Only store full trajectory if sampling decision permits
        if should_store_full_trajectory:
            walk_trajectories.append(walk_trajectory)
        else:
            # For memory efficiency, store only summary statistics
            summary_trajectory = WalkTrajectoryMetadata(sample_idx, start_walk)
            summary_trajectory.generated_walk = generated_walk
            summary_trajectory.final_length = len(generated_walk)
            summary_trajectory.termination_reason = walk_trajectory.termination_reason
            summary_trajectory.termination_step = walk_trajectory.termination_step
            summary_trajectory.violation_timeline = walk_trajectory.violation_timeline
            # Don't store probability_distributions or detailed uncertainty trajectory
            summary_trajectory.compute_statistics()
            walk_trajectories.append(summary_trajectory)
        
        # Update progress bar with current error rates
        if verbose:
            walk_iterator.set_postfix({
                "repeater_err": f"{repeater_errors/(sample_idx+1):.2%}",
                "ascender_err": f"{ascender_errors/(sample_idx+1):.2%}",
                "even_err": f"{even_errors/(sample_idx+1):.2%}",
                "broken_graph": f"{broken_graph_errors/(sample_idx+1):.2%}"
            })

        # Store walk-level data
        kl_divergence_series.append(kl_series)
        if track_token_details:
            token_level_data.extend(walk_token_data)

        evaluation_results.append(
            {
                "start_walk": start_walk,
                "generated_walk": generated_walk,
                "walk_length": len(generated_walk),
                "num_generation_steps": counter,
            }
        )

    # --- Summarize error rates ---
    error_summary = {
        "repeater_error_rate": repeater_errors / num_walks,
        "ascender_error_rate": ascender_errors / num_walks,
        "even_error_rate": even_errors / num_walks,
        "broken_graph_error_rate": broken_graph_errors / num_walks,
        "total_steps": total_steps,
        "avg_steps_per_walk": total_steps / num_walks if num_walks > 0 else 0,
    }

    if verbose:
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        print(f"  Total walks evaluated: {num_walks}")
        print(f"  Total generation steps: {total_steps}")
        print(f"  Average steps per walk: {error_summary['avg_steps_per_walk']:.2f}")
        print("\nError Rates:")
        print(f"  Repeater violations: {repeater_errors}/{num_walks} ({error_summary['repeater_error_rate']:.2%})")
        print(f"  Ascender violations: {ascender_errors}/{num_walks} ({error_summary['ascender_error_rate']:.2%})")
        print(f"  Even rule violations: {even_errors}/{num_walks} ({error_summary['even_error_rate']:.2%})")
        print(f"  Broken graph connections: {broken_graph_errors}/{num_walks} ({error_summary['broken_graph_error_rate']:.2%})")
        
        # Calculate overall accuracy
        total_errors = repeater_errors + ascender_errors + even_errors + broken_graph_errors
        overall_accuracy = 1 - (total_errors / (num_walks * len(rules)))
        print(f"\n  Overall rule compliance: {overall_accuracy:.2%}")
        print("="*60 + "\n")

    # Analyze progressive difficulty patterns
    progressive_analysis = analyze_progressive_difficulty(token_level_data) if track_token_details else {}
    
    # Select exemplar walks for paper documentation
    exemplar_walks = select_exemplar_walks(evaluation_results, token_level_data, progressive_analysis) if track_token_details else {}
    
    # NEW: Create aggregate trajectory metadata
    evaluation_trajectories = EvaluationTrajectoryMetadata(walk_trajectories) if walk_trajectories else None
    
    # Return enhanced results including token-level data, progressive analysis, exemplars, and trajectories
    return evaluation_results, error_summary, kl_divergence_series, token_level_data, progressive_analysis, exemplar_walks, evaluation_trajectories


def count_rule_violations(walk, graph, rules):
    violations = 0
    for i in range(len(walk)):
        if not check_rule_compliance(walk[: i + 1], graph, rules):
            violations += 1
    return violations


def kl_divergence(p_logits, q_probs):
    """
    Compute KL divergence D_KL(P || Q) where:
    - p_logits: model output logits (unnormalized, shape [vocab_size])
    - q_probs: reference distribution (probabilities, shape [vocab_size])
    """
    # Ensure both tensors are on the same device
    if q_probs.device != p_logits.device:
        q_probs = q_probs.to(p_logits.device)
    
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    # Add small epsilon to avoid log(0)
    q_probs = q_probs + 1e-8
    kl = torch.sum(torch.exp(p_log_probs) * (p_log_probs - torch.log(q_probs)))
    return kl.item()


def negative_exponential_distribution(vocab_size, scale=1.0):
    """
    Returns a negative exponential distribution over vocab indices.
    """
    x = np.arange(vocab_size)
    probs = np.exp(-x / scale)
    probs /= probs.sum()
    return torch.tensor(probs, dtype=torch.float32)


def uniform_distribution(vocab_size):
    """
    Returns a uniform distribution over vocab indices.
    """
    probs = np.ones(vocab_size) / vocab_size
    return torch.tensor(probs, dtype=torch.float32)


def graph_edge_distribution(vertex, graph, vocab):
    """
    Returns a distribution based on actual edge weights from the graph structure.
    """
    vocab_size = len(vocab.token2idx)
    probs = np.zeros(vocab_size)
    
    if vertex < graph.n:
        neighbors, edge_probs = graph.get_edge_probabilities(vertex)
        for neighbor, prob in zip(neighbors, edge_probs):
            neighbor_token = str(neighbor)
            if neighbor_token in vocab.token2idx:
                neighbor_idx = vocab.token2idx[neighbor_token]
                probs[neighbor_idx] = prob
    
    return torch.tensor(probs, dtype=torch.float32)


def uniform_random_baseline(vocab_size):
    """
    Returns a uniform distribution over all vocabulary tokens.
    """
    probs = np.ones(vocab_size) / vocab_size
    return torch.tensor(probs, dtype=torch.float32)


def valid_neighbors_baseline(vertex, graph, vocab):
    """
    Returns a uniform distribution over valid graph neighbors only.
    """
    vocab_size = len(vocab.token2idx)
    probs = np.zeros(vocab_size)
    
    if vertex < graph.n:
        neighbors = graph.get_neighbors(vertex)
        if len(neighbors) > 0:
            neighbor_prob = 1.0 / len(neighbors)
            for neighbor in neighbors:
                neighbor_token = str(neighbor)
                if neighbor_token in vocab.token2idx:
                    neighbor_idx = vocab.token2idx[neighbor_token]
                    probs[neighbor_idx] = neighbor_prob
    
    return torch.tensor(probs, dtype=torch.float32)


def degree_weighted_baseline(vertex, graph, vocab):
    """
    Returns a distribution weighted by neighbor degrees.
    """
    vocab_size = len(vocab.token2idx)
    probs = np.zeros(vocab_size)
    
    if vertex < graph.n:
        neighbors = graph.get_neighbors(vertex)
        if len(neighbors) > 0:
            # Weight by degree of each neighbor
            degrees = np.array([graph.get_degree(neighbor) for neighbor in neighbors])
            if degrees.sum() > 0:
                degree_probs = degrees / degrees.sum()
                for neighbor, prob in zip(neighbors, degree_probs):
                    neighbor_token = str(neighbor)
                    if neighbor_token in vocab.token2idx:
                        neighbor_idx = vocab.token2idx[neighbor_token]
                        probs[neighbor_idx] = prob
    
    return torch.tensor(probs, dtype=torch.float32)


def repeater_oracle_baseline(vertex, graph, vocab, rules, walk_history):
    """
    Oracle baseline that knows exactly what to do for repeater rules.
    Returns probability 1.0 for the correct repeater continuation, 0.0 elsewhere.
    
    Args:
        vertex: Current vertex in walk
        graph: Graph object
        vocab: Vocabulary object
        rules: List of rule objects (to find repeater rule)
        walk_history: List of vertices visited so far in walk
        
    Returns:
        Tensor with oracle repeater probabilities
    """
    vocab_size = len(vocab.token2idx)
    probs = np.zeros(vocab_size)
    
    # Find the repeater rule
    repeater_rule = None
    for rule in rules:
        if hasattr(rule, 'member_nodes') and hasattr(rule, 'k_values'):
            if vertex in rule.member_nodes:
                repeater_rule = rule
                break
    
    if repeater_rule is None or len(walk_history) < 2:
        # If no repeater rule applies or insufficient history, fall back to uniform valid
        return valid_neighbors_baseline(vertex, graph, vocab)
    
    # Get the k-value for this vertex
    k = repeater_rule.k_values.get(vertex, 2)  # Default k=2 if not found
    
    # Check if we have enough history to determine the pattern
    if len(walk_history) < k:
        # Not enough history, return uniform over valid neighbors
        return valid_neighbors_baseline(vertex, graph, vocab)
    
    # Find the vertex that was visited k steps ago
    target_vertex = walk_history[-(k+1)] if len(walk_history) > k else None
    
    if target_vertex is not None:
        # Check if target vertex is a valid neighbor
        neighbors = graph.get_neighbors(vertex)
        if target_vertex in neighbors:
            target_token = str(target_vertex)
            if target_token in vocab.token2idx:
                target_idx = vocab.token2idx[target_token]
                probs[target_idx] = 1.0
                return torch.tensor(probs, dtype=torch.float32)
    
    # If can't find valid repeater move, fall back to uniform valid
    return valid_neighbors_baseline(vertex, graph, vocab)


def ascender_oracle_baseline(vertex, graph, vocab, rules):
    """
    Oracle baseline that knows exactly what to do for ascender rules.
    Returns uniform probability over all valid neighbors that are larger than current vertex.
    
    Args:
        vertex: Current vertex in walk
        graph: Graph object  
        vocab: Vocabulary object
        rules: List of rule objects (to find ascender rule)
        
    Returns:
        Tensor with oracle ascender probabilities
    """
    vocab_size = len(vocab.token2idx)
    probs = np.zeros(vocab_size)
    
    # Check if this vertex has an ascender rule
    is_ascender = False
    for rule in rules:
        if hasattr(rule, 'member_nodes') and vertex in rule.member_nodes:
            if 'ascender' in str(type(rule)).lower():
                is_ascender = True
                break
    
    if not is_ascender:
        # If no ascender rule applies, fall back to uniform valid
        return valid_neighbors_baseline(vertex, graph, vocab)
    
    # Get valid neighbors that are larger than current vertex
    neighbors = graph.get_neighbors(vertex)
    valid_ascenders = [n for n in neighbors if n > vertex]
    
    if len(valid_ascenders) > 0:
        # Uniform probability over valid ascending moves
        ascender_prob = 1.0 / len(valid_ascenders)
        for neighbor in valid_ascenders:
            neighbor_token = str(neighbor)
            if neighbor_token in vocab.token2idx:
                neighbor_idx = vocab.token2idx[neighbor_token]
                probs[neighbor_idx] = ascender_prob
    else:
        # No valid ascending moves, fall back to uniform valid
        return valid_neighbors_baseline(vertex, graph, vocab)
    
    return torch.tensor(probs, dtype=torch.float32)


def even_oracle_baseline(vertex, graph, vocab, rules):
    """
    Oracle baseline that knows exactly what to do for even rules.
    Returns uniform probability over all valid neighbors that are even numbers.
    
    Args:
        vertex: Current vertex in walk
        graph: Graph object
        vocab: Vocabulary object
        rules: List of rule objects (to find even rule)
        
    Returns:
        Tensor with oracle even probabilities
    """
    vocab_size = len(vocab.token2idx)
    probs = np.zeros(vocab_size)
    
    # Check if this vertex has an even rule
    is_even_rule = False
    for rule in rules:
        if hasattr(rule, 'member_nodes') and vertex in rule.member_nodes:
            if 'even' in str(type(rule)).lower():
                is_even_rule = True
                break
    
    if not is_even_rule:
        # If no even rule applies, fall back to uniform valid
        return valid_neighbors_baseline(vertex, graph, vocab)
    
    # Get valid neighbors that are even
    neighbors = graph.get_neighbors(vertex)
    valid_evens = [n for n in neighbors if n % 2 == 0]
    
    if len(valid_evens) > 0:
        # Uniform probability over valid even moves
        even_prob = 1.0 / len(valid_evens)
        for neighbor in valid_evens:
            neighbor_token = str(neighbor)
            if neighbor_token in vocab.token2idx:
                neighbor_idx = vocab.token2idx[neighbor_token]
                probs[neighbor_idx] = even_prob
    else:
        # No valid even moves, fall back to uniform valid
        return valid_neighbors_baseline(vertex, graph, vocab)
    
    return torch.tensor(probs, dtype=torch.float32)


def rule_aware_oracle_baseline(vertex, graph, vocab, rules, walk_history):
    """
    Combined oracle baseline that knows the current rule context and responds optimally.
    This represents a "certain model" that knows exactly what to do in any rule situation.
    
    Args:
        vertex: Current vertex in walk
        graph: Graph object
        vocab: Vocabulary object
        rules: List of rule objects
        walk_history: List of vertices visited so far in walk
        
    Returns:
        Tensor with oracle probabilities for optimal rule compliance
    """
    # Check which rule applies to current vertex (in order of precedence)
    
    # 1. Check for repeater rule (most specific)
    for rule in rules:
        if hasattr(rule, 'k_values') and vertex in getattr(rule, 'member_nodes', []):
            return repeater_oracle_baseline(vertex, graph, vocab, rules, walk_history)
    
    # 2. Check for ascender rule  
    for rule in rules:
        if hasattr(rule, 'member_nodes') and vertex in rule.member_nodes:
            if 'ascender' in str(type(rule)).lower():
                return ascender_oracle_baseline(vertex, graph, vocab, rules)
    
    # 3. Check for even rule
    for rule in rules:
        if hasattr(rule, 'member_nodes') and vertex in rule.member_nodes:
            if 'even' in str(type(rule)).lower():
                return even_oracle_baseline(vertex, graph, vocab, rules)
    
    # 4. If no specific rule applies, use graph structure (optimal for general navigation)
    return graph_edge_distribution(vertex, graph, vocab)


def optimal_path_baseline(vertex, graph, vocab, rules, walk_history):
    """
    Optimal path baseline that assigns probability 1.0 to the single best rule-compliant move.
    This is the most "certain" baseline - it knows exactly the one correct next step.
    
    Args:
        vertex: Current vertex in walk
        graph: Graph object
        vocab: Vocabulary object
        rules: List of rule objects
        walk_history: List of vertices visited so far in walk
        
    Returns:
        Tensor with probability 1.0 for optimal move, 0.0 elsewhere
    """
    vocab_size = len(vocab.token2idx)
    probs = np.zeros(vocab_size)
    
    # Get the rule-aware oracle distribution
    oracle_dist = rule_aware_oracle_baseline(vertex, graph, vocab, rules, walk_history)
    
    # Find the highest probability move from oracle
    max_idx = torch.argmax(oracle_dist)
    max_prob = oracle_dist[max_idx].item()
    
    if max_prob > 0:
        # Assign probability 1.0 to the best move
        probs[max_idx] = 1.0
    else:
        # Fallback: if oracle has no valid moves, use uniform valid
        return valid_neighbors_baseline(vertex, graph, vocab)
    
    return torch.tensor(probs, dtype=torch.float32)


def entropy_from_logits(logits):
    """
    Compute entropy from model logits.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs)
    return entropy.item()


def cross_entropy(p_probs, q_probs):
    """
    Compute cross-entropy H(P,Q) = -Σ p(i) log q(i).
    
    Args:
        p_probs: True distribution (tensor)
        q_probs: Predicted distribution (tensor)
        
    Returns:
        Cross-entropy value (float)
    """
    # Ensure tensors and add small epsilon to avoid log(0)
    if not torch.is_tensor(p_probs):
        p_probs = torch.tensor(p_probs, dtype=torch.float32)
    if not torch.is_tensor(q_probs):
        q_probs = torch.tensor(q_probs, dtype=torch.float32)
    
    epsilon = 1e-8
    q_probs = q_probs + epsilon
    
    return -torch.sum(p_probs * torch.log(q_probs)).item()


def mutual_information(p_joint, p_x, p_y):
    """
    Compute mutual information I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y))).
    
    Args:
        p_joint: Joint probability distribution P(X,Y)
        p_x: Marginal distribution P(X) 
        p_y: Marginal distribution P(Y)
        
    Returns:
        Mutual information value (float)
    """
    epsilon = 1e-8
    
    if not torch.is_tensor(p_joint):
        p_joint = torch.tensor(p_joint, dtype=torch.float32)
    if not torch.is_tensor(p_x):
        p_x = torch.tensor(p_x, dtype=torch.float32)
    if not torch.is_tensor(p_y):
        p_y = torch.tensor(p_y, dtype=torch.float32)
    
    # Compute product of marginals
    marginal_product = torch.outer(p_x, p_y)
    
    # Add epsilon to avoid division by zero
    p_joint = p_joint + epsilon
    marginal_product = marginal_product + epsilon
    
    # I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = torch.sum(p_joint * torch.log(p_joint / marginal_product))
    
    return mi.item()


def conditional_entropy(p_joint, p_x):
    """
    Compute conditional entropy H(Y|X) = H(X,Y) - H(X).
    
    Args:
        p_joint: Joint probability distribution P(X,Y)
        p_x: Marginal distribution P(X)
        
    Returns:
        Conditional entropy value (float)
    """
    epsilon = 1e-8
    
    if not torch.is_tensor(p_joint):
        p_joint = torch.tensor(p_joint, dtype=torch.float32)
    if not torch.is_tensor(p_x):
        p_x = torch.tensor(p_x, dtype=torch.float32)
    
    # Joint entropy H(X,Y)
    joint_entropy = -torch.sum(p_joint * torch.log(p_joint + epsilon)).item()
    
    # Marginal entropy H(X)
    marginal_entropy = -torch.sum(p_x * torch.log(p_x + epsilon)).item()
    
    return joint_entropy - marginal_entropy


def entropy_rate(entropy_sequence):
    """
    Compute entropy rate (derivative) from a sequence of entropy values.
    
    Args:
        entropy_sequence: List or array of entropy values over time
        
    Returns:
        Dictionary with entropy rate statistics
    """
    if len(entropy_sequence) < 2:
        return {'rate': 0.0, 'trend': 0.0, 'acceleration': 0.0}
    
    sequence = np.array(entropy_sequence)
    
    # First derivative (rate of change)
    rates = np.diff(sequence)
    
    # Second derivative (acceleration)
    accelerations = np.diff(rates) if len(rates) > 1 else np.array([0.0])
    
    # Linear trend
    if len(sequence) > 2:
        time_steps = np.arange(len(sequence))
        trend_slope = np.polyfit(time_steps, sequence, 1)[0]
    else:
        trend_slope = rates[0] if len(rates) > 0 else 0.0
    
    return {
        'rate': float(np.mean(rates)),
        'rate_std': float(np.std(rates)),
        'trend': float(trend_slope),
        'acceleration': float(np.mean(accelerations)),
        'final_rate': float(rates[-1]) if len(rates) > 0 else 0.0
    }


def maximum_entropy_reference(vocab_size):
    """
    Compute maximum possible entropy for given vocabulary size.
    
    Args:
        vocab_size: Size of vocabulary
        
    Returns:
        Maximum entropy value (log(vocab_size))
    """
    return float(np.log(vocab_size))


def entropy_efficiency(entropy, max_entropy):
    """
    Compute entropy efficiency (how much of maximum entropy is used).
    
    Args:
        entropy: Actual entropy value
        max_entropy: Maximum possible entropy
        
    Returns:
        Efficiency ratio (0-1 scale)
    """
    if max_entropy <= 0:
        return 0.0
    return min(1.0, max(0.0, entropy / max_entropy))


def compute_comprehensive_entropy_metrics(prediction_probs, baselines, vocab_size, walk_history_entropies=None):
    """
    Compute comprehensive entropy metrics including all information theory measures.
    
    Args:
        prediction_probs: Model prediction probabilities
        baselines: Dictionary of baseline distributions
        vocab_size: Vocabulary size for maximum entropy calculation
        walk_history_entropies: Optional sequence of entropy values for rate calculation
        
    Returns:
        Dictionary with comprehensive entropy metrics
    """
    model_entropy = -torch.sum(prediction_probs * torch.log(prediction_probs + 1e-8)).item()
    max_entropy = maximum_entropy_reference(vocab_size)
    
    metrics = {
        'model_entropy': model_entropy,
        'max_entropy': max_entropy,
        'entropy_efficiency': entropy_efficiency(model_entropy, max_entropy),
        'cross_entropies': {},
        'mutual_informations': {},
        'conditional_entropies': {},
        'information_gains': {},
        'relative_entropies': {},
        'entropy_improvements': {}
    }
    
    # Compute metrics against each baseline
    for baseline_name, baseline_dist in baselines.items():
        # Cross-entropy
        cross_ent = cross_entropy(baseline_dist, prediction_probs)
        metrics['cross_entropies'][baseline_name] = cross_ent
        
        # Baseline entropy
        baseline_entropy = -torch.sum(baseline_dist * torch.log(baseline_dist + 1e-8)).item()
        
        # Information gain
        info_gain = baseline_entropy - model_entropy
        metrics['information_gains'][baseline_name] = info_gain
        
        # Relative entropy
        rel_entropy = model_entropy / baseline_entropy if baseline_entropy > 0 else float('inf')
        metrics['relative_entropies'][baseline_name] = rel_entropy
        
        # Entropy improvement
        entropy_improvement = info_gain / baseline_entropy if baseline_entropy > 0 else 0.0
        metrics['entropy_improvements'][baseline_name] = entropy_improvement
        
        # Mutual information (approximation using marginal distributions)
        try:
            # Simple approximation: treat as binary problem (predicted vs not predicted)
            p_pred = torch.max(prediction_probs).item()
            p_base = torch.max(baseline_dist).item()
            
            # Joint distribution approximation
            p_joint = torch.tensor([[p_pred * p_base, p_pred * (1-p_base)],
                                   [(1-p_pred) * p_base, (1-p_pred) * (1-p_base)]])
            p_x = torch.tensor([p_pred, 1-p_pred])
            p_y = torch.tensor([p_base, 1-p_base])
            
            mi = mutual_information(p_joint, p_x, p_y)
            metrics['mutual_informations'][baseline_name] = mi
            
            # Conditional entropy
            cond_ent = conditional_entropy(p_joint, p_x)
            metrics['conditional_entropies'][baseline_name] = cond_ent
            
        except Exception:
            # If MI/conditional entropy computation fails, set to 0
            metrics['mutual_informations'][baseline_name] = 0.0
            metrics['conditional_entropies'][baseline_name] = 0.0
    
    # Entropy rate if walk history is provided
    if walk_history_entropies:
        entropy_rate_metrics = entropy_rate(walk_history_entropies + [model_entropy])
        metrics['entropy_rate'] = entropy_rate_metrics
    
    return metrics


def get_top_k_predictions(logits, vocab, k=5):
    """
    Get top k predictions with their probabilities.
    """
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    predictions = []
    for i in range(k):
        idx = top_k_indices[i].item()
        prob = top_k_probs[i].item()
        token = vocab.idx2token[idx]
        predictions.append({'token': token, 'probability': prob, 'idx': idx})
    
    return predictions


def analyze_rule_violations_for_token(current_vertex, predicted_vertex, predicted_walk, graph, rules, vocab, prediction_probs):
    """
    Analyze potential rule violations for a specific token prediction.
    
    Args:
        current_vertex: Current position in walk
        predicted_vertex: Model's predicted next vertex 
        predicted_walk: Walk including the predicted next step
        graph: Graph structure
        rules: List of rule objects
        vocab: Vocabulary object
        prediction_probs: Softmax probabilities for all possible next tokens
        
    Returns:
        Dictionary with detailed rule violation analysis
    """
    violations = {
        'has_violation': False,
        'violation_types': [],
        'rule_specific_analysis': {},
        'alternative_valid_tokens': [],
        'violation_probabilities': {}
    }
    
    # Check if predicted token is valid
    if not predicted_vertex.isdigit():
        return violations
        
    predicted_node = int(predicted_vertex)
    
    # Check each rule type
    for rule in rules:
        rule_name = type(rule).__name__.lower()
        rule_analysis = {
            'violates': False,
            'violation_position': None,
            'context_nodes': [],
            'expected_behavior': ''
        }
        
        if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
            rule_analysis = analyze_ascender_violation(predicted_walk, rule, predicted_node)
        elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
            rule_analysis = analyze_even_violation(predicted_walk, rule, predicted_node)
        elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            rule_analysis = analyze_repeater_violation(predicted_walk, rule, predicted_node)
            
        violations['rule_specific_analysis'][rule_name] = rule_analysis
        
        if rule_analysis['violates']:
            violations['has_violation'] = True
            violations['violation_types'].append(rule_name)
    
    # Check graph connectivity violation
    if current_vertex < graph.n and not graph.has_edge(current_vertex, predicted_node):
        violations['has_violation'] = True
        violations['violation_types'].append('invalid_edge')
        violations['rule_specific_analysis']['invalid_edge'] = {
            'violates': True,
            'violation_position': len(predicted_walk) - 1,
            'context_nodes': [current_vertex],
            'expected_behavior': f'Valid neighbors of {current_vertex}'
        }
    
    # Find valid alternatives and their probabilities
    if current_vertex < graph.n:
        valid_neighbors = graph.get_neighbors(current_vertex)
        for neighbor in valid_neighbors:
            neighbor_token = str(neighbor)
            if neighbor_token in vocab.token2idx:
                neighbor_idx = vocab.token2idx[neighbor_token]
                prob = prediction_probs[neighbor_idx].item()
                
                # Check if this neighbor would violate rules
                test_walk = predicted_walk[:-1] + [neighbor]
                would_violate = False
                for rule in rules:
                    if not rule.is_satisfied_by(test_walk, graph):
                        would_violate = True
                        break
                
                if not would_violate:
                    violations['alternative_valid_tokens'].append({
                        'token': neighbor_token,
                        'probability': prob,
                        'idx': neighbor_idx
                    })
    
    # Calculate violation probabilities for each rule type
    violations['violation_probabilities'] = calculate_violation_probabilities(
        current_vertex, predicted_walk, graph, rules, vocab, prediction_probs
    )
    
    return violations


def analyze_ascender_violation(walk, rule, predicted_node):
    """Analyze ascender rule violations."""
    analysis = {'violates': False, 'violation_position': None, 'context_nodes': [], 'expected_behavior': ''}
    
    # Find ascender nodes in walk history
    ascender_positions = []
    for i, node in enumerate(walk[:-1]):  # Exclude the predicted node
        if node in rule.member_nodes:
            ascender_positions.append((i, node))
    
    if ascender_positions:
        # Check if predicted node violates ascender rule
        last_ascender_pos, last_ascender_node = ascender_positions[-1]
        if predicted_node <= last_ascender_node:
            analysis['violates'] = True
            analysis['violation_position'] = len(walk) - 1
            analysis['context_nodes'] = [last_ascender_node]
            analysis['expected_behavior'] = f'Must be > {last_ascender_node} (last ascender)'
    
    return analysis


def analyze_even_violation(walk, rule, predicted_node):
    """Analyze even rule violations."""
    analysis = {'violates': False, 'violation_position': None, 'context_nodes': [], 'expected_behavior': ''}
    
    # Find even rule nodes in walk history
    even_positions = []
    for i, node in enumerate(walk[:-1]):  # Exclude the predicted node
        if node in rule.member_nodes:
            even_positions.append((i, node))
    
    if even_positions and predicted_node % 2 != 0:
        last_even_pos, last_even_node = even_positions[-1]
        analysis['violates'] = True
        analysis['violation_position'] = len(walk) - 1
        analysis['context_nodes'] = [last_even_node]
        analysis['expected_behavior'] = f'Must be even (after visiting even node {last_even_node})'
    
    return analysis


def analyze_repeater_violation(walk, rule, predicted_node):
    """Analyze repeater rule violations."""
    analysis = {'violates': False, 'violation_position': None, 'context_nodes': [], 'expected_behavior': ''}
    
    # Check if predicted node is a repeater
    if predicted_node in rule.members_nodes_dict:
        k = rule.members_nodes_dict[predicted_node]
        
        # Find previous occurrences of this node
        previous_positions = [i for i, node in enumerate(walk[:-1]) if node == predicted_node]
        
        if previous_positions:
            last_position = previous_positions[-1]
            current_position = len(walk) - 1
            distance = current_position - last_position
            
            if distance != k:
                analysis['violates'] = True
                analysis['violation_position'] = current_position
                analysis['context_nodes'] = [predicted_node]
                analysis['expected_behavior'] = f'Must appear every {k} steps (last at position {last_position})'
    
    return analysis


def fit_exponential_mle(probs):
    """
    Fit an exponential distribution to the model's probability distribution using MLE.
    Returns a tensor with the fitted exponential distribution.
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Fit exponential parameter using MLE
    # For exponential dist: p(k) ∝ exp(-λk), we fit λ to match the decay rate
    top_prob = sorted_probs[0].item()
    vocab_size = len(probs)
    
    if top_prob > 0:
        # Estimate lambda such that the distribution shape matches observed probs
        # Use the ratio between top probabilities to estimate decay rate
        if sorted_probs[1].item() > 0:
            lambda_param = -torch.log(sorted_probs[1] / sorted_probs[0]).item()
        else:
            lambda_param = -np.log(top_prob * vocab_size)
        lambda_param = max(lambda_param, 0.1)  # Avoid extreme values
    else:
        lambda_param = 1.0
    
    # Create normalized exponential distribution (on same device as probs)
    indices = torch.arange(vocab_size, dtype=torch.float32, device=probs.device)
    exp_unnorm = torch.exp(-lambda_param * indices)
    exp_dist = exp_unnorm / torch.sum(exp_unnorm)
    
    # Reorder to match original probability ordering
    fitted_dist = torch.zeros_like(probs)
    fitted_dist[sorted_indices] = exp_dist
    
    return fitted_dist


def compute_gini_coefficient(probs):
    """
    Compute Gini coefficient for a probability distribution.
    Higher values indicate more inequality (more concentrated distribution).
    """
    # Sort probabilities
    sorted_probs = torch.sort(probs)[0]
    n = len(sorted_probs)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    
    # Gini coefficient formula
    gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    return gini.item() if torch.is_tensor(gini) else gini


def analyze_repeater_context(current_vertex, walk, rules):
    """
    Analyze if current position is near a repeater deadline.
    Returns context information about repeater rules.
    """
    context = {
        'is_repeater_node': False,
        'steps_to_deadline': None,
        'expected_repeater': None,
        'k_value': None
    }
    
    # Find repeater rules
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            # Check if any node in walk is a repeater
            for node in rule.members_nodes_dict:
                if node in walk:
                    # Find last occurrence
                    last_occurrence_idx = None
                    for i in range(len(walk) - 1, -1, -1):
                        if walk[i] == node:
                            last_occurrence_idx = i
                            break
                    
                    if last_occurrence_idx is not None:
                        k = rule.members_nodes_dict[node]
                        steps_since = len(walk) - last_occurrence_idx
                        steps_to_deadline = k - steps_since
                        
                        # Update context if this is the nearest deadline
                        if context['steps_to_deadline'] is None or steps_to_deadline < context['steps_to_deadline']:
                            context['is_repeater_node'] = (current_vertex == node)
                            context['steps_to_deadline'] = steps_to_deadline
                            context['expected_repeater'] = node if steps_to_deadline == 0 else None
                            context['k_value'] = k
    
    return context


def calculate_normalized_metrics(prediction_probs, ref_distributions, predicted_idx):
    """
    Calculate normalized performance metrics comparing model to baselines.
    
    Args:
        prediction_probs: Model's prediction probabilities
        ref_distributions: Dictionary of baseline distributions
        predicted_idx: Index of model's prediction
        
    Returns:
        Dictionary of normalized metrics
    """
    model_prob = prediction_probs[predicted_idx].item()
    model_entropy = entropy_from_logits(torch.log(prediction_probs))
    
    normalized_metrics = {
        'model_confidence': model_prob,
        'model_entropy': model_entropy,
        'model_perplexity': torch.exp(-torch.log(prediction_probs[predicted_idx])).item(),
        'skill_scores': {},
        'information_gains': {},
        'relative_entropies': {},
        'normalized_surprises': {},
        'baseline_comparisons': {}
    }
    
    for baseline_name, baseline_dist in ref_distributions.items():
        baseline_prob = baseline_dist[predicted_idx].item() if baseline_dist[predicted_idx] > 0 else 1e-8
        baseline_entropy = -torch.sum(baseline_dist * torch.log(baseline_dist + 1e-8)).item()
        
        # Skill score: (model_accuracy - baseline_accuracy) / (1 - baseline_accuracy)
        # For single prediction, accuracy is just the probability assigned to the correct token
        if baseline_prob < 1.0:
            skill_score = (model_prob - baseline_prob) / (1.0 - baseline_prob)
        else:
            skill_score = 0.0
        
        # Information gain: entropy reduction from baseline to model
        information_gain = baseline_entropy - model_entropy
        
        # Relative entropy: how much more/less uncertain is model vs baseline
        relative_entropy = model_entropy / baseline_entropy if baseline_entropy > 0 else float('inf')
        
        # Normalized surprise: how much more/less surprised vs baseline
        if baseline_prob > 0:
            baseline_log_prob = -torch.log(torch.tensor(baseline_prob)).item()
            normalized_surprise = -torch.log(prediction_probs[predicted_idx]).item() / baseline_log_prob if baseline_log_prob != 0 else float('inf')
        else:
            normalized_surprise = float('inf')
        
        normalized_metrics['skill_scores'][baseline_name] = skill_score
        normalized_metrics['information_gains'][baseline_name] = information_gain
        normalized_metrics['relative_entropies'][baseline_name] = relative_entropy
        normalized_metrics['normalized_surprises'][baseline_name] = normalized_surprise
        normalized_metrics['baseline_comparisons'][baseline_name] = {
            'baseline_prob': baseline_prob,
            'baseline_entropy': baseline_entropy,
            'prob_ratio': model_prob / baseline_prob if baseline_prob > 0 else float('inf')
        }
    
    return normalized_metrics


def calculate_violation_probabilities(current_vertex, walk, graph, rules, vocab, prediction_probs):
    """Calculate probability mass assigned to rule-violating tokens."""
    violation_probs = {
        'ascender_violation_prob': 0.0,
        'even_violation_prob': 0.0,
        'repeater_violation_prob': 0.0,
        'invalid_edge_prob': 0.0,
        'total_violation_prob': 0.0
    }
    
    # Check each possible next token
    for token, prob in zip(vocab.idx2token.values(), prediction_probs):
        if not token.isdigit():
            continue
            
        node = int(token)
        test_walk = walk + [node]
        
        # Check graph connectivity
        if current_vertex < graph.n and not graph.has_edge(current_vertex, node):
            violation_probs['invalid_edge_prob'] += prob.item()
            violation_probs['total_violation_prob'] += prob.item()
            continue
        
        # Check rule violations
        for rule in rules:
            if not rule.is_satisfied_by(test_walk, graph):
                if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                    violation_probs['ascender_violation_prob'] += prob.item()
                elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
                    violation_probs['even_violation_prob'] += prob.item()
                elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                    violation_probs['repeater_violation_prob'] += prob.item()
                
                violation_probs['total_violation_prob'] += prob.item()
                break  # Don't double count if multiple rules violated
    
    return violation_probs


def analyze_progressive_difficulty(token_level_data):
    """
    Analyze how model performance changes throughout walks (progressive difficulty).
    
    Args:
        token_level_data: List of token-level dictionaries
        
    Returns:
        Dictionary with progressive analysis metrics
    """
    if not token_level_data:
        return {}
    
    # Group by step position
    by_step = {}
    for token_data in token_level_data:
        step = token_data['step_idx']
        if step not in by_step:
            by_step[step] = []
        by_step[step].append(token_data)
    
    progressive_metrics = {
        'by_step_position': {},
        'context_saturation_analysis': {},
        'entropy_trajectory': {},
        'skill_score_progression': {}
    }
    
    # Analyze by step position
    for step, step_data in by_step.items():
        if not step_data:
            continue
            
        step_metrics = {
            'num_samples': len(step_data),
            'avg_entropy': np.mean([d['entropy'] for d in step_data]),
            'avg_confidence': np.mean([d['prediction_confidence'] for d in step_data]),
            'avg_perplexity': np.mean([d['normalized_metrics']['model_perplexity'] for d in step_data]),
            'skill_scores': {},
            'distribution_metrics': {}
        }
        
        # Average skill scores for each baseline
        baseline_names = step_data[0]['normalized_metrics']['skill_scores'].keys()
        for baseline in baseline_names:
            scores = [d['normalized_metrics']['skill_scores'][baseline] for d in step_data if not np.isinf(d['normalized_metrics']['skill_scores'][baseline])]
            step_metrics['skill_scores'][baseline] = np.mean(scores) if scores else 0.0
        
        # Add distribution comparison metrics if available
        if 'core_distribution_comparison' in step_data[0]:
            distribution_samples = [d for d in step_data if 'core_distribution_comparison' in d]
            if distribution_samples:
                step_metrics['distribution_metrics'] = {
                    'avg_kl_from_graph': np.mean([
                        d['core_distribution_comparison']['distribution_distances']['graph_structure']['kl_divergence']
                        for d in distribution_samples
                        if 'graph_structure' in d['core_distribution_comparison']['distribution_distances']
                    ]),
                    'avg_structural_awareness': np.mean([
                        d['core_distribution_comparison']['prediction_quality_scores']['structural_awareness']
                        for d in distribution_samples
                        if 'prediction_quality_scores' in d['core_distribution_comparison']
                    ]),
                    'avg_neighbor_prioritization': np.mean([
                        d['core_distribution_comparison']['prediction_quality_scores']['neighbor_prioritization']
                        for d in distribution_samples
                        if 'prediction_quality_scores' in d['core_distribution_comparison']
                    ]),
                    'avg_overall_quality': np.mean([
                        d['core_distribution_comparison']['prediction_quality_scores']['overall_quality']
                        for d in distribution_samples
                        if 'prediction_quality_scores' in d['core_distribution_comparison']
                    ]),
                    'avg_top1_agreement_with_graph': np.mean([
                        d['core_distribution_comparison']['distribution_overlap_analysis']['graph_structure']['agreement_on_top_k']['top_1']
                        for d in distribution_samples
                        if 'graph_structure' in d['core_distribution_comparison']['distribution_overlap_analysis']
                        and 'agreement_on_top_k' in d['core_distribution_comparison']['distribution_overlap_analysis']['graph_structure']
                    ])
                }
        
        progressive_metrics['by_step_position'][step] = step_metrics
    
    # Analyze context saturation (performance vs context length)
    by_context_length = {}
    for token_data in token_level_data:
        ctx_len = token_data['context_length']
        if ctx_len not in by_context_length:
            by_context_length[ctx_len] = []
        by_context_length[ctx_len].append(token_data)
    
    for ctx_len, ctx_data in by_context_length.items():
        if not ctx_data:
            continue
            
        ctx_metrics = {
            'num_samples': len(ctx_data),
            'avg_entropy': np.mean([d['entropy'] for d in ctx_data]),
            'avg_confidence': np.mean([d['prediction_confidence'] for d in ctx_data])
        }
        progressive_metrics['context_saturation_analysis'][ctx_len] = ctx_metrics
    
    # Calculate moving averages for entropy trajectory
    if len(by_step) > 1:
        sorted_steps = sorted(by_step.keys())
        entropies = []
        confidences = []
        for step in sorted_steps:
            step_data = by_step[step]
            entropies.append(np.mean([d['entropy'] for d in step_data]))
            confidences.append(np.mean([d['prediction_confidence'] for d in step_data]))
        
        progressive_metrics['entropy_trajectory'] = {
            'steps': sorted_steps,
            'entropies': entropies,
            'confidences': confidences,
            'entropy_trend': np.polyfit(sorted_steps, entropies, 1)[0] if len(sorted_steps) > 1 else 0.0,
            'confidence_trend': np.polyfit(sorted_steps, confidences, 1)[0] if len(sorted_steps) > 1 else 0.0
        }
    
    return progressive_metrics


def select_exemplar_walks(evaluation_results, token_level_data, progressive_analysis, num_exemplars=10):
    """
    Select interesting walk exemplars for paper documentation.
    
    Args:
        evaluation_results: List of walk evaluation results
        token_level_data: Token-level analysis data
        progressive_analysis: Progressive difficulty analysis
        num_exemplars: Number of exemplars to select for each category
        
    Returns:
        Dictionary of exemplar walks organized by interesting characteristics
    """
    exemplars = {
        'perfect_rule_compliance': [],
        'rule_violations': [],
        'context_window_failures': [],
        'high_confidence_predictions': [],
        'low_confidence_predictions': [],
        'entropy_progression_examples': [],
        'baseline_comparison_examples': [],
        'repeater_learning_examples': [],
        'repeater_failure_examples': []
    }
    
    # Group token data by walk
    walks_data = {}
    for token in token_level_data:
        walk_idx = token['walk_idx']
        if walk_idx not in walks_data:
            walks_data[walk_idx] = []
        walks_data[walk_idx].append(token)
    
    # Analyze each walk for exemplar selection
    walk_analyses = []
    for walk_idx, walk_result in enumerate(evaluation_results):
        if walk_idx >= len(walks_data):
            continue
            
        walk_tokens = walks_data[walk_idx]
        if not walk_tokens:
            continue
            
        # Calculate walk-level metrics
        analysis = analyze_walk_for_exemplars(walk_result, walk_tokens, walk_idx)
        walk_analyses.append(analysis)
    
    # Sort and select exemplars for each category
    
    # Perfect rule compliance (no violations, high confidence)
    perfect_walks = [w for w in walk_analyses if w['total_violations'] == 0]
    perfect_walks.sort(key=lambda x: x['avg_confidence'], reverse=True)
    exemplars['perfect_rule_compliance'] = perfect_walks[:num_exemplars]
    
    # Rule violations (sorted by violation count)
    violation_walks = [w for w in walk_analyses if w['total_violations'] > 0]
    violation_walks.sort(key=lambda x: x['total_violations'], reverse=True)
    exemplars['rule_violations'] = violation_walks[:num_exemplars]
    
    # Context window failures (violations likely due to exceeding context)
    context_failures = [w for w in walk_analyses if w.get('likely_context_failure', False)]
    context_failures.sort(key=lambda x: x['max_step_reached'], reverse=True)
    exemplars['context_window_failures'] = context_failures[:num_exemplars]
    
    # High confidence predictions
    all_walks = walk_analyses[:]
    all_walks.sort(key=lambda x: x['avg_confidence'], reverse=True)
    exemplars['high_confidence_predictions'] = all_walks[:num_exemplars]
    
    # Low confidence predictions
    all_walks.sort(key=lambda x: x['avg_confidence'])
    exemplars['low_confidence_predictions'] = all_walks[:num_exemplars]
    
    # Entropy progression examples (walks showing clear entropy trends)
    entropy_walks = [w for w in walk_analyses if abs(w.get('entropy_slope', 0)) > 0.1]
    entropy_walks.sort(key=lambda x: abs(x.get('entropy_slope', 0)), reverse=True)
    exemplars['entropy_progression_examples'] = entropy_walks[:num_exemplars]
    
    # Baseline comparison examples (good skill scores)
    skill_walks = [w for w in walk_analyses if w.get('avg_skill_score', 0) > 0.5]
    skill_walks.sort(key=lambda x: x.get('avg_skill_score', 0), reverse=True)
    exemplars['baseline_comparison_examples'] = skill_walks[:num_exemplars]
    
    # Repeater learning examples (successful repeater rule following)
    repeater_success = [w for w in walk_analyses if w.get('repeater_success', False)]
    repeater_success.sort(key=lambda x: x.get('repeater_steps_correct', 0), reverse=True)
    exemplars['repeater_learning_examples'] = repeater_success[:num_exemplars]
    
    # Repeater failure examples
    repeater_failure = [w for w in walk_analyses if w.get('repeater_violations', 0) > 0]
    repeater_failure.sort(key=lambda x: x.get('repeater_violations', 0), reverse=True)
    exemplars['repeater_failure_examples'] = repeater_failure[:num_exemplars]
    
    return exemplars


def analyze_walk_for_exemplars(walk_result, walk_tokens, walk_idx):
    """
    Analyze a single walk to determine its characteristics for exemplar selection.
    
    Args:
        walk_result: Single walk result from evaluation_results
        walk_tokens: Token-level data for this walk
        walk_idx: Walk index
        
    Returns:
        Dictionary with walk analysis metrics
    """
    analysis = {
        'walk_idx': walk_idx,
        'start_walk': walk_result['start_walk'],
        'generated_walk': walk_result['generated_walk'],
        'walk_length': walk_result['walk_length'],
        'num_generation_steps': walk_result['num_generation_steps'],
        'max_step_reached': max([t['step_idx'] for t in walk_tokens]) if walk_tokens else 0
    }
    
    if not walk_tokens:
        return analysis
    
    # Calculate aggregate metrics
    confidences = [t['prediction_confidence'] for t in walk_tokens]
    entropies = [t['entropy'] for t in walk_tokens]
    
    analysis.update({
        'avg_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'confidence_std': np.std(confidences),
        'avg_entropy': np.mean(entropies),
        'entropy_std': np.std(entropies)
    })
    
    # Entropy trend analysis
    if len(entropies) > 1:
        steps = list(range(len(entropies)))
        analysis['entropy_slope'] = np.polyfit(steps, entropies, 1)[0]
    else:
        analysis['entropy_slope'] = 0.0
    
    # Rule violation analysis
    total_violations = 0
    repeater_violations = 0
    context_related_violations = 0
    
    for token in walk_tokens:
        if token['rule_violations']['has_violation']:
            total_violations += 1
            if 'repeater' in token['rule_violations']['violation_types']:
                repeater_violations += 1
            # Check if violation occurs after context window size
            if token['step_idx'] >= 5:  # Assuming context window of 5
                context_related_violations += 1
    
    analysis.update({
        'total_violations': total_violations,
        'repeater_violations': repeater_violations,
        'context_related_violations': context_related_violations,
        'likely_context_failure': context_related_violations > 0 and context_related_violations >= total_violations * 0.5
    })
    
    # Skill score analysis
    if walk_tokens and 'normalized_metrics' in walk_tokens[0]:
        skill_scores = []
        for token in walk_tokens:
            for baseline, score in token['normalized_metrics']['skill_scores'].items():
                if not np.isinf(score) and not np.isnan(score):
                    skill_scores.append(score)
        
        if skill_scores:
            analysis['avg_skill_score'] = np.mean(skill_scores)
            analysis['max_skill_score'] = np.max(skill_scores)
        else:
            analysis['avg_skill_score'] = 0.0
            analysis['max_skill_score'] = 0.0
    
    # Repeater-specific analysis
    repeater_steps_correct = 0
    repeater_steps_total = 0
    
    for token in walk_tokens:
        if 'repeater_context' in token and token['repeater_context']['is_repeater_node']:
            repeater_steps_total += 1
            if token['repeater_context']['steps_to_deadline'] == 0:
                # This should be a repeater step
                if not token['rule_violations']['has_violation']:
                    repeater_steps_correct += 1
    
    analysis.update({
        'repeater_steps_correct': repeater_steps_correct,
        'repeater_steps_total': repeater_steps_total,
        'repeater_success': repeater_steps_total > 0 and repeater_steps_correct == repeater_steps_total
    })
    
    return analysis


def compute_ks_distance(p_probs, q_probs):
    """
    Compute Kolmogorov-Smirnov distance between two probability distributions.
    KS distance is the maximum difference between the cumulative distribution functions.
    
    Args:
        p_probs: First probability distribution (numpy array or tensor)
        q_probs: Second probability distribution (numpy array or tensor)
        
    Returns:
        KS distance (float)
    """
    # Convert to numpy if needed
    if torch.is_tensor(p_probs):
        p_probs = p_probs.cpu().detach().numpy()
    if torch.is_tensor(q_probs):
        q_probs = q_probs.cpu().detach().numpy()
    
    # Compute CDFs
    p_cdf = np.cumsum(p_probs)
    q_cdf = np.cumsum(q_probs)
    
    # KS distance is maximum difference between CDFs
    ks_distance = np.max(np.abs(p_cdf - q_cdf))
    
    return float(ks_distance)


def compute_js_divergence(p_probs, q_probs):
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    JS divergence is a symmetric version of KL divergence.
    
    Args:
        p_probs: First probability distribution
        q_probs: Second probability distribution
        
    Returns:
        JS divergence (float)
    """
    # Convert to numpy if needed
    if torch.is_tensor(p_probs):
        p_probs = p_probs.cpu().detach().numpy()
    if torch.is_tensor(q_probs):
        q_probs = q_probs.cpu().detach().numpy()
    
    # Compute average distribution
    m = 0.5 * (p_probs + q_probs)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_probs = p_probs + epsilon
    q_probs = q_probs + epsilon
    m = m + epsilon
    
    # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_pm = np.sum(p_probs * np.log(p_probs / m))
    kl_qm = np.sum(q_probs * np.log(q_probs / m))
    js_div = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(js_div)


def compute_wasserstein_distance(p_probs, q_probs):
    """
    Compute Wasserstein distance (Earth Mover's Distance) between two probability distributions.
    Assumes the support points are ordered (e.g., vocabulary indices).
    
    Args:
        p_probs: First probability distribution
        q_probs: Second probability distribution
        
    Returns:
        Wasserstein distance (float) or None if scipy not available
    """
    try:
        from scipy.stats import wasserstein_distance
        
        # Convert to numpy if needed
        if torch.is_tensor(p_probs):
            p_probs = p_probs.cpu().detach().numpy()
        if torch.is_tensor(q_probs):
            q_probs = q_probs.cpu().detach().numpy()
        
        # Assume support points are indices 0, 1, 2, ...
        support = np.arange(len(p_probs))
        
        # Compute Wasserstein distance
        w_dist = wasserstein_distance(support, support, p_probs, q_probs)
        
        return float(w_dist)
    except ImportError:
        return None


def compute_distribution_distances(prediction_probs, ref_distributions, vocab=None):
    """
    Compute various distribution distance metrics between model predictions and baselines.
    
    Args:
        prediction_probs: Model's full probability distribution over vocab (tensor)
        ref_distributions: Dictionary of baseline distributions
        vocab: Vocabulary object (optional)
        
    Returns:
        Dictionary of distance metrics for each baseline
    """
    distances = {
        'kl_divergences': {},
        'ks_distances': {},
        'js_divergences': {},
        'wasserstein_distances': {}
    }
    
    for baseline_name, baseline_dist in ref_distributions.items():
        # Ensure both are on same device
        if baseline_dist.device != prediction_probs.device:
            baseline_dist = baseline_dist.to(prediction_probs.device)
        
        # KL Divergence (existing function)
        distances['kl_divergences'][baseline_name] = kl_divergence(prediction_probs, baseline_dist)
        
        # KS Distance
        distances['ks_distances'][baseline_name] = compute_ks_distance(prediction_probs, baseline_dist)
        
        # JS Divergence
        distances['js_divergences'][baseline_name] = compute_js_divergence(prediction_probs, baseline_dist)
        
        # Wasserstein Distance (optional, requires scipy)
        w_dist = compute_wasserstein_distance(prediction_probs, baseline_dist)
        if w_dist is not None:
            distances['wasserstein_distances'][baseline_name] = w_dist
    
    return distances


def analyze_probability_distribution(prediction_probs, vocab, graph, current_vertex):
    """
    Comprehensive analysis of the probability distribution at a single step.
    
    Args:
        prediction_probs: Model's probability distribution (tensor)
        vocab: Vocabulary object
        graph: Graph object
        current_vertex: Current position in walk
        
    Returns:
        Dictionary with distribution statistics and node-level analysis
    """
    probs_np = prediction_probs.cpu().detach().numpy()
    
    # Sort probabilities for top-k analysis
    sorted_probs = np.sort(probs_np)[::-1]  # Descending order
    
    analysis = {
        # Distribution shape metrics
        'entropy': float(-np.sum(probs_np * np.log(probs_np + 1e-10))),
        'mode_probability': float(np.max(probs_np)),
        'mode_index': int(np.argmax(probs_np)),
        
        # Concentration metrics
        'top_1_mass': float(sorted_probs[0]),
        'top_5_mass': float(np.sum(sorted_probs[:5])),
        'top_10_mass': float(np.sum(sorted_probs[:10])),
        
        # Support size (nodes with meaningful probability)
        'effective_support_size': int(np.sum(probs_np > 0.001)),
        'high_confidence_nodes': int(np.sum(probs_np > 0.1)),
        
        # Tail behavior
        'tail_mass': float(np.sum(sorted_probs[10:])) if len(sorted_probs) > 10 else 0.0,
        'min_nonzero_prob': float(np.min(probs_np[probs_np > 0])) if np.any(probs_np > 0) else 0.0,
        
        # Graph-aware metrics
        'valid_neighbor_mass': 0.0,
        'invalid_edge_mass': 0.0,
        'rule_compliant_mass': 0.0,
        'rule_violating_mass': 0.0
    }
    
    # Calculate graph-aware probability masses
    if current_vertex < graph.n:
        valid_neighbors = set(graph.get_neighbors(current_vertex))
        
        for idx, prob in enumerate(probs_np):
            if vocab and idx < len(vocab.idx2token):
                token = vocab.idx2token[idx]
            else:
                token = str(idx)
            
            if token.isdigit():
                node = int(token)
                
                # Valid vs invalid edges
                if node in valid_neighbors:
                    analysis['valid_neighbor_mass'] += float(prob)
                else:
                    analysis['invalid_edge_mass'] += float(prob)
    
    # Normalize masses to ensure they sum correctly
    total_graph_mass = analysis['valid_neighbor_mass'] + analysis['invalid_edge_mass']
    if total_graph_mass > 1.0:
        analysis['valid_neighbor_mass'] /= total_graph_mass
        analysis['invalid_edge_mass'] /= total_graph_mass
    
    return analysis


def get_large_scale_trajectory_config(num_walks, sample_rate=0.02, stratified=True):
    """
    Generate trajectory sampling configuration optimized for large-scale experiments.
    
    Args:
        num_walks: Total number of walks to be generated
        sample_rate: Fraction of walks to store full trajectory data for
        stratified: Whether to use stratified sampling by termination outcome
        
    Returns:
        Dictionary with trajectory sampling configuration
    """
    # Calculate minimum samples to ensure representation of all outcomes
    min_samples_per_outcome = max(100, int(num_walks * sample_rate / 8))  # Distribute across ~8 outcomes
    
    config = {
        "enabled": True,
        "sample_rate": sample_rate,
        "stratified": stratified,
        "min_samples_per_outcome": min_samples_per_outcome,
        "store_full_distributions": sample_rate <= 0.05,  # Only store full distributions for small sample rates
        "memory_efficient": True,
        "description": f"Large-scale sampling: {sample_rate:.1%} of {num_walks:,} walks = {int(num_walks * sample_rate):,} full trajectories"
    }
    
    return config


def compute_core_distribution_comparison(prediction_probs, current_vertex, graph, vocab, device, rules=None, walk_history=None):
    """
    Enhanced comparison of LLM predictions against baseline distributions including oracle baselines.
    Baselines include:
    1. Graph structure distribution (the "true" edge probabilities)
    2. Uniform distribution (over valid neighbors)
    3. Exponential distribution (fitted to model predictions)
    4. Rule-aware oracle (knows optimal rule-compliant moves)
    5. Optimal path oracle (assigns 1.0 to single best move)
    
    Args:
        prediction_probs: Model's prediction probabilities (tensor)
        current_vertex: Current position in walk
        graph: Graph object with edge probabilities
        vocab: Vocabulary object
        device: PyTorch device
        rules: List of rule objects (for oracle baselines)
        walk_history: Walk history (for repeater oracle)
        
    Returns:
        Dictionary with detailed comparison metrics
    """
    vocab_size = len(vocab.token2idx)
    
    # Create the core baseline distributions
    baselines = {}
    
    # 1. Graph structure distribution (the "ground truth" for this graph)
    baselines['graph_structure'] = graph_edge_distribution(current_vertex, graph, vocab).to(device)
    
    # 2. Uniform distribution over valid neighbors
    baselines['uniform_valid'] = valid_neighbors_baseline(current_vertex, graph, vocab).to(device)
    
    # 3. Exponential distribution fitted to model's predictions
    baselines['exponential_fitted'] = fit_exponential_mle(prediction_probs)
    
    # 4. Full uniform (over all vocab) for reference
    baselines['uniform_full'] = uniform_random_baseline(vocab_size).to(device)
    
    # 5. Oracle baselines (if rules provided)
    if rules is not None:
        # Rule-aware oracle: knows optimal rule-compliant behavior
        baselines['rule_aware_oracle'] = rule_aware_oracle_baseline(
            current_vertex, graph, vocab, rules, walk_history or []
        ).to(device)
        
        # Optimal path oracle: probability 1.0 for single best move
        baselines['optimal_path_oracle'] = optimal_path_baseline(
            current_vertex, graph, vocab, rules, walk_history or []
        ).to(device)
        
        # Individual rule oracles for detailed analysis
        baselines['repeater_oracle'] = repeater_oracle_baseline(
            current_vertex, graph, vocab, rules, walk_history or []
        ).to(device)
        
        baselines['ascender_oracle'] = ascender_oracle_baseline(
            current_vertex, graph, vocab, rules
        ).to(device)
        
        baselines['even_oracle'] = even_oracle_baseline(
            current_vertex, graph, vocab, rules
        ).to(device)
    
    comparison = {
        'model_distribution_stats': analyze_model_distribution(prediction_probs),
        'baseline_distributions': {},
        'distribution_distances': {},
        'relative_performance': {},
        'distribution_overlap_analysis': {},
        'prediction_quality_scores': {}
    }
    
    # Analyze each baseline distribution
    for baseline_name, baseline_dist in baselines.items():
        comparison['baseline_distributions'][baseline_name] = {
            'entropy': -torch.sum(baseline_dist * torch.log(baseline_dist + 1e-8)).item(),
            'max_prob': torch.max(baseline_dist).item(),
            'effective_support': torch.sum(baseline_dist > 0.001).item(),
            'valid_neighbor_mass': compute_valid_neighbor_mass(baseline_dist, current_vertex, graph, vocab)
        }
        
        # Compute distances between model and this baseline
        comparison['distribution_distances'][baseline_name] = {
            'kl_divergence': kl_divergence(torch.log(prediction_probs), baseline_dist),
            'reverse_kl': kl_divergence(torch.log(baseline_dist + 1e-8), prediction_probs),
            'js_divergence': compute_js_divergence(prediction_probs, baseline_dist),
            'ks_distance': compute_ks_distance(prediction_probs, baseline_dist),
            'l1_distance': torch.sum(torch.abs(prediction_probs - baseline_dist)).item(),
            'l2_distance': torch.sqrt(torch.sum((prediction_probs - baseline_dist)**2)).item(),
            'cosine_similarity': compute_cosine_similarity(prediction_probs, baseline_dist)
        }
        
        # Analyze distribution overlap
        comparison['distribution_overlap_analysis'][baseline_name] = analyze_distribution_overlap(
            prediction_probs, baseline_dist, current_vertex, graph, vocab
        )
    
    # Compute relative performance metrics
    comparison['relative_performance'] = compute_relative_performance_metrics(
        prediction_probs, baselines, current_vertex, graph, vocab
    )
    
    # Prediction quality assessment
    comparison['prediction_quality_scores'] = assess_prediction_quality(
        prediction_probs, baselines, current_vertex, graph, vocab
    )
    
    return comparison


def analyze_model_distribution(prediction_probs):
    """Analyze key characteristics of the model's prediction distribution."""
    probs_np = prediction_probs.cpu().detach().numpy()
    sorted_probs = np.sort(probs_np)[::-1]
    
    return {
        'entropy': float(-torch.sum(prediction_probs * torch.log(prediction_probs + 1e-8)).item()),
        'max_probability': float(torch.max(prediction_probs).item()),
        'top_5_mass': float(np.sum(sorted_probs[:5])),
        'top_10_mass': float(np.sum(sorted_probs[:10])),
        'effective_support_size': int(torch.sum(prediction_probs > 0.001).item()),
        'gini_coefficient': compute_gini_coefficient(prediction_probs),
        'concentration_ratio': float(sorted_probs[0] / (sorted_probs[1] + 1e-8)),  # How concentrated vs second choice
        'tail_mass': float(np.sum(sorted_probs[10:])) if len(sorted_probs) > 10 else 0.0
    }


def compute_valid_neighbor_mass(distribution, current_vertex, graph, vocab):
    """Compute what fraction of probability mass is on valid graph neighbors."""
    if current_vertex >= graph.n:
        return 0.0
    
    valid_neighbors = set(graph.get_neighbors(current_vertex))
    valid_mass = 0.0
    
    for neighbor in valid_neighbors:
        neighbor_token = str(neighbor)
        if neighbor_token in vocab.token2idx:
            neighbor_idx = vocab.token2idx[neighbor_token]
            if neighbor_idx < len(distribution):
                valid_mass += distribution[neighbor_idx].item()
    
    return float(valid_mass)


def compute_cosine_similarity(dist1, dist2):
    """Compute cosine similarity between two probability distributions."""
    dot_product = torch.sum(dist1 * dist2)
    norm1 = torch.sqrt(torch.sum(dist1 ** 2))
    norm2 = torch.sqrt(torch.sum(dist2 ** 2))
    
    if norm1 > 0 and norm2 > 0:
        return (dot_product / (norm1 * norm2)).item()
    else:
        return 0.0


def analyze_distribution_overlap(model_probs, baseline_probs, current_vertex, graph, vocab):
    """Analyze how much the model and baseline distributions overlap."""
    analysis = {
        'overlap_coefficient': 0.0,  # Sum of min(p_model, p_baseline)
        'agreement_on_top_k': {},    # Do they agree on top k choices?
        'valid_neighbor_agreement': 0.0,  # Agreement specifically on valid neighbors
        'invalid_edge_disagreement': 0.0   # How much does model put on invalid edges vs baseline
    }
    
    # Overlap coefficient (Bhattacharyya coefficient)
    analysis['overlap_coefficient'] = torch.sum(torch.min(model_probs, baseline_probs)).item()
    
    # Top-k agreement
    for k in [1, 3, 5, 10]:
        model_top_k = torch.topk(model_probs, min(k, len(model_probs)))[1]
        baseline_top_k = torch.topk(baseline_probs, min(k, len(baseline_probs)))[1]
        
        # Convert to sets and compute intersection
        model_set = set(model_top_k.cpu().detach().numpy())
        baseline_set = set(baseline_top_k.cpu().detach().numpy())
        intersection_size = len(model_set.intersection(baseline_set))
        
        analysis['agreement_on_top_k'][f'top_{k}'] = intersection_size / k
    
    # Valid neighbor analysis
    if current_vertex < graph.n:
        valid_neighbors = set(graph.get_neighbors(current_vertex))
        valid_model_mass = 0.0
        valid_baseline_mass = 0.0
        valid_overlap = 0.0
        
        invalid_model_mass = 0.0
        invalid_baseline_mass = 0.0
        
        for idx, (m_prob, b_prob) in enumerate(zip(model_probs, baseline_probs)):
            if idx < len(vocab.idx2token):
                token = vocab.idx2token[idx]
                if token.isdigit():
                    node = int(token)
                    if node in valid_neighbors:
                        valid_model_mass += m_prob.item()
                        valid_baseline_mass += b_prob.item()
                        valid_overlap += min(m_prob.item(), b_prob.item())
                    else:
                        invalid_model_mass += m_prob.item()
                        invalid_baseline_mass += b_prob.item()
        
        # Agreement on valid neighbors (how similar are they when restricted to valid neighbors)
        if valid_model_mass > 0 and valid_baseline_mass > 0:
            analysis['valid_neighbor_agreement'] = valid_overlap / min(valid_model_mass, valid_baseline_mass)
        
        # Disagreement on invalid edges
        analysis['invalid_edge_disagreement'] = abs(invalid_model_mass - invalid_baseline_mass)
    
    return analysis


def compute_relative_performance_metrics(model_probs, baselines, current_vertex, graph, vocab):
    """Compute how model performs relative to each baseline."""
    metrics = {}
    
    # For each baseline, compute relative metrics
    for baseline_name, baseline_dist in baselines.items():
        baseline_metrics = {
            'entropy_improvement': 0.0,        # Lower entropy = more confident
            'concentration_improvement': 0.0,   # Higher max prob = more decisive
            'valid_neighbor_focus': 0.0,       # Better focus on valid neighbors
            'information_efficiency': 0.0      # Information gain per bit
        }
        
        # Entropy comparison
        model_entropy = -torch.sum(model_probs * torch.log(model_probs + 1e-8)).item()
        baseline_entropy = -torch.sum(baseline_dist * torch.log(baseline_dist + 1e-8)).item()
        
        if baseline_entropy > 0:
            baseline_metrics['entropy_improvement'] = (baseline_entropy - model_entropy) / baseline_entropy
        
        # Concentration comparison
        model_max = torch.max(model_probs).item()
        baseline_max = torch.max(baseline_dist).item()
        
        if baseline_max > 0:
            baseline_metrics['concentration_improvement'] = (model_max - baseline_max) / baseline_max
        
        # Valid neighbor focus
        model_valid_mass = compute_valid_neighbor_mass(model_probs, current_vertex, graph, vocab)
        baseline_valid_mass = compute_valid_neighbor_mass(baseline_dist, current_vertex, graph, vocab)
        
        if baseline_valid_mass > 0:
            baseline_metrics['valid_neighbor_focus'] = (model_valid_mass - baseline_valid_mass) / baseline_valid_mass
        
        # Information efficiency (how much information per unit of entropy)
        if model_entropy > 0:
            baseline_metrics['information_efficiency'] = model_valid_mass / model_entropy
        
        metrics[baseline_name] = baseline_metrics
    
    return metrics


def assess_prediction_quality(model_probs, baselines, current_vertex, graph, vocab):
    """Assess overall prediction quality compared to baselines."""
    quality_scores = {}
    
    # Quality dimensions to assess
    dimensions = {
        'structural_awareness': 0.0,    # How well does it respect graph structure
        'concentration_quality': 0.0,   # Appropriate confidence level
        'distributional_fit': 0.0,     # How well does it match expected distributions
        'neighbor_prioritization': 0.0  # Focus on reachable nodes
    }
    
    # Structural awareness: Compare with graph structure baseline
    if 'graph_structure' in baselines:
        graph_dist = baselines['graph_structure']
        structural_similarity = compute_cosine_similarity(model_probs, graph_dist)
        dimensions['structural_awareness'] = max(0.0, structural_similarity)
    
    # Concentration quality: Not too uniform, not too concentrated
    model_entropy = -torch.sum(model_probs * torch.log(model_probs + 1e-8)).item()
    model_max = torch.max(model_probs).item()
    
    # Good concentration: decisive but not overconfident
    # Penalize both extreme uniformity and extreme concentration
    uniformity_penalty = abs(model_entropy - np.log(len(model_probs))) / np.log(len(model_probs))
    concentration_penalty = model_max if model_max > 0.8 else 0.0
    dimensions['concentration_quality'] = 1.0 - 0.5 * (uniformity_penalty + concentration_penalty)
    
    # Distributional fit: How close to reasonable baselines
    if 'uniform_valid' in baselines and 'exponential_fitted' in baselines:
        uniform_sim = compute_cosine_similarity(model_probs, baselines['uniform_valid'])
        exp_sim = compute_cosine_similarity(model_probs, baselines['exponential_fitted'])
        dimensions['distributional_fit'] = max(uniform_sim, exp_sim)
    
    # Neighbor prioritization: Focus on valid graph neighbors
    dimensions['neighbor_prioritization'] = compute_valid_neighbor_mass(model_probs, current_vertex, graph, vocab)
    
    # Compute overall quality score (weighted average)
    weights = {
        'structural_awareness': 0.4,
        'concentration_quality': 0.2,
        'distributional_fit': 0.2,
        'neighbor_prioritization': 0.2
    }
    
    overall_score = sum(dimensions[dim] * weights[dim] for dim in dimensions)
    
    quality_scores.update(dimensions)
    quality_scores['overall_quality'] = overall_score
    
    return quality_scores


def estimate_trajectory_memory_usage(num_walks, trajectory_config, vocab_size=1000, avg_steps_per_walk=20):
    """
    Estimate memory usage for trajectory storage.
    
    Args:
        num_walks: Total number of walks
        trajectory_config: Trajectory sampling configuration
        vocab_size: Size of vocabulary
        avg_steps_per_walk: Average number of steps per walk
        
    Returns:
        Dictionary with memory estimates in MB
    """
    sample_rate = trajectory_config.get("sample_rate", 1.0)
    store_full_distributions = trajectory_config.get("store_full_distributions", True)
    
    # Number of walks storing full trajectory data
    full_trajectory_walks = int(num_walks * sample_rate)
    summary_only_walks = num_walks - full_trajectory_walks
    
    estimates = {
        # Full trajectory storage
        "full_trajectories_mb": 0,
        "summary_trajectories_mb": 0,
        "total_mb": 0
    }
    
    if store_full_distributions:
        # Each probability distribution: vocab_size * 4 bytes (float32)
        dist_size_bytes = vocab_size * 4
        # Each full trajectory: avg_steps * dist_size + metadata
        full_traj_size = (avg_steps_per_walk * dist_size_bytes + 1024) / (1024 * 1024)  # Convert to MB
        estimates["full_trajectories_mb"] = full_trajectory_walks * full_traj_size
    
    # Summary trajectories: only metadata and statistics (~1KB each)
    summary_traj_size = 1024 / (1024 * 1024)  # 1KB in MB
    estimates["summary_trajectories_mb"] = num_walks * summary_traj_size  # All walks have summary
    
    estimates["total_mb"] = estimates["full_trajectories_mb"] + estimates["summary_trajectories_mb"]
    
    return estimates


def extract_violation_time_series(token_level_data, lookback_window=20, max_cases_per_rule=10, min_violation_confidence=0.7):
    """
    Extract time series of entropy metrics leading up to rule violations.
    
    Identifies rule violation tokens and extracts the preceding sequence of entropy
    metrics to analyze how uncertainty patterns change before rule-breaking decisions.
    
    Args:
        token_level_data: List of token-level dictionaries from evaluation
        lookback_window: Number of tokens before violation to include in series
        max_cases_per_rule: Maximum violation cases to extract per rule type
        min_violation_confidence: Minimum model confidence required for violation
        
    Returns:
        Dictionary containing violation time series data organized by rule type
    """
    violation_time_series = {
        'repeater_violations': [],
        'ascender_violations': [],
        'even_violations': [],
        'mixed_violations': [],
        'metadata': {
            'lookback_window': lookback_window,
            'total_violations_found': 0,
            'cases_extracted': 0,
            'violation_confidence_threshold': min_violation_confidence
        }
    }
    
    # Group token data by walk to enable lookback analysis
    walk_data = {}
    for token in token_level_data:
        walk_idx = token.get('walk_idx', 0)
        if walk_idx not in walk_data:
            walk_data[walk_idx] = []
        walk_data[walk_idx].append(token)
    
    # Sort tokens within each walk by step index
    for walk_idx in walk_data:
        walk_data[walk_idx].sort(key=lambda x: x.get('step_idx', 0))
    
    # Extract violation cases from each walk
    for walk_idx, walk_tokens in walk_data.items():
        for i, token in enumerate(walk_tokens):
            # Check if this token represents a rule violation
            if not token.get('rule_violations', {}).get('has_violation', False):
                continue
            
            # Check if violation meets confidence threshold
            confidence = token.get('prediction_confidence', 0.0)
            if confidence < min_violation_confidence:
                continue
            
            violation_time_series['metadata']['total_violations_found'] += 1
            
            # Ensure we have enough lookback data
            if i < lookback_window:
                continue  # Not enough preceding tokens for full time series
            
            # Extract the violation time series
            violation_case = extract_single_violation_case(
                walk_tokens[i-lookback_window:i+1],  # Include violation token
                violation_token_idx=lookback_window,  # Position of violation in series
                lookback_window=lookback_window
            )
            
            if violation_case is None:
                continue
            
            # Categorize by rule type
            violation_types = token['rule_violations'].get('violation_types', [])
            
            # Determine primary violation category for this case
            if 'repeater' in violation_types:
                category = 'repeater_violations'
            elif 'ascender' in violation_types:
                category = 'ascender_violations'
            elif 'even' in violation_types:
                category = 'even_violations'
            else:
                category = 'mixed_violations'
            
            # Check if we have room for more cases in this category
            if len(violation_time_series[category]) >= max_cases_per_rule:
                continue
            
            violation_time_series[category].append(violation_case)
            violation_time_series['metadata']['cases_extracted'] += 1
    
    # Add summary statistics
    for category in ['repeater_violations', 'ascender_violations', 'even_violations', 'mixed_violations']:
        violation_time_series['metadata'][f'{category}_count'] = len(violation_time_series[category])
    
    return violation_time_series


def extract_single_violation_case(token_sequence, violation_token_idx, lookback_window):
    """
    Extract entropy metrics time series for a single violation case.
    
    Args:
        token_sequence: List of token dictionaries including the violation
        violation_token_idx: Index of the violation token in the sequence
        lookback_window: Number of preceding tokens to analyze
        
    Returns:
        Dictionary with time series data for this violation case
    """
    if len(token_sequence) != lookback_window + 1:
        return None
    
    violation_token = token_sequence[violation_token_idx]
    
    # Initialize time series data
    case_data = {
        'walk_idx': violation_token.get('walk_idx'),
        'violation_step': violation_token.get('step_idx'),
        'violation_types': violation_token.get('rule_violations', {}).get('violation_types', []),
        'violation_confidence': violation_token.get('prediction_confidence', 0.0),
        'context_length': violation_token.get('context_length', 0),
        
        # Time series arrays (index 0 = earliest, index -1 = violation)
        'time_steps': list(range(-lookback_window, 1)),  # -20, -19, ..., -1, 0
        'model_entropy': [],
        'prediction_confidence': [],
        'kl_divergences': {},  # Will hold time series for each baseline
        'cross_entropy': {},
        'mutual_information': {},
        'information_gain': {},
        'violation_probability_mass': [],
        'entropy_rate': [],
        'context_tokens': []
    }
    
    # Extract baseline names from first token with distribution comparison
    baseline_names = []
    for token in token_sequence:
        if 'core_distribution_comparison' in token:
            distances = token['core_distribution_comparison'].get('distribution_distances', {})
            if distances:
                baseline_names = list(distances.keys())
                break
    
    # Initialize baseline time series
    for baseline in baseline_names:
        case_data['kl_divergences'][baseline] = []
        case_data['cross_entropy'][baseline] = []
        case_data['mutual_information'][baseline] = []
        case_data['information_gain'][baseline] = []
    
    # Extract time series data from each token
    for token in token_sequence:
        # Basic metrics
        case_data['model_entropy'].append(token.get('entropy', 0.0))
        case_data['prediction_confidence'].append(token.get('prediction_confidence', 0.0))
        case_data['context_tokens'].append(token.get('predicted_vertex', ''))
        
        # Calculate entropy rate (rate of change in entropy)
        if len(case_data['model_entropy']) > 1:
            entropy_rate = case_data['model_entropy'][-1] - case_data['model_entropy'][-2]
            case_data['entropy_rate'].append(entropy_rate)
        else:
            case_data['entropy_rate'].append(0.0)
        
        # Extract baseline comparisons
        if 'core_distribution_comparison' in token:
            distances = token['core_distribution_comparison'].get('distribution_distances', {})
            
            for baseline in baseline_names:
                if baseline in distances:
                    baseline_data = distances[baseline]
                    case_data['kl_divergences'][baseline].append(
                        baseline_data.get('kl_divergence', 0.0))
                    case_data['cross_entropy'][baseline].append(
                        baseline_data.get('cross_entropy', 0.0))
                    case_data['mutual_information'][baseline].append(
                        baseline_data.get('mutual_information', 0.0))
                    case_data['information_gain'][baseline].append(
                        baseline_data.get('information_gain', 0.0))
                else:
                    # Fill missing data with zeros
                    case_data['kl_divergences'][baseline].append(0.0)
                    case_data['cross_entropy'][baseline].append(0.0)
                    case_data['mutual_information'][baseline].append(0.0)
                    case_data['information_gain'][baseline].append(0.0)
        else:
            # Fill missing baseline data with zeros
            for baseline in baseline_names:
                case_data['kl_divergences'][baseline].append(0.0)
                case_data['cross_entropy'][baseline].append(0.0)
                case_data['mutual_information'][baseline].append(0.0)
                case_data['information_gain'][baseline].append(0.0)
        
        # Extract rule violation probability mass if available
        rule_violations = token.get('rule_violations', {})
        violation_probs = rule_violations.get('violation_probabilities', {})
        total_violation_mass = sum(violation_probs.values())
        case_data['violation_probability_mass'].append(total_violation_mass)
    
    # Add violation context information
    case_data['violation_context'] = {
        'predicted_vertex': violation_token.get('predicted_vertex'),
        'current_vertex': violation_token.get('current_vertex'),
        'is_valid_edge': violation_token.get('is_valid_edge', False),
        'rule_specific_analysis': violation_token.get('rule_violations', {}).get('rule_specific_analysis', {}),
        'walk_so_far': violation_token.get('walk_so_far', [])
    }
    
    return case_data


def analyze_violation_entropy_patterns(violation_time_series):
    """
    Analyze patterns in entropy metrics leading up to rule violations.
    
    Args:
        violation_time_series: Output from extract_violation_time_series()
        
    Returns:
        Dictionary with statistical analysis of entropy patterns before violations
    """
    patterns = {
        'overall_statistics': {},
        'rule_type_comparisons': {},
        'predictive_indicators': {},
        'entropy_signatures': {}
    }
    
    # Analyze each rule type
    for rule_type in ['repeater_violations', 'ascender_violations', 'even_violations']:
        if not violation_time_series[rule_type]:
            continue
        
        rule_data = violation_time_series[rule_type]
        
        # Aggregate entropy metrics across all cases
        aggregated_entropy = []
        aggregated_kl_oracle = []  # KL from rule-aware oracle
        aggregated_kl_structure = []  # KL from graph structure
        
        for case in rule_data:
            aggregated_entropy.extend(case['model_entropy'])
            
            if 'rule_aware_oracle' in case['kl_divergences']:
                aggregated_kl_oracle.extend(case['kl_divergences']['rule_aware_oracle'])
            
            if 'graph_structure' in case['kl_divergences']:
                aggregated_kl_structure.extend(case['kl_divergences']['graph_structure'])
        
        # Calculate statistics for this rule type
        if aggregated_entropy:
            patterns['rule_type_comparisons'][rule_type] = {
                'mean_entropy': np.mean(aggregated_entropy),
                'entropy_std': np.std(aggregated_entropy),
                'mean_kl_oracle': np.mean(aggregated_kl_oracle) if aggregated_kl_oracle else 0,
                'mean_kl_structure': np.mean(aggregated_kl_structure) if aggregated_kl_structure else 0,
                'entropy_trend': 'increasing' if len(aggregated_entropy) > 1 and 
                                 np.corrcoef(range(len(aggregated_entropy)), aggregated_entropy)[0,1] > 0 
                                 else 'decreasing',
                'num_cases': len(rule_data)
            }
    
    # Identify common predictive patterns
    patterns['predictive_indicators'] = {
        'entropy_spike_threshold': 0.5,  # Entropy increase that predicts violations
        'kl_divergence_warning': 1.0,   # KL divergence level that indicates trouble
        'confidence_collapse_rate': 0.1  # Rate of confidence decrease before violation
    }
    
    return patterns
