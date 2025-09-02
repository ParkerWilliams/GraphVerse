"""Fast evaluation module for quick model testing."""

import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from typing import Dict, List, Tuple
from graphverse.graph.base import Graph


def evaluate_model_fast(
    model,
    graph: Graph,
    rules: List,
    vocab,
    num_walks: int = 1000,  # Reduced default
    min_start_length: int = 5,
    max_start_length: int = 10,
    max_walk_length: int = 100,
    verbose: bool = True,
    device: str = None
) -> Dict:
    """
    Fast evaluation that only tracks essential metrics.
    
    Skips expensive computations like:
    - Multiple reference distributions
    - Distribution comparisons
    - Detailed token-level analysis
    
    Returns:
        Dictionary with basic error rates and statistics
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    if verbose:
        print("\n" + "="*60)
        print("FAST MODEL EVALUATION")
        print("="*60)
        print(f"  Number of walks: {num_walks}")
        print(f"  Device: {device}")
        print("  Tracking: Basic error rates only")
        print("="*60 + "\n")
    
    # Error counters
    repeater_errors = 0
    ascender_errors = 0
    even_errors = 0
    broken_graph_errors = 0
    total_steps = 0
    completed_walks = 0
    
    # Get unknown token index (use <PAD> as fallback)
    unk_token = vocab.token2idx.get("<UNK>", vocab.token2idx.get("<PAD>", 0))
    
    # Create progress bar
    if verbose:
        walk_iterator = tqdm(range(num_walks), desc="Evaluating", unit="walk")
    else:
        walk_iterator = range(num_walks)
    
    with torch.no_grad():  # Disable gradient computation for speed
        for walk_idx in walk_iterator:
            # Generate random starting walk
            start_node = random.randint(0, graph.n - 1)
            walk_length = random.randint(min_start_length, max_start_length)
            
            # Simple walk generation without rule checking
            current = start_node
            generated_walk = [current]
            
            for _ in range(walk_length - 1):
                neighbors = graph.get_neighbors(current)
                if len(neighbors) == 0:
                    break
                current = random.choice(neighbors)
                generated_walk.append(current)
            
            # Convert to tensor
            input_ids = [vocab.token2idx.get(str(node), unk_token) 
                        for node in generated_walk]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            current_vertex = generated_walk[-1]
            step_count = 0
            
            # Generate continuation
            while current_vertex in range(graph.n) and step_count < max_walk_length:
                # Get model prediction
                logits = model(input_tensor)
                next_vertex_idx = torch.argmax(logits[0, -1]).item()
                
                # Convert to vertex  
                next_vertex_str = vocab.idx2token.get(next_vertex_idx, "<PAD>")
                
                # Check for end token
                if next_vertex_str == "<END>":
                    completed_walks += 1
                    break
                
                # Try to parse as integer
                try:
                    next_vertex = int(next_vertex_str)
                except (ValueError, TypeError):
                    break
                
                # Check if edge exists (broken graph error)
                if not graph.has_edge(current_vertex, next_vertex):
                    broken_graph_errors += 1
                    break
                
                # Quick rule violation checks (simplified)
                # Check repeater violation
                for rule in rules:
                    if hasattr(rule, 'check_repeater_violation'):
                        if rule.check_repeater_violation(generated_walk, next_vertex):
                            repeater_errors += 1
                            break
                    elif hasattr(rule, 'check_ascender_violation'):
                        if rule.check_ascender_violation(current_vertex, next_vertex):
                            ascender_errors += 1
                            break
                    elif hasattr(rule, 'check_even_violation'):
                        if rule.check_even_violation(current_vertex, next_vertex):
                            even_errors += 1
                            break
                
                # Add to walk
                generated_walk.append(next_vertex)
                
                # Update input tensor
                new_token_idx = vocab.token2idx.get(str(next_vertex), unk_token)
                new_token = torch.tensor([[new_token_idx]], dtype=torch.long).to(device)
                input_tensor = torch.cat([input_tensor, new_token], dim=1)
                
                # Keep tensor size manageable
                if input_tensor.size(1) > 100:
                    input_tensor = input_tensor[:, -100:]
                
                current_vertex = next_vertex
                step_count += 1
                total_steps += 1
            
            # Update progress bar with current error rates
            if verbose and walk_idx > 0:
                walk_iterator.set_postfix({
                    'broken': f"{100*broken_graph_errors/(walk_idx+1):.1f}%",
                    'steps': total_steps
                })
    
    # Calculate final statistics
    results = {
        'num_walks': num_walks,
        'total_steps': total_steps,
        'completed_walks': completed_walks,
        'error_rates': {
            'repeater': repeater_errors / max(1, total_steps),
            'ascender': ascender_errors / max(1, total_steps),
            'even': even_errors / max(1, total_steps),
            'broken_graph': broken_graph_errors / max(1, num_walks)
        },
        'avg_steps_per_walk': total_steps / max(1, num_walks)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"  Total walks: {num_walks}")
        print(f"  Total steps: {total_steps}")
        print(f"  Avg steps/walk: {results['avg_steps_per_walk']:.1f}")
        print(f"  Completed walks: {completed_walks}")
        print("\nError Rates:")
        print(f"  Broken graph: {results['error_rates']['broken_graph']*100:.2f}%")
        print(f"  Repeater: {results['error_rates']['repeater']*100:.2f}%")
        print(f"  Ascender: {results['error_rates']['ascender']*100:.2f}%")
        print(f"  Even: {results['error_rates']['even']*100:.2f}%")
        print("="*60)
    
    return results