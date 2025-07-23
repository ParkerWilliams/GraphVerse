import random

import torch
import torch.nn.functional as F
import numpy as np

from ..graph.walk import check_rule_compliance, generate_valid_walk


def evaluate_model(
    model,
    graph,
    vocab,
    num_walks,
    min_start_length,
    max_start_length,
    rules,
    verbose=False,
):
    model.eval()

    evaluation_results = []
    repeater_errors = 0
    ascender_errors = 0
    even_errors = 0
    broken_graph_errors = 0
    total_steps = 0

    # --- New: Store KL divergence time series for each walk ---
    kl_divergence_series = []

    for sample_idx in range(num_walks):
        if verbose and (sample_idx + 1) % 10 == 0:
            print(f"Evaluating sample {sample_idx + 1}/{num_walks}")

        start_walk = generate_valid_walk(
            graph, random.choice(range(graph.n)), min_start_length, max_start_length, rules
        )

        input_tensor = torch.tensor(
            [vocab.token2idx[str(node)] for node in start_walk], dtype=torch.long
        ).unsqueeze(0)

        generated_walk = start_walk[:]
        current_vertex = start_walk[-1]

        counter = 0
        kl_series = []  # KL values for this walk

        while current_vertex in range(graph.n):
            logits = model(input_tensor)
            next_vertex_idx = torch.argmax(logits[0, -1]).item()
            next_vertex = vocab.idx2token[next_vertex_idx]

            # --- New: Compute KL divergence ---
            vocab_size = logits.shape[-1]
            ref_dist = negative_exponential_distribution(vocab_size, scale=5.0)  # scale can be tuned
            kl = kl_divergence(logits[0, -1], ref_dist)
            kl_series.append(kl)
            # --- End new code ---

            if next_vertex == "<END>":
                break

            try:
                next_vertex_int = int(next_vertex)
            except ValueError:
                break  # Not a valid node

            # --- Broken graph connection check ---
            if not graph.has_edge(current_vertex, next_vertex_int):
                broken_graph_errors += 1
                break

            generated_walk.append(next_vertex_int)
            input_tensor = torch.cat(
                (input_tensor, torch.tensor([[next_vertex_idx]], dtype=torch.long)),
                dim=1,
            )

            current_vertex = next_vertex_int
            counter += 1
            total_steps += 1

        # --- Rule violation tracking ---
        for rule in rules:
            if hasattr(rule, "is_repeater_rule") and rule.is_repeater_rule:
                if not rule.is_satisfied_by(generated_walk, graph):
                    repeater_errors += 1
            elif hasattr(rule, "is_ascender_rule") and rule.is_ascender_rule:
                if not rule.is_satisfied_by(generated_walk, graph):
                    ascender_errors += 1
            elif hasattr(rule, "is_even_rule") and rule.is_even_rule:
                if not rule.is_satisfied_by(generated_walk, graph):
                    even_errors += 1

        # --- New: Store KL series for this walk ---
        kl_divergence_series.append(kl_series)
        # --- End new code ---

        evaluation_results.append(
            {
                "start_walk": start_walk,
                "generated_walk": generated_walk,
                # Optionally, add more details here
            }
        )

    # --- Summarize error rates ---
    error_summary = {
        "repeater_error_rate": repeater_errors / num_walks,
        "ascender_error_rate": ascender_errors / num_walks,
        "even_error_rate": even_errors / num_walks,
        "broken_graph_error_rate": broken_graph_errors / num_walks,
        "total_steps": total_steps,
    }

    if verbose:
        print("Evaluation completed.")
        print("Error summary:", error_summary)

    # --- New: Return KL divergence series as part of results ---
    return evaluation_results, error_summary, kl_divergence_series


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
