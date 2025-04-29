import random

import torch

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

    for sample_idx in range(num_walks):
        if verbose and (sample_idx + 1) % 10 == 0:
            print(f"Evaluating sample {sample_idx + 1}/{num_walks}")

        start_walk = generate_valid_walk(
            graph, random.choice(list(graph.nodes)), min_start_length, max_start_length, rules
        )

        input_tensor = torch.tensor(
            [vocab.token2idx[str(node)] for node in start_walk], dtype=torch.long
        ).unsqueeze(0)

        generated_walk = start_walk[:]
        current_vertex = start_walk[-1]

        counter = 0
        while current_vertex in graph.nodes:
            logits = model(input_tensor)
            next_vertex_idx = torch.argmax(logits[0, -1]).item()
            next_vertex = vocab.idx2token[next_vertex_idx]

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
                if not rule.is_satisfied_by(graph, generated_walk):
                    repeater_errors += 1
            elif hasattr(rule, "is_ascender_rule") and rule.is_ascender_rule:
                if not rule.is_satisfied_by(graph, generated_walk):
                    ascender_errors += 1
            elif hasattr(rule, "is_even_rule") and rule.is_even_rule:
                if not rule.is_satisfied_by(graph, generated_walk):
                    even_errors += 1

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

    return evaluation_results, error_summary


def count_rule_violations(walk, graph, rules):
    violations = 0
    for i in range(len(walk)):
        if not check_rule_compliance(walk[: i + 1], graph, rules):
            violations += 1
    return violations
