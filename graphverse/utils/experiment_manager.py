import csv
import os
import json
import pickle
import numpy as np
from datetime import datetime

def create_experiment_folder(base="experiments"):
    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, run_name)
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    os.makedirs(os.path.join(path, "evaluation"), exist_ok=True)
    return path

def save_config(config, folder):
    config_path = os.path.join(folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def save_error_summary(error_summary, folder):
    path = os.path.join(folder, "evaluation", "error_summary.json")
    with open(path, "w") as f:
        json.dump(error_summary, f, indent=2)

def save_kl_divergence_series(kl_series, folder):
    path = os.path.join(folder, "evaluation", "kl_divergence_timeseries.csv")
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["walk_idx", "step_idx", "kl_divergence"])
        for walk_idx, walk_kl in enumerate(kl_series):
            for step_idx, kl in enumerate(walk_kl):
                writer.writerow([walk_idx, step_idx, kl])


def save_token_level_data(token_data, folder):
    """
    Save detailed token-by-token analysis data.
    
    Args:
        token_data: List of token-level dictionaries from evaluate_model
        folder: Experiment folder path
    """
    # Save as JSON for full detail
    json_path = os.path.join(folder, "evaluation", "token_level_data.json")
    with open(json_path, "w") as f:
        json.dump(token_data, f, indent=2)
    
    # Save as CSV for easier analysis
    csv_path = os.path.join(folder, "evaluation", "token_level_data.csv")
    if token_data:
        with open(csv_path, "w", newline="") as csvfile:
            # Extract all possible field names
            fieldnames = set()
            for item in token_data:
                fieldnames.update(item.keys())
                if 'kl_divergences' in item:
                    for kl_type in item['kl_divergences'].keys():
                        fieldnames.add(f'kl_{kl_type}')
                if 'top_5_predictions' in item:
                    for i in range(min(5, len(item['top_5_predictions']))):
                        fieldnames.add(f'top_{i+1}_token')
                        fieldnames.add(f'top_{i+1}_prob')
            
            # Remove nested fields from main fieldnames
            csv_fieldnames = [f for f in fieldnames if f not in ['kl_divergences', 'top_5_predictions', 'context_tokens']]
            csv_fieldnames = sorted(csv_fieldnames)
            
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            
            for item in token_data:
                row = {}
                for field in csv_fieldnames:
                    if field.startswith('kl_'):
                        kl_type = field[3:]  # Remove 'kl_' prefix
                        row[field] = item.get('kl_divergences', {}).get(kl_type, '')
                    elif field.startswith('top_'):
                        # Parse top prediction fields
                        parts = field.split('_')
                        if len(parts) >= 3:
                            idx = int(parts[1]) - 1
                            field_type = parts[2]  # 'token' or 'prob'
                            top_preds = item.get('top_5_predictions', [])
                            if idx < len(top_preds):
                                if field_type == 'token':
                                    row[field] = top_preds[idx]['token']
                                elif field_type == 'prob':
                                    row[field] = top_preds[idx]['probability']
                    else:
                        row[field] = item.get(field, '')
                
                writer.writerow(row)


def save_token_summary_stats(token_data, folder):
    """
    Save summary statistics about token-level performance.
    
    Args:
        token_data: List of token-level dictionaries from evaluate_model
        folder: Experiment folder path
    """
    if not token_data:
        return
    
    # Calculate summary statistics
    stats = {
        'total_tokens': len(token_data),
        'unique_walks': len(set(item['walk_idx'] for item in token_data)),
        'avg_context_length': sum(item['context_length'] for item in token_data) / len(token_data),
        'valid_edge_rate': sum(1 for item in token_data if item['is_valid_edge']) / len(token_data),
        'avg_prediction_confidence': sum(item['prediction_confidence'] for item in token_data) / len(token_data),
        'avg_entropy': sum(item['entropy'] for item in token_data) / len(token_data),
    }
    
    # Rule violation statistics
    violation_stats = calculate_violation_statistics(token_data)
    stats.update(violation_stats)
    
    # KL divergence statistics
    kl_stats = {}
    if token_data and 'kl_divergences' in token_data[0]:
        kl_types = token_data[0]['kl_divergences'].keys()
        for kl_type in kl_types:
            kl_values = [item['kl_divergences'][kl_type] for item in token_data]
            kl_stats[f'avg_kl_{kl_type}'] = sum(kl_values) / len(kl_values)
            kl_stats[f'min_kl_{kl_type}'] = min(kl_values)
            kl_stats[f'max_kl_{kl_type}'] = max(kl_values)
    
    stats.update(kl_stats)
    
    # Position-based statistics
    max_position = max(item['step_idx'] for item in token_data)
    stats['max_generation_steps'] = max_position
    
    # Context length distribution
    context_lengths = [item['context_length'] for item in token_data]
    stats['min_context_length'] = min(context_lengths)
    stats['max_context_length'] = max(context_lengths)
    
    # Save summary
    summary_path = os.path.join(folder, "evaluation", "token_summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)


def calculate_violation_statistics(token_data):
    """Calculate detailed rule violation statistics."""
    stats = {}
    
    if not token_data or 'rule_violations' not in token_data[0]:
        return stats
    
    total_tokens = len(token_data)
    
    # Overall violation rates
    tokens_with_violations = sum(1 for item in token_data if item['rule_violations']['has_violation'])
    stats['overall_violation_rate'] = tokens_with_violations / total_tokens
    
    # Rule-specific violation rates
    rule_types = ['ascender_rule', 'even_rule', 'repeater_rule', 'invalid_edge']
    for rule_type in rule_types:
        violations = sum(1 for item in token_data 
                        if rule_type in item['rule_violations']['violation_types'])
        stats[f'{rule_type}_violation_rate'] = violations / total_tokens
    
    # Average violation probabilities
    if token_data and 'violation_probabilities' in token_data[0]['rule_violations']:
        for prob_type in ['ascender_violation_prob', 'even_violation_prob', 
                         'repeater_violation_prob', 'invalid_edge_prob', 'total_violation_prob']:
            probs = [item['rule_violations']['violation_probabilities'].get(prob_type, 0) 
                    for item in token_data]
            stats[f'avg_{prob_type}'] = sum(probs) / len(probs)
            stats[f'max_{prob_type}'] = max(probs)
    
    # Confidence analysis by violation type
    violated_tokens = [item for item in token_data if item['rule_violations']['has_violation']]
    valid_tokens = [item for item in token_data if not item['rule_violations']['has_violation']]
    
    if violated_tokens:
        stats['avg_confidence_violated_predictions'] = sum(item['prediction_confidence'] for item in violated_tokens) / len(violated_tokens)
    if valid_tokens:
        stats['avg_confidence_valid_predictions'] = sum(item['prediction_confidence'] for item in valid_tokens) / len(valid_tokens)
    
    # Alternative token analysis
    tokens_with_alternatives = [item for item in token_data 
                               if item['rule_violations']['alternative_valid_tokens']]
    if tokens_with_alternatives:
        stats['tokens_with_valid_alternatives_rate'] = len(tokens_with_alternatives) / total_tokens
        avg_alternatives = sum(len(item['rule_violations']['alternative_valid_tokens']) 
                             for item in tokens_with_alternatives) / len(tokens_with_alternatives)
        stats['avg_valid_alternatives_per_decision'] = avg_alternatives
    
    return stats


def save_rule_violation_analysis(token_data, folder):
    """
    Save detailed rule violation analysis as separate files.
    
    Args:
        token_data: List of token-level dictionaries from evaluate_model
        folder: Experiment folder path
    """
    if not token_data:
        return
    
    # Extract violation-specific data
    violations_only = []
    for item in token_data:
        if item['rule_violations']['has_violation']:
            violation_summary = {
                'walk_idx': item['walk_idx'],
                'step_idx': item['step_idx'],
                'current_vertex': item['current_vertex'],
                'predicted_vertex': item['predicted_vertex'],
                'prediction_confidence': item['prediction_confidence'],
                'violation_types': item['rule_violations']['violation_types'],
                'violation_probabilities': item['rule_violations']['violation_probabilities'],
                'rule_specific_analysis': item['rule_violations']['rule_specific_analysis'],
                'walk_context': item['walk_so_far']
            }
            violations_only.append(violation_summary)
    
    # Save violations summary
    violations_path = os.path.join(folder, "evaluation", "rule_violations.json")
    with open(violations_path, "w") as f:
        json.dump(violations_only, f, indent=2)
    
    # Save violation probabilities time series
    prob_series_path = os.path.join(folder, "evaluation", "violation_probabilities.csv")
    with open(prob_series_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "walk_idx", "step_idx", "ascender_violation_prob", "even_violation_prob",
            "repeater_violation_prob", "invalid_edge_prob", "total_violation_prob"
        ])
        
        for item in token_data:
            if 'violation_probabilities' in item['rule_violations']:
                vp = item['rule_violations']['violation_probabilities']
                writer.writerow([
                    item['walk_idx'], item['step_idx'],
                    vp.get('ascender_violation_prob', 0),
                    vp.get('even_violation_prob', 0),
                    vp.get('repeater_violation_prob', 0),
                    vp.get('invalid_edge_prob', 0),
                    vp.get('total_violation_prob', 0)
                ])


def save_exemplar_walks(exemplar_walks, folder):
    """
    Save exemplar walks for paper documentation.
    
    Args:
        exemplar_walks: Dictionary of exemplar walks by category
        folder: Experiment folder path
    """
    import numpy as np
    
    if not exemplar_walks:
        return
    
    # Save detailed exemplars as JSON
    exemplars_path = os.path.join(folder, "evaluation", "exemplar_walks.json")
    with open(exemplars_path, "w") as f:
        json.dump(exemplar_walks, f, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x))
    
    # Create a summary of exemplar counts
    exemplar_summary = {}
    for category, walks in exemplar_walks.items():
        exemplar_summary[category] = {
            'count': len(walks),
            'category_description': get_exemplar_category_description(category)
        }
        
        if walks:
            # Add summary statistics for this category
            if 'avg_confidence' in walks[0]:
                confidences = [w['avg_confidence'] for w in walks if 'avg_confidence' in w]
                if confidences:
                    exemplar_summary[category]['avg_confidence_range'] = {
                        'min': min(confidences),
                        'max': max(confidences),
                        'mean': sum(confidences) / len(confidences)
                    }
            
            if 'total_violations' in walks[0]:
                violations = [w['total_violations'] for w in walks if 'total_violations' in w]
                if violations:
                    exemplar_summary[category]['violation_range'] = {
                        'min': min(violations),
                        'max': max(violations),
                        'mean': sum(violations) / len(violations)
                    }
    
    # Save exemplar summary
    summary_path = os.path.join(folder, "evaluation", "exemplar_summary.json")
    with open(summary_path, "w") as f:
        json.dump(exemplar_summary, f, indent=2)


def get_exemplar_category_description(category):
    """Get human-readable description for exemplar categories."""
    descriptions = {
        'perfect_rule_compliance': 'Walks with no rule violations and high confidence',
        'rule_violations': 'Walks with the most rule violations',
        'context_window_failures': 'Walks likely failing due to context window limitations',
        'high_confidence_predictions': 'Walks with highest average prediction confidence',
        'low_confidence_predictions': 'Walks with lowest average prediction confidence',
        'entropy_progression_examples': 'Walks showing clear entropy trends over time',
        'baseline_comparison_examples': 'Walks with strong performance vs baselines',
        'repeater_learning_examples': 'Walks successfully following repeater rules',
        'repeater_failure_examples': 'Walks failing at repeater rule compliance'
    }
    return descriptions.get(category, f'Examples of {category.replace("_", " ")}')


def save_baseline_performance_summary(token_data, progressive_analysis, folder):
    """
    Save comprehensive baseline performance summary.
    
    Args:
        token_data: List of token-level dictionaries from evaluate_model
        progressive_analysis: Progressive difficulty analysis results
        folder: Experiment folder path
    """
    import numpy as np
    
    if not token_data:
        return
    
    # Calculate overall baseline performance metrics
    baseline_summary = {
        'overall_metrics': {},
        'by_baseline_type': {},
        'progressive_analysis': progressive_analysis,
        'statistical_significance': {}
    }
    
    # Extract baseline names from first token
    if token_data and 'normalized_metrics' in token_data[0]:
        baseline_names = list(token_data[0]['normalized_metrics']['skill_scores'].keys())
        
        for baseline_name in baseline_names:
            # Collect all valid scores for this baseline
            skill_scores = []
            info_gains = []
            relative_entropies = []
            prob_ratios = []
            
            for token in token_data:
                metrics = token['normalized_metrics']
                
                skill_score = metrics['skill_scores'][baseline_name]
                if not np.isinf(skill_score) and not np.isnan(skill_score):
                    skill_scores.append(skill_score)
                
                info_gain = metrics['information_gains'][baseline_name]
                if not np.isinf(info_gain) and not np.isnan(info_gain):
                    info_gains.append(info_gain)
                
                rel_entropy = metrics['relative_entropies'][baseline_name]
                if not np.isinf(rel_entropy) and not np.isnan(rel_entropy):
                    relative_entropies.append(rel_entropy)
                
                prob_ratio = metrics['baseline_comparisons'][baseline_name]['prob_ratio']
                if not np.isinf(prob_ratio) and not np.isnan(prob_ratio):
                    prob_ratios.append(prob_ratio)
            
            # Calculate summary statistics
            baseline_summary['by_baseline_type'][baseline_name] = {
                'skill_score': {
                    'mean': np.mean(skill_scores) if skill_scores else 0.0,
                    'std': np.std(skill_scores) if skill_scores else 0.0,
                    'median': np.median(skill_scores) if skill_scores else 0.0,
                    'min': np.min(skill_scores) if skill_scores else 0.0,
                    'max': np.max(skill_scores) if skill_scores else 0.0,
                    'count': len(skill_scores)
                },
                'information_gain': {
                    'mean': np.mean(info_gains) if info_gains else 0.0,
                    'std': np.std(info_gains) if info_gains else 0.0,
                    'median': np.median(info_gains) if info_gains else 0.0
                },
                'relative_entropy': {
                    'mean': np.mean(relative_entropies) if relative_entropies else 1.0,
                    'std': np.std(relative_entropies) if relative_entropies else 0.0,
                    'median': np.median(relative_entropies) if relative_entropies else 1.0
                },
                'probability_ratio': {
                    'mean': np.mean(prob_ratios) if prob_ratios else 1.0,
                    'std': np.std(prob_ratios) if prob_ratios else 0.0,
                    'median': np.median(prob_ratios) if prob_ratios else 1.0
                }
            }
    
    # Overall performance summary
    all_entropies = [token['entropy'] for token in token_data]
    all_confidences = [token['prediction_confidence'] for token in token_data]
    all_perplexities = [token['normalized_metrics']['model_perplexity'] for token in token_data]
    
    baseline_summary['overall_metrics'] = {
        'average_entropy': np.mean(all_entropies),
        'average_confidence': np.mean(all_confidences),
        'average_perplexity': np.mean(all_perplexities),
        'entropy_std': np.std(all_entropies),
        'confidence_std': np.std(all_confidences),
        'total_tokens_analyzed': len(token_data)
    }
    
    # Save baseline performance summary
    baseline_path = os.path.join(folder, "evaluation", "baseline_performance_summary.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline_summary, f, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x))


def save_trajectory_metadata(trajectories, folder):
    """
    Save comprehensive trajectory metadata in multiple formats.
    
    Args:
        trajectories: EvaluationTrajectoryMetadata object
        folder: Experiment folder path
    """
    # 1. Save as pickle for full object preservation
    trajectory_path = os.path.join(folder, "evaluation", "trajectory_metadata.pkl")
    with open(trajectory_path, "wb") as f:
        pickle.dump(trajectories, f)
    
    # 2. Export summary as JSON for quick inspection
    summary_path = os.path.join(folder, "evaluation", "trajectory_summary.json")
    with open(summary_path, "w") as f:
        json.dump(trajectories.get_summary(), f, indent=2, 
                 default=lambda x: float(x) if np.isscalar(x) else str(x))
    
    # 3. Export trajectory metrics as CSV for analysis
    export_trajectory_csv(trajectories, folder)
    
    # 4. Optional: Export to HDF5 if available
    try:
        save_trajectories_hdf5(trajectories, folder)
    except ImportError:
        pass  # HDF5 not available, skip


def export_trajectory_csv(trajectories, folder):
    """
    Export key trajectory metrics to CSV format.
    
    Args:
        trajectories: EvaluationTrajectoryMetadata object  
        folder: Experiment folder path
    """
    csv_path = os.path.join(folder, "evaluation", "trajectory_metrics.csv")
    
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            'walk_idx', 'termination_reason', 'termination_step', 'final_length',
            'mean_entropy', 'std_entropy', 'mean_confidence', 'std_confidence',
            'entropy_slope', 'num_critical_points', 'num_violations',
            'mean_kl_uniform_random', 'mean_ks_uniform_random'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for traj in trajectories.walk_trajectories:
            row = {
                'walk_idx': traj.walk_idx,
                'termination_reason': traj.termination_reason,
                'termination_step': traj.termination_step,
                'final_length': traj.final_length,
                'num_critical_points': len(traj.critical_points),
                'num_violations': len(traj.violation_timeline)
            }
            
            # Add statistics if computed
            if traj.trajectory_statistics:
                for key in ['mean_entropy', 'std_entropy', 'mean_confidence', 
                           'std_confidence', 'entropy_slope', 'mean_kl_uniform_random',
                           'mean_ks_uniform_random']:
                    if key in traj.trajectory_statistics:
                        row[key] = traj.trajectory_statistics[key]
            
            writer.writerow(row)


def save_trajectories_hdf5(trajectories, folder):
    """
    Save trajectory data in HDF5 format for efficient numerical analysis.
    
    Args:
        trajectories: EvaluationTrajectoryMetadata object
        folder: Experiment folder path
    """
    try:
        import h5py
    except ImportError:
        return  # HDF5 not available
    
    h5_path = os.path.join(folder, "evaluation", "trajectories.h5")
    
    with h5py.File(h5_path, 'w') as f:
        # Create metadata group
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['num_walks'] = trajectories.num_walks
        
        # Store outcome distribution
        outcome_grp = meta_grp.create_group('outcomes')
        for outcome, trajs in trajectories.outcome_groups.items():
            outcome_grp.attrs[outcome] = len(trajs)
        
        # Create groups for each walk
        for traj in trajectories.walk_trajectories:
            walk_grp = f.create_group(f"walk_{traj.walk_idx}")
            
            # Store basic metadata as attributes
            walk_grp.attrs['termination_reason'] = traj.termination_reason or 'unknown'
            walk_grp.attrs['termination_step'] = traj.termination_step or -1
            walk_grp.attrs['final_length'] = traj.final_length
            
            # Store uncertainty trajectories as datasets
            traj_grp = walk_grp.create_group('uncertainty_trajectory')
            
            # Store entropies and confidences
            if traj.uncertainty_trajectory['entropies']:
                traj_grp.create_dataset('entropies', 
                                       data=traj.uncertainty_trajectory['entropies'])
            if traj.uncertainty_trajectory['confidences']:
                traj_grp.create_dataset('confidences', 
                                       data=traj.uncertainty_trajectory['confidences'])
            
            # Store KL divergences
            kl_grp = traj_grp.create_group('kl_divergences')
            for baseline, values in traj.uncertainty_trajectory['kl_divergences'].items():
                if values:
                    kl_grp.create_dataset(baseline, data=values)
            
            # Store KS distances
            ks_grp = traj_grp.create_group('ks_distances')
            for baseline, values in traj.uncertainty_trajectory['ks_distances'].items():
                if values:
                    ks_grp.create_dataset(baseline, data=values)
            
            # Store full probability distributions if available
            if traj.probability_distributions:
                prob_data = np.array(traj.probability_distributions)
                walk_grp.create_dataset('probability_distributions', 
                                       data=prob_data,
                                       compression='gzip',
                                       compression_opts=9)
            
            # Store distribution statistics
            dist_stats_grp = walk_grp.create_group('distribution_stats')
            for stat_name, values in traj.distribution_stats.items():
                if values:
                    dist_stats_grp.create_dataset(stat_name, data=values)
    
    print(f"Trajectory data saved to HDF5: {h5_path}")