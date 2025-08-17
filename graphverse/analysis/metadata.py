"""
Metadata classes for comprehensive experiment analysis and tracking.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter


class GraphMetadata:
    """
    Metadata about the graph structure and rule composition.
    """
    
    def __init__(self, graph, rules):
        self.n_nodes = graph.n
        self.n_edges = int(np.sum(graph.adjacency > 0))
        self.edge_density = self.n_edges / (self.n_nodes * (self.n_nodes - 1)) if self.n_nodes > 1 else 0
        self.is_connected = graph.is_connected()
        
        # Analyze rules
        self.rule_composition = self._analyze_rule_composition(rules)
        self.node_rule_mapping = self._create_node_rule_mapping(rules)
        
    def _analyze_rule_composition(self, rules):
        """Analyze the composition of rules in the graph."""
        composition = {
            'total_rule_nodes': 0,
            'rule_types': {},
            'rule_percentages': {}
        }
        
        for rule in rules:
            if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                rule_type = 'ascender'
                nodes = list(rule.member_nodes)
                composition['rule_types'][rule_type] = {
                    'count': len(nodes),
                    'nodes': nodes,
                    'selection_criteria': 'middle 20% of node range'
                }
            elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
                rule_type = 'even'
                nodes = list(rule.member_nodes)
                composition['rule_types'][rule_type] = {
                    'count': len(nodes),
                    'nodes': nodes,
                    'selection_criteria': 'even-numbered nodes'
                }
            elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                rule_type = 'repeater'
                nodes = list(rule.member_nodes)
                k_values = dict(rule.members_nodes_dict)
                composition['rule_types'][rule_type] = {
                    'count': len(nodes),
                    'nodes': nodes,
                    'k_values': k_values,
                    'k_range': f"{min(k_values.values())}-{max(k_values.values())}" if k_values else "N/A",
                    'selection_criteria': 'random from remaining nodes'
                }
            
            composition['total_rule_nodes'] += len(rule.member_nodes)
        
        # Calculate percentages
        for rule_type, data in composition['rule_types'].items():
            composition['rule_percentages'][rule_type] = (data['count'] / self.n_nodes) * 100
        
        composition['total_rule_percentage'] = (composition['total_rule_nodes'] / self.n_nodes) * 100
        composition['regular_nodes'] = self.n_nodes - composition['total_rule_nodes']
        composition['regular_percentage'] = (composition['regular_nodes'] / self.n_nodes) * 100
        
        return composition
    
    def _create_node_rule_mapping(self, rules):
        """Create mapping from node to rule type."""
        mapping = {}
        
        for rule in rules:
            if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                for node in rule.member_nodes:
                    mapping[node] = 'ascender'
            elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
                for node in rule.member_nodes:
                    mapping[node] = 'even'
            elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                for node in rule.member_nodes:
                    mapping[node] = 'repeater'
        
        return mapping
    
    def get_summary(self):
        """Get a summary dictionary of graph metadata."""
        return {
            'nodes': self.n_nodes,
            'edges': self.n_edges,
            'edge_density': self.edge_density,
            'is_connected': self.is_connected,
            'rule_composition': self.rule_composition
        }


class TrainingCorpusMetadata:
    """
    Comprehensive metadata about the training corpus composition and characteristics.
    """
    
    def __init__(self, walks, rules, vocab=None):
        self.total_walks = len(walks)
        self.vocab = vocab
        
        # Basic corpus statistics
        self.walk_statistics = self._compute_walk_statistics(walks)
        
        # Rule-based analysis
        self.rule_exposure = self._analyze_rule_exposure(walks, rules)
        
        # Structural analysis
        self.structural_analysis = self._analyze_structural_patterns(walks)
        
        # Vocabulary analysis
        self.vocabulary_analysis = self._analyze_vocabulary_usage(walks, vocab) if vocab else {}
    
    def _compute_walk_statistics(self, walks):
        """Compute basic statistical properties of walks."""
        if not walks:
            return {}
        
        lengths = [len(walk) for walk in walks]
        
        return {
            'total_walks': len(walks),
            'length_stats': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths)
            },
            'total_steps': sum(lengths),
            'unique_sequences': len(set(tuple(walk) for walk in walks)),
            'sequence_diversity': len(set(tuple(walk) for walk in walks)) / len(walks)
        }
    
    def _analyze_rule_exposure(self, walks, rules):
        """Analyze rule exposure patterns in the corpus."""
        if not walks or not rules:
            return {}
        
        # Get rule nodes
        rule_nodes = {}
        all_rule_nodes = set()
        
        for rule in rules:
            if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                rule_nodes['ascender'] = set(rule.member_nodes)
                all_rule_nodes.update(rule.member_nodes)
            elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
                rule_nodes['even'] = set(rule.member_nodes)
                all_rule_nodes.update(rule.member_nodes)
            elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                rule_nodes['repeater'] = set(rule.member_nodes)
                all_rule_nodes.update(rule.member_nodes)
        
        # Analyze walks
        exposure_stats = {
            'rule_node_frequencies': Counter(),
            'walk_classifications': {
                'no_rules': 0,
                'single_rule_type': 0,
                'multiple_rule_types': 0
            },
            'rule_type_exposure': {rule_type: 0 for rule_type in rule_nodes.keys()},
            'co_occurrence_patterns': defaultdict(int)
        }
        
        for walk in walks:
            walk_rule_types = set()
            walk_has_rules = False
            
            # Check each node in the walk
            for node in walk:
                if node in all_rule_nodes:
                    walk_has_rules = True
                    exposure_stats['rule_node_frequencies'][node] += 1
                    
                    # Determine rule type
                    for rule_type, nodes in rule_nodes.items():
                        if node in nodes:
                            walk_rule_types.add(rule_type)
            
            # Classify walk
            if not walk_has_rules:
                exposure_stats['walk_classifications']['no_rules'] += 1
            elif len(walk_rule_types) == 1:
                exposure_stats['walk_classifications']['single_rule_type'] += 1
            else:
                exposure_stats['walk_classifications']['multiple_rule_types'] += 1
            
            # Count rule type exposures
            for rule_type in walk_rule_types:
                exposure_stats['rule_type_exposure'][rule_type] += 1
            
            # Track co-occurrence patterns
            if len(walk_rule_types) > 1:
                pattern = tuple(sorted(walk_rule_types))
                exposure_stats['co_occurrence_patterns'][pattern] += 1
        
        # Calculate percentages
        exposure_stats['exposure_percentages'] = {}
        for rule_type, count in exposure_stats['rule_type_exposure'].items():
            exposure_stats['exposure_percentages'][rule_type] = (count / self.total_walks) * 100
        
        exposure_stats['exposure_percentages']['any_rule'] = (
            (self.total_walks - exposure_stats['walk_classifications']['no_rules']) / self.total_walks * 100
        )
        exposure_stats['exposure_percentages']['no_rules'] = (
            exposure_stats['walk_classifications']['no_rules'] / self.total_walks * 100
        )
        
        return exposure_stats
    
    def _analyze_structural_patterns(self, walks):
        """Analyze structural patterns in walks."""
        if not walks:
            return {}
        
        patterns = {
            'node_transition_frequencies': Counter(),
            'node_frequencies': Counter(),
            'start_node_distribution': Counter(),
            'end_node_distribution': Counter(),
            'path_patterns': {
                'loops': 0,
                'back_and_forth': 0,
                'monotonic_increasing': 0,
                'monotonic_decreasing': 0
            }
        }
        
        for walk in walks:
            if len(walk) == 0:
                continue
                
            # Node frequencies
            for node in walk:
                patterns['node_frequencies'][node] += 1
            
            # Start/end distributions
            patterns['start_node_distribution'][walk[0]] += 1
            patterns['end_node_distribution'][walk[-1]] += 1
            
            # Transition frequencies
            for i in range(len(walk) - 1):
                transition = (walk[i], walk[i + 1])
                patterns['node_transition_frequencies'][transition] += 1
            
            # Pattern analysis
            if len(set(walk)) < len(walk):
                patterns['path_patterns']['loops'] += 1
            
            if len(walk) > 2 and any(walk[i] == walk[i + 2] for i in range(len(walk) - 2)):
                patterns['path_patterns']['back_and_forth'] += 1
            
            if walk == sorted(walk):
                patterns['path_patterns']['monotonic_increasing'] += 1
            elif walk == sorted(walk, reverse=True):
                patterns['path_patterns']['monotonic_decreasing'] += 1
        
        return patterns
    
    def _analyze_vocabulary_usage(self, walks, vocab):
        """Analyze vocabulary usage patterns."""
        if not vocab:
            return {}
        
        # Flatten all walks into tokens
        all_tokens = []
        for walk in walks:
            all_tokens.extend([str(node) for node in walk])
        
        token_frequencies = Counter(all_tokens)
        
        return {
            'vocab_size': len(vocab),
            'tokens_used': len(set(all_tokens)),
            'vocab_coverage': len(set(all_tokens)) / len(vocab),
            'token_frequencies': dict(token_frequencies.most_common(20)),  # Top 20
            'rare_tokens': [token for token, freq in token_frequencies.items() if freq == 1],
            'token_distribution_stats': {
                'mean_frequency': np.mean(list(token_frequencies.values())),
                'std_frequency': np.std(list(token_frequencies.values())),
                'max_frequency': max(token_frequencies.values()),
                'min_frequency': min(token_frequencies.values())
            }
        }
    
    def get_summary(self):
        """Get a summary dictionary of corpus metadata."""
        summary = {
            'basic_stats': self.walk_statistics,
            'rule_exposure': {
                'exposure_percentages': self.rule_exposure.get('exposure_percentages', {}),
                'walk_classifications': self.rule_exposure.get('walk_classifications', {})
            }
        }
        
        if self.vocabulary_analysis:
            summary['vocabulary'] = {
                'coverage': self.vocabulary_analysis.get('vocab_coverage', 0),
                'tokens_used': self.vocabulary_analysis.get('tokens_used', 0)
            }
        
        return summary


class ExperimentMetadata:
    """
    Combined metadata for complete experiment analysis.
    """
    
    def __init__(self, graph_metadata: GraphMetadata, corpus_metadata: TrainingCorpusMetadata, 
                 config: Dict[str, Any], model_performance: Optional[Dict[str, Any]] = None,
                 evaluation_trajectories: Optional['EvaluationTrajectoryMetadata'] = None):
        self.graph = graph_metadata
        self.corpus = corpus_metadata
        self.config = config
        self.model_performance = model_performance or {}
        self.evaluation_trajectories = evaluation_trajectories  # NEW: Trajectory analysis
        
        # Compute derived metrics
        self.analysis = self._compute_analysis_metrics()
    
    def _compute_analysis_metrics(self):
        """Compute derived metrics for analysis."""
        analysis = {}
        
        # Rule density vs exposure correlation
        if hasattr(self.graph, 'rule_composition') and hasattr(self.corpus, 'rule_exposure'):
            rule_density_exposure = {}
            for rule_type in ['ascender', 'even', 'repeater']:
                graph_percent = self.graph.rule_composition.get('rule_percentages', {}).get(rule_type, 0)
                corpus_percent = self.corpus.rule_exposure.get('exposure_percentages', {}).get(rule_type, 0)
                
                if graph_percent > 0:
                    exposure_ratio = corpus_percent / graph_percent
                    rule_density_exposure[rule_type] = {
                        'graph_percentage': graph_percent,
                        'corpus_exposure_percentage': corpus_percent,
                        'exposure_amplification': exposure_ratio
                    }
            
            analysis['rule_density_vs_exposure'] = rule_density_exposure
        
        # Training efficiency metrics
        if self.corpus.walk_statistics:
            analysis['training_efficiency'] = {
                'sequences_per_unique': self.corpus.walk_statistics.get('total_walks', 0) / max(1, self.corpus.walk_statistics.get('unique_sequences', 1)),
                'diversity_score': self.corpus.walk_statistics.get('sequence_diversity', 0),
                'avg_walk_length': self.corpus.walk_statistics.get('length_stats', {}).get('mean', 0)
            }
        
        return analysis
    
    def get_comprehensive_summary(self):
        """Get comprehensive summary for analysis."""
        summary = {
            'experiment_config': self.config,
            'graph_metadata': self.graph.get_summary(),
            'corpus_metadata': self.corpus.get_summary(),
            'analysis_metrics': self.analysis,
            'model_performance': self.model_performance
        }
        
        # NEW: Include trajectory analysis if available
        if self.evaluation_trajectories:
            summary['trajectory_analysis'] = self.evaluation_trajectories.get_summary()
        
        return summary
    
    def export_for_analysis(self, include_detailed=False):
        """Export metadata in format suitable for external analysis tools."""
        export_data = self.get_comprehensive_summary()
        
        if include_detailed:
            export_data['detailed'] = {
                'graph': {
                    'node_rule_mapping': self.graph.node_rule_mapping,
                    'rule_composition_full': self.graph.rule_composition
                },
                'corpus': {
                    'rule_node_frequencies': dict(self.corpus.rule_exposure.get('rule_node_frequencies', {})),
                    'structural_patterns': self.corpus.structural_analysis,
                    'vocabulary_analysis': self.corpus.vocabulary_analysis
                }
            }
        
        return export_data


class WalkTrajectoryMetadata:
    """
    Comprehensive trajectory analysis for individual walks.
    Tracks uncertainty metrics and probability distributions at each step.
    """
    
    def __init__(self, walk_idx: int, start_walk: List[int], generated_walk: Optional[List[int]] = None):
        self.walk_idx = walk_idx
        self.start_walk = start_walk
        self.generated_walk = generated_walk or []
        
        # Termination details
        self.termination_reason = None  # "completed", "invalid_edge", "end_token", "invalid_token", "max_length"
        self.termination_step = None
        self.final_length = 0
        
        # Full probability distributions at each step (stored as numpy arrays)
        self.probability_distributions = []  # List of full vocab-sized probability vectors
        
        # Step-by-step uncertainty metrics (lists indexed by step)
        self.uncertainty_trajectory = {
            'kl_divergences': {  # KL divergence from various baselines
                'uniform_random': [],
                'valid_neighbors': [],
                'degree_weighted': [],
                'graph_structure': [],
                'exponential_mle': []
            },
            'ks_distances': {  # Kolmogorov-Smirnov distances
                'uniform_random': [],
                'valid_neighbors': [],
                'degree_weighted': [],
                'graph_structure': [],
                'exponential_mle': []
            },
            'js_divergences': {},  # Jensen-Shannon divergences (symmetric KL)
            'wasserstein_distances': {},  # Earth Mover's Distance
            'entropies': [],
            'confidences': [],
            'perplexities': [],
            'cumulative_violation_probs': [],
            'skill_scores': {},  # Performance vs baselines
            'information_gains': {},  # Entropy reduction from baselines
            'relative_entropies': {},  # Ratio of entropies
            'normalized_surprises': {}  # Surprise vs baseline surprise
        }
        
        # Distribution statistics at each step
        self.distribution_stats = {
            'top_k_mass': [],  # Mass in top k predictions [top1, top5, top10]
            'effective_support_size': [],  # Number of nodes with non-negligible probability
            'distribution_concentration': [],  # Gini coefficient or similar
            'mode_probability': [],  # Probability of most likely node
            'tail_mass': [],  # Probability mass in unlikely outcomes
            'valid_neighbor_mass': [],  # Probability on valid graph neighbors
            'invalid_edge_mass': []  # Probability on invalid edges
        }
        
        # Violation tracking
        self.violation_timeline = []  # List of (step, violation_type, context) tuples
        self.critical_points = []  # Steps where uncertainty sharply changes
        self.rule_violation_probabilities = []  # Probability mass on rule-violating tokens at each step
        
        # Recovery patterns
        self.recovery_events = []  # Steps where model recovers from high uncertainty
        
        # Aggregate trajectory metrics (computed after walk completion)
        self.trajectory_statistics = {}
    
    def add_step_metrics(self, step_idx: int, metrics: Dict[str, Any]):
        """Add metrics for a single step in the walk."""
        # Add to various trajectory lists
        if 'entropy' in metrics:
            self.uncertainty_trajectory['entropies'].append(metrics['entropy'])
        if 'confidence' in metrics:
            self.uncertainty_trajectory['confidences'].append(metrics['confidence'])
        if 'perplexity' in metrics:
            self.uncertainty_trajectory['perplexities'].append(metrics['perplexity'])
        
        # Add KL divergences
        if 'kl_divergences' in metrics:
            for baseline, value in metrics['kl_divergences'].items():
                if baseline in self.uncertainty_trajectory['kl_divergences']:
                    self.uncertainty_trajectory['kl_divergences'][baseline].append(value)
        
        # Add KS distances
        if 'ks_distances' in metrics:
            for baseline, value in metrics['ks_distances'].items():
                if baseline in self.uncertainty_trajectory['ks_distances']:
                    self.uncertainty_trajectory['ks_distances'][baseline].append(value)
        
        # Add distribution statistics
        if 'distribution_stats' in metrics:
            stats = metrics['distribution_stats']
            if 'top_k_mass' in stats:
                self.distribution_stats['top_k_mass'].append(stats['top_k_mass'])
            if 'effective_support_size' in stats:
                self.distribution_stats['effective_support_size'].append(stats['effective_support_size'])
            if 'mode_probability' in stats:
                self.distribution_stats['mode_probability'].append(stats['mode_probability'])
            if 'tail_mass' in stats:
                self.distribution_stats['tail_mass'].append(stats['tail_mass'])
    
    def detect_critical_point(self, step_idx: int, threshold: float = 0.5):
        """Detect if current step is a critical point (sharp uncertainty change)."""
        if len(self.uncertainty_trajectory['entropies']) < 2:
            return False
        
        current_entropy = self.uncertainty_trajectory['entropies'][-1]
        previous_entropy = self.uncertainty_trajectory['entropies'][-2]
        entropy_change = abs(current_entropy - previous_entropy)
        
        if entropy_change > threshold:
            self.critical_points.append((step_idx, 'entropy_spike', entropy_change))
            return True
        
        # Also check for sharp confidence drops
        if len(self.uncertainty_trajectory['confidences']) >= 2:
            current_conf = self.uncertainty_trajectory['confidences'][-1]
            previous_conf = self.uncertainty_trajectory['confidences'][-2]
            conf_change = previous_conf - current_conf  # Drop is positive
            
            if conf_change > threshold:
                self.critical_points.append((step_idx, 'confidence_drop', conf_change))
                return True
        
        return False
    
    def compute_statistics(self):
        """Compute aggregate statistics after walk completion."""
        if not self.uncertainty_trajectory['entropies']:
            return
        
        self.trajectory_statistics = {
            'mean_entropy': np.mean(self.uncertainty_trajectory['entropies']),
            'std_entropy': np.std(self.uncertainty_trajectory['entropies']),
            'mean_confidence': np.mean(self.uncertainty_trajectory['confidences']),
            'std_confidence': np.std(self.uncertainty_trajectory['confidences']),
            'num_critical_points': len(self.critical_points),
            'num_violations': len(self.violation_timeline),
            'termination_reason': self.termination_reason,
            'walk_length': self.final_length
        }
        
        # Compute trajectory slope (trend)
        if len(self.uncertainty_trajectory['entropies']) > 1:
            steps = np.arange(len(self.uncertainty_trajectory['entropies']))
            entropy_slope = np.polyfit(steps, self.uncertainty_trajectory['entropies'], 1)[0]
            self.trajectory_statistics['entropy_slope'] = entropy_slope
        
        # Compute average KL divergences
        for baseline in self.uncertainty_trajectory['kl_divergences']:
            if self.uncertainty_trajectory['kl_divergences'][baseline]:
                mean_kl = np.mean(self.uncertainty_trajectory['kl_divergences'][baseline])
                self.trajectory_statistics[f'mean_kl_{baseline}'] = mean_kl
        
        # Compute average KS distances
        for baseline in self.uncertainty_trajectory['ks_distances']:
            if self.uncertainty_trajectory['ks_distances'][baseline]:
                mean_ks = np.mean(self.uncertainty_trajectory['ks_distances'][baseline])
                self.trajectory_statistics[f'mean_ks_{baseline}'] = mean_ks
    
    def get_summary(self):
        """Get summary dictionary for export."""
        return {
            'walk_idx': self.walk_idx,
            'termination_reason': self.termination_reason,
            'termination_step': self.termination_step,
            'final_length': self.final_length,
            'statistics': self.trajectory_statistics,
            'num_critical_points': len(self.critical_points),
            'num_violations': len(self.violation_timeline)
        }


class EvaluationTrajectoryMetadata:
    """
    Aggregate trajectory analysis across all evaluation walks.
    """
    
    def __init__(self, walk_trajectories: List[WalkTrajectoryMetadata]):
        self.walk_trajectories = walk_trajectories
        self.num_walks = len(walk_trajectories)
        
        # Group trajectories by termination reason
        self.outcome_groups = self._group_by_outcome()
        
        # Statistical trajectory analysis
        self.trajectory_patterns = self._analyze_trajectory_patterns()
        self.divergence_analysis = self._analyze_divergence_points()
        self.phase_transitions = self._identify_phase_transitions()
        
        # Progressive difficulty analysis
        self.progressive_metrics = self._compute_progressive_metrics()
        
        # Trajectory clustering
        self.trajectory_clusters = self._cluster_trajectories()
        
        # Critical step identification
        self.critical_steps = self._identify_critical_steps()
    
    def _group_by_outcome(self):
        """Group trajectories by their termination reason."""
        groups = defaultdict(list)
        for traj in self.walk_trajectories:
            groups[traj.termination_reason].append(traj)
        return dict(groups)
    
    def _analyze_trajectory_patterns(self):
        """Identify common patterns in uncertainty evolution."""
        patterns = {
            'monotonic_increasing': [],  # Uncertainty increases steadily
            'monotonic_decreasing': [],  # Uncertainty decreases steadily
            'u_shaped': [],              # High → Low → High
            'inverted_u': [],            # Low → High → Low
            'volatile': [],              # High variance in uncertainty
            'stable': []                 # Low variance in uncertainty
        }
        
        for traj in self.walk_trajectories:
            if not traj.uncertainty_trajectory['entropies']:
                continue
            
            entropies = traj.uncertainty_trajectory['entropies']
            
            # Check for monotonic patterns
            if all(entropies[i] <= entropies[i+1] for i in range(len(entropies)-1)):
                patterns['monotonic_increasing'].append(traj.walk_idx)
            elif all(entropies[i] >= entropies[i+1] for i in range(len(entropies)-1)):
                patterns['monotonic_decreasing'].append(traj.walk_idx)
            
            # Check for U-shaped patterns
            if len(entropies) >= 3:
                mid_point = len(entropies) // 2
                if entropies[0] > entropies[mid_point] < entropies[-1]:
                    patterns['u_shaped'].append(traj.walk_idx)
                elif entropies[0] < entropies[mid_point] > entropies[-1]:
                    patterns['inverted_u'].append(traj.walk_idx)
            
            # Check for volatility
            if np.std(entropies) > 1.0:  # High variance threshold
                patterns['volatile'].append(traj.walk_idx)
            elif np.std(entropies) < 0.2:  # Low variance threshold
                patterns['stable'].append(traj.walk_idx)
        
        return patterns
    
    def _analyze_divergence_points(self):
        """Find where successful and failed walks diverge."""
        divergence_analysis = {}
        
        # Compare successful vs failed walks
        successful = self.outcome_groups.get('completed', [])
        failed = [t for t in self.walk_trajectories if t.termination_reason != 'completed']
        
        if successful and failed:
            # Find minimum common length
            min_length = min(
                min(len(t.uncertainty_trajectory['entropies']) for t in successful if t.uncertainty_trajectory['entropies']),
                min(len(t.uncertainty_trajectory['entropies']) for t in failed if t.uncertainty_trajectory['entropies'])
            )
            
            divergence_points = []
            for step in range(min_length):
                successful_entropies = [t.uncertainty_trajectory['entropies'][step] for t in successful 
                                       if len(t.uncertainty_trajectory['entropies']) > step]
                failed_entropies = [t.uncertainty_trajectory['entropies'][step] for t in failed 
                                   if len(t.uncertainty_trajectory['entropies']) > step]
                
                if successful_entropies and failed_entropies:
                    # Statistical test for difference
                    mean_diff = abs(np.mean(successful_entropies) - np.mean(failed_entropies))
                    if mean_diff > 0.5:  # Threshold for significant difference
                        divergence_points.append((step, mean_diff))
            
            divergence_analysis['divergence_points'] = divergence_points
            divergence_analysis['first_divergence'] = divergence_points[0] if divergence_points else None
        
        return divergence_analysis
    
    def _identify_phase_transitions(self):
        """Detect common phase transition points across walks."""
        all_critical_points = []
        for traj in self.walk_trajectories:
            for step, transition_type, magnitude in traj.critical_points:
                all_critical_points.append((step, transition_type, magnitude))
        
        # Group by step position
        transitions_by_step = defaultdict(list)
        for step, transition_type, magnitude in all_critical_points:
            transitions_by_step[step].append((transition_type, magnitude))
        
        # Identify common transition points
        phase_transitions = {
            'common_transition_steps': [],
            'transition_distribution': {}
        }
        
        for step, transitions in transitions_by_step.items():
            if len(transitions) >= self.num_walks * 0.1:  # At least 10% of walks
                phase_transitions['common_transition_steps'].append(step)
                phase_transitions['transition_distribution'][step] = len(transitions) / self.num_walks
        
        return phase_transitions
    
    def _compute_progressive_metrics(self):
        """Compute metrics showing how performance changes over walk progression."""
        progressive = {
            'entropy_by_step': defaultdict(list),
            'confidence_by_step': defaultdict(list),
            'kl_by_step': defaultdict(lambda: defaultdict(list))
        }
        
        for traj in self.walk_trajectories:
            for step, entropy in enumerate(traj.uncertainty_trajectory['entropies']):
                progressive['entropy_by_step'][step].append(entropy)
            
            for step, conf in enumerate(traj.uncertainty_trajectory['confidences']):
                progressive['confidence_by_step'][step].append(conf)
            
            for baseline in traj.uncertainty_trajectory['kl_divergences']:
                for step, kl in enumerate(traj.uncertainty_trajectory['kl_divergences'][baseline]):
                    progressive['kl_by_step'][baseline][step].append(kl)
        
        # Compute averages
        summary = {}
        for step, entropies in progressive['entropy_by_step'].items():
            summary[f'step_{step}_mean_entropy'] = np.mean(entropies)
            summary[f'step_{step}_std_entropy'] = np.std(entropies)
        
        return summary
    
    def _cluster_trajectories(self):
        """Cluster trajectories based on their uncertainty patterns."""
        # Simplified clustering - in practice would use sklearn or similar
        clusters = {
            'high_uncertainty': [],
            'low_uncertainty': [],
            'increasing_uncertainty': [],
            'decreasing_uncertainty': []
        }
        
        for traj in self.walk_trajectories:
            if 'mean_entropy' in traj.trajectory_statistics:
                mean_entropy = traj.trajectory_statistics['mean_entropy']
                
                if mean_entropy > 2.0:
                    clusters['high_uncertainty'].append(traj.walk_idx)
                elif mean_entropy < 1.0:
                    clusters['low_uncertainty'].append(traj.walk_idx)
                
                if 'entropy_slope' in traj.trajectory_statistics:
                    slope = traj.trajectory_statistics['entropy_slope']
                    if slope > 0.1:
                        clusters['increasing_uncertainty'].append(traj.walk_idx)
                    elif slope < -0.1:
                        clusters['decreasing_uncertainty'].append(traj.walk_idx)
        
        return clusters
    
    def _identify_critical_steps(self):
        """Identify steps that are critical across many walks."""
        step_criticality = defaultdict(int)
        
        for traj in self.walk_trajectories:
            for step, _, _ in traj.critical_points:
                step_criticality[step] += 1
        
        # Find steps that are critical for many walks
        critical_steps = []
        threshold = self.num_walks * 0.05  # 5% of walks
        
        for step, count in step_criticality.items():
            if count >= threshold:
                critical_steps.append((step, count / self.num_walks))
        
        return sorted(critical_steps)
    
    def get_summary(self):
        """Get summary dictionary for export."""
        return {
            'num_walks': self.num_walks,
            'outcome_distribution': {k: len(v) for k, v in self.outcome_groups.items()},
            'trajectory_patterns': {k: len(v) for k, v in self.trajectory_patterns.items()},
            'divergence_analysis': self.divergence_analysis,
            'phase_transitions': self.phase_transitions,
            'critical_steps': self.critical_steps,
            'clusters': {k: len(v) for k, v in self.trajectory_clusters.items()}
        }