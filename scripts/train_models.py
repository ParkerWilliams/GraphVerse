#!/usr/bin/env python3
"""
Large-Scale Model Training Script

Trains models for all context window sizes needed for the large-scale experiment.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.large_scale_config import LARGE_SCALE_CONFIG
from configs.medium_scale_config import MEDIUM_SCALE_CONFIG, get_repeater_config_for_context
from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.data.preparation import prepare_training_data
from graphverse.llm.training import train_model
from graphverse.data.preparation import WalkVocabulary
import torch


def load_graph_and_rules(graph_path="large_scale_graph"):
    """
    Load graph and rules from saved files.
    
    Args:
        graph_path: Path to graph files (without extension)
        
    Returns:
        Tuple of (graph, rules, rule_info)
    """
    print(f"Loading graph from {graph_path}...")
    
    # Load graph
    graph = Graph.load_graph(graph_path)
    
    # Load rule information
    with open(f"{graph_path}_rules.json", "r") as f:
        rule_info = json.load(f)
    
    # Recreate rule objects
    ascender_rule = AscenderRule(rule_info["ascender_nodes"])
    even_rule = EvenRule(rule_info["even_nodes"]) 
    repeater_rule = RepeaterRule(rule_info["repeater_nodes_dict"])
    
    rules = [ascender_rule, even_rule, repeater_rule]
    
    print(f"Graph loaded: {graph.n:,} vertices")
    print(f"Rules loaded: {len(ascender_rule.member_nodes)} ascenders, "
          f"{len(even_rule.member_nodes)} evens, {len(repeater_rule.member_nodes)} repeaters")
    
    return graph, rules, rule_info


def prepare_context_training_data(graph, rules, context_window_size, config, verbose=True):
    """
    Prepare training data optimized for a specific context window size.
    
    Args:
        graph: Graph object
        rules: List of rule objects
        context_window_size: Size of context window for this model
        config: Large scale configuration
        verbose: Whether to print progress
        
    Returns:
        Tuple of (training_data, vocab)
    """
    if verbose:
        print(f"\nPreparing training data for context window {context_window_size}...")
    
    # Calculate walk lengths for context boundary testing
    # Walks must be 3-4x context window size to provide rich examples
    # and ensure repeater patterns can complete even when started late in the walk
    min_walk_length = context_window_size * 3  # 3w minimum
    max_walk_length = context_window_size * 4  # 4w maximum
    
    # Number of training walks - use 100K as specified
    training_walks = 100000
    
    if verbose:
        print(f"  Walk length range: {min_walk_length}-{max_walk_length}")
        print(f"  Training walks: {training_walks:,}")
    
    # Generate training data
    start_time = time.time()
    training_data, vocab, corpus_metadata = prepare_training_data(
        graph=graph,
        num_walks=training_walks,
        min_length=min_walk_length,
        max_length=max_walk_length,
        rules=rules,
        verbose=verbose
    )
    
    if verbose:
        generation_time = time.time() - start_time
        print(f"  Training data generated in {generation_time:.1f} seconds")
        print(f"  Vocabulary size: {len(vocab.token2idx)}")
        print(f"  Training sequences: {len(training_data)}")
    
    return training_data, vocab


def train_context_model(training_data, vocab, context_window_size, config, output_dir, verbose=True):
    """
    Train a model for a specific context window size.
    
    Args:
        training_data: Training data sequences
        vocab: Vocabulary object
        context_window_size: Context window size for this model
        config: Large scale configuration  
        output_dir: Directory to save model
        verbose: Whether to print progress
        
    Returns:
        Trained model
    """
    if verbose:
        print(f"\nTraining model for context window {context_window_size}...")
    
    training_config = config["training"]
    
    # Adjust training parameters based on context size
    # Larger contexts may need more epochs or different learning rates
    epochs = training_config["epochs"]
    if context_window_size >= 128:
        epochs = int(epochs * 1.2)  # More training for larger contexts
    
    learning_rate = training_config["learning_rate"]
    if context_window_size <= 16:
        learning_rate *= 1.5  # Higher learning rate for smaller contexts
    
    if verbose:
        print(f"  Hidden size: {training_config['hidden_size']}")
        print(f"  Layers: {training_config['num_layers']}")
        print(f"  Heads: {training_config['num_heads']}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max length: {context_window_size}")
    
    # Train model
    start_time = time.time()
    model = train_model(
        training_data=training_data,
        vocab=vocab,
        hidden_size=training_config["hidden_size"],
        num_layers=training_config["num_layers"],
        num_heads=training_config["num_heads"],
        dropout=training_config["dropout"],
        batch_size=training_config["batch_size"],
        num_epochs=epochs,
        learning_rate=learning_rate,
        verbose=verbose
    )
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"  Training completed in {training_time/60:.1f} minutes")
    
    # Save model and vocab
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"model_ctx_{context_window_size}.pt")
    vocab_path = os.path.join(output_dir, f"vocab_ctx_{context_window_size}.pkl")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(vocab.token2idx),
            'hidden_size': training_config["hidden_size"],
            'num_layers': training_config["num_layers"],
            'num_heads': training_config["num_heads"],
            'dropout': training_config["dropout"]
        },
        'context_window_size': context_window_size,
        'training_time': training_time,
        'training_config': training_config
    }, model_path)
    
    # Save vocabulary
    import pickle
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    
    if verbose:
        print(f"  Model saved: {model_path}")
        print(f"  Vocab saved: {vocab_path}")
    
    return model


def train_all_models(graph_path="large_scale_graph", output_dir="large_scale_models", 
                    context_windows=None, verbose=True, config=None):
    """
    Train models for all context window sizes.
    
    Args:
        graph_path: Path to graph files
        output_dir: Directory to save all models
        context_windows: List of context window sizes (None for config default)
        verbose: Whether to print progress
        config: Configuration object (None uses LARGE_SCALE_CONFIG)
        
    Returns:
        Dictionary mapping context_window -> model_path
    """
    # Use provided config or default to large-scale
    if config is None:
        config = LARGE_SCALE_CONFIG
        scale_name = "LARGE-SCALE"
    else:
        scale_name = "MEDIUM-SCALE" if config == MEDIUM_SCALE_CONFIG else "CUSTOM-SCALE"
    
    if context_windows is None:
        context_windows = config["context_windows"]
    
    if verbose:
        print("=" * 80)
        print(f"{scale_name} MODEL TRAINING")
        print("=" * 80)
        print(f"Context windows: {context_windows}")
        print(f"Output directory: {output_dir}")
        print(f"Graph size: {config['n']:,} vertices")
        print(f"Training config: {config['training']['epochs']} epochs, {config['training']['hidden_size']} hidden size")
    
    # Load graph and rules
    graph, rules, rule_info = load_graph_and_rules(graph_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "scale_config": config,
            "context_windows": context_windows,
            "graph_path": graph_path,
            "rule_info": rule_info,
            "scale_type": scale_name.lower().replace('-', '_')
        }, f, indent=2)
    
    model_paths = {}
    total_start_time = time.time()
    
    for i, context_size in enumerate(context_windows):
        if verbose:
            print(f"\n{'='*50}")
            print(f"TRAINING MODEL {i+1}/{len(context_windows)}: CONTEXT {context_size}")
            print(f"{'='*50}")
            
        # Get context-specific repeater configuration for 4-bucket boundary testing
        repeater_config = get_repeater_config_for_context(context_size)
        if verbose:
            print(f"4-bucket boundary testing design:")
            print(f"  {repeater_config['bucket_design']}")
            print(f"  Expected: Learnable {repeater_config['learnable_k_values']} vs Challenging {repeater_config['challenging_k_values']}")
        
        try:
            # Prepare training data for this context size
            training_data, vocab = prepare_context_training_data(
                graph, rules, context_size, config, verbose
            )
            
            # Train model
            model = train_context_model(
                training_data, vocab, context_size, config, 
                output_dir, verbose
            )
            
            model_path = os.path.join(output_dir, f"model_ctx_{context_size}.pt")
            model_paths[context_size] = model_path
            
            if verbose:
                elapsed = time.time() - total_start_time
                remaining = len(context_windows) - (i + 1)
                avg_time = elapsed / (i + 1)
                eta = remaining * avg_time
                
                print(f"✅ Context {context_size} completed")
                print(f"   Elapsed: {elapsed/60:.1f} min, ETA: {eta/60:.1f} min")
                
        except Exception as e:
            print(f"❌ Error training model for context {context_size}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print("MODEL TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Models trained: {len(model_paths)}")
        print(f"Output directory: {output_dir}")
        
        print("\nModel files:")
        for ctx, path in model_paths.items():
            print(f"  Context {ctx}: {path}")
    
    return model_paths


def validate_models(model_paths, verbose=True):
    """
    Validate that all models were created successfully.
    
    Args:
        model_paths: Dictionary of context_window -> model_path
        verbose: Whether to print validation results
        
    Returns:
        True if all models are valid, False otherwise
    """
    if verbose:
        print("\n" + "=" * 40)
        print("MODEL VALIDATION")
        print("=" * 40)
    
    validation_passed = True
    
    for context_size, model_path in model_paths.items():
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                print(f"❌ Model file missing: {model_path}")
                validation_passed = False
                continue
            
            # Try to load model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check required fields
            required_fields = ['model_state_dict', 'model_config', 'context_window_size']
            for field in required_fields:
                if field not in checkpoint:
                    print(f"❌ Context {context_size}: Missing field '{field}'")
                    validation_passed = False
                    continue
            
            # Check context window matches
            if checkpoint['context_window_size'] != context_size:
                print(f"❌ Context {context_size}: Context size mismatch")
                validation_passed = False
                continue
            
            if verbose:
                config = checkpoint['model_config']
                training_time = checkpoint.get('training_time', 0)
                print(f"✅ Context {context_size}: {config['vocab_size']} vocab, "
                      f"{config['hidden_size']} hidden, {training_time/60:.1f}min training")
                
        except Exception as e:
            print(f"❌ Context {context_size}: Error loading model - {e}")
            validation_passed = False
    
    if verbose:
        if validation_passed:
            print("\n🎉 Model validation PASSED")
        else:
            print("\n❌ Model validation FAILED")
    
    return validation_passed


def main():
    """Main model training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for large-scale context boundary analysis")
    parser.add_argument("--graph", "-g", default="large_scale_graph",
                       help="Path to graph files (default: large_scale_graph)")
    parser.add_argument("--output", "-o", default="large_scale_models",
                       help="Output directory for models (default: large_scale_models)")
    parser.add_argument("--contexts", nargs="+", type=int,
                       help="Context window sizes to train (default: from config)")
    parser.add_argument("--medium-scale", action="store_true",
                       help="Use medium-scale configuration")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after training")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Choose configuration and set defaults based on scale
    config = MEDIUM_SCALE_CONFIG if args.medium_scale else LARGE_SCALE_CONFIG
    scale_name = "medium-scale" if args.medium_scale else "large-scale"
    context_windows = args.contexts if args.contexts else config["context_windows"]
    
    if verbose:
        print(f"Using {scale_name} configuration:")
        print(f"  Context windows: {context_windows}")
        print(f"  Training epochs: {config['training']['epochs']}")
        print(f"  Model size: {config['training']['hidden_size']} hidden, {config['training']['num_layers']} layers")
    
    try:
        # Train all models
        model_paths = train_all_models(
            graph_path=args.graph,
            output_dir=args.output,
            context_windows=context_windows,
            verbose=verbose,
            config=config
        )
        
        # Validate if requested
        if args.validate:
            validation_passed = validate_models(model_paths, verbose)
            if not validation_passed:
                print("❌ Validation failed!")
                sys.exit(1)
        
        if verbose:
            print(f"\n✅ All models successfully trained!")
            print("Ready for large-scale context boundary experiments!")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()