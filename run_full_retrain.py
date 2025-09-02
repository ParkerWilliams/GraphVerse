#!/usr/bin/env python3
"""
Full retraining script with fixed walk generation.
This runs the complete retraining pipeline with proper parameters.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    """Run the full retraining pipeline."""
    
    print("="*70)
    print("FULL MODEL RETRAINING WITH FIXED WALK GENERATION")
    print("="*70)
    print()
    print("This script will:")
    print("  1. Create a pre-built dense graph (1000 nodes, 40% density)")
    print("  2. Generate 100,000 training walks using ONLY existing edges")
    print("  3. Train a new model from scratch")
    print("  4. Evaluate to verify proper graph edge following")
    print()
    print("Expected outcomes:")
    print("  - No edges added during walk generation")
    print("  - Model learns to follow actual graph structure")
    print("  - Broken graph rate should be <10%")
    print()
    print("Configuration:")
    print("  - Graph: 1000 nodes, ~200K edges")
    print("  - Training: 100K walks, 15 epochs")
    print("  - Context window: 16 tokens")
    print("  - Walk length: 32 tokens")
    print("  - Repeater k-values: [8, 14, 18, 24]")
    print()
    print("-"*70)
    
    # Ask for confirmation
    response = input("\nProceed with full retraining? This will take ~30-60 minutes. (y/n): ")
    if response.lower() != 'y':
        print("Retraining cancelled.")
        return
    
    print("\n" + "="*70)
    print("STARTING RETRAINING")
    print("="*70)
    
    # Record start time
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run the retrain_fixed_model.py script
    script_path = Path(__file__).parent / "retrain_fixed_model.py"
    
    try:
        # Use the venv python
        venv_python = Path(__file__).parent / "venv" / "bin" / "python"
        if not venv_python.exists():
            # Fallback to system python
            venv_python = "python3"
        
        # Run the training
        result = subprocess.run(
            [str(venv_python), str(script_path)],
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print("\n" + "="*70)
        print("RETRAINING COMPLETE")
        print("="*70)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {elapsed}")
        print()
        print("✅ SUCCESS: Model has been retrained with fixed walk generation!")
        print()
        print("The retrained model and data are saved in:")
        print("  experiments/fixed_run_[timestamp]/")
        print()
        print("Next steps:")
        print("  1. Run evaluation on the new model to verify performance")
        print("  2. Compare with the original broken model")
        print("  3. Use the new model for further experiments")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Retraining failed with exit code {e.returncode}")
        print("Check the output above for error details.")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Retraining interrupted by user.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())