#!/usr/bin/env python3
"""
Clean up the repository by organizing test scripts and removing old files.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_repository(dry_run=True):
    """
    Organize and clean up the repository.
    
    Args:
        dry_run: If True, only show what would be done without making changes
    """
    
    repo_root = Path(__file__).parent
    
    print("="*70)
    print("REPOSITORY CLEANUP")
    print("="*70)
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL CLEANUP'}")
    print()
    
    # Create archive directories
    archive_dir = repo_root / "archive"
    test_archive = archive_dir / "test_scripts"
    old_results = archive_dir / "old_results"
    
    if not dry_run:
        test_archive.mkdir(parents=True, exist_ok=True)
        old_results.mkdir(parents=True, exist_ok=True)
    
    # Patterns for test scripts to move
    test_patterns = [
        "test_*.py",
        "examine_*.py",
        "check_*.py",
        "view_*.py",
        "visualize_*.py",
        "clean_*.py"
    ]
    
    # Important scripts to keep in root
    keep_in_root = {
        "run_full_retrain.py",
        "retrain_fixed_model.py",
        "retrain_small_test.py",
        "verify_retrain_fix.py",
        "monitor_retrain.py",
        "cleanup_repository.py",
        "run_small_scale_analysis.py"
    }
    
    print("📁 Test Scripts to Archive:")
    print("-"*40)
    test_files = []
    for pattern in test_patterns:
        for file in repo_root.glob(pattern):
            if file.name not in keep_in_root:
                test_files.append(file)
                print(f"  {file.name}")
    
    if test_files:
        print(f"\nTotal: {len(test_files)} test scripts")
        if not dry_run:
            for file in test_files:
                dest = test_archive / file.name
                shutil.move(str(file), str(dest))
            print("✅ Moved to archive/test_scripts/")
    else:
        print("  None found")
    
    # Old result directories
    print("\n📁 Old Result Directories:")
    print("-"*40)
    result_patterns = [
        "small_results",
        "small_models",
        "outputs",
        "images"
    ]
    
    old_dirs = []
    for pattern in result_patterns:
        path = repo_root / pattern
        if path.exists() and path.is_dir():
            old_dirs.append(path)
            # Count contents
            num_files = sum(1 for _ in path.rglob("*") if _.is_file())
            print(f"  {pattern}/ ({num_files} files)")
    
    if old_dirs:
        if not dry_run:
            for dir_path in old_dirs:
                dest = old_results / dir_path.name
                if dest.exists():
                    # Add timestamp to avoid conflicts
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest = old_results / f"{dir_path.name}_{timestamp}"
                shutil.move(str(dir_path), str(dest))
            print("✅ Moved to archive/old_results/")
    else:
        print("  None found")
    
    # Old numpy files
    print("\n📁 Data Files:")
    print("-"*40)
    npy_files = list(repo_root.glob("*.npy"))
    pkl_files = list(repo_root.glob("*.pkl"))
    
    data_files = npy_files + pkl_files
    if data_files:
        for file in data_files:
            print(f"  {file.name}")
        
        if not dry_run:
            data_archive = archive_dir / "data_files"
            data_archive.mkdir(parents=True, exist_ok=True)
            for file in data_files:
                dest = data_archive / file.name
                shutil.move(str(file), str(dest))
            print("✅ Moved to archive/data_files/")
    else:
        print("  None found")
    
    # Visualization directory
    print("\n📁 Visualization Files:")
    print("-"*40)
    viz_dir = repo_root / "graphverse" / "visualization"
    if viz_dir.exists():
        num_files = sum(1 for _ in viz_dir.rglob("*") if _.is_file())
        print(f"  graphverse/visualization/ ({num_files} files)")
        
        if not dry_run:
            # Keep the main visualization module but archive generated files
            generated_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.jpg"))
            if generated_files:
                viz_archive = old_results / "visualizations"
                viz_archive.mkdir(exist_ok=True)
                for file in generated_files:
                    dest = viz_archive / file.name
                    shutil.move(str(file), str(dest))
                print(f"✅ Moved {len(generated_files)} image files to archive")
    else:
        print("  Not found")
    
    # Summary
    print("\n" + "="*70)
    print("CLEANUP SUMMARY")
    print("="*70)
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN - no files were actually moved.")
        print("   Run with --execute to perform the actual cleanup.")
    else:
        print("\n✅ Repository cleanup complete!")
        print("   Archived files are in: archive/")
    
    print("\n📂 Current repository structure:")
    print("  .")
    print("  ├── graphverse/          (core library)")
    print("  ├── configs/             (experiment configs)")
    print("  ├── scripts/             (main scripts)")
    print("  ├── experiments/         (model outputs)")
    print("  ├── archive/             (old files)")
    print("  │   ├── test_scripts/")
    print("  │   ├── old_results/")
    print("  │   └── data_files/")
    print("  ├── retrain_fixed_model.py")
    print("  ├── run_full_retrain.py")
    print("  └── verify_retrain_fix.py")
    
    return len(test_files) + len(old_dirs) + len(data_files)


def main():
    """Main cleanup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up the GraphVerse repository")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually perform the cleanup (default is dry run)")
    args = parser.parse_args()
    
    num_items = cleanup_repository(dry_run=not args.execute)
    
    if not args.execute and num_items > 0:
        print("\n💡 To execute the cleanup, run:")
        print("   python cleanup_repository.py --execute")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())