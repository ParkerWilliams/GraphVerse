"""
Advanced checkpointing system for GraphVerse large-scale experiments.
Handles state persistence, recovery, and incremental progress saving.
"""

import os
import json
import pickle
import time
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib


class CheckpointManager:
    """
    Manages experiment checkpoints with versioning, validation, and recovery.
    """
    
    def __init__(self, experiment_folder: str, max_checkpoints: int = 10):
        """
        Initialize checkpoint manager.
        
        Args:
            experiment_folder: Main experiment folder
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.experiment_folder = experiment_folder
        self.checkpoint_folder = os.path.join(experiment_folder, "checkpoints")
        self.max_checkpoints = max_checkpoints
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        
        # State tracking
        self.checkpoint_history = []
        self.load_checkpoint_history()
    
    def save_checkpoint(self, 
                       state: Dict[str, Any], 
                       checkpoint_name: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> str:
        """
        Save a checkpoint with state and metadata.
        
        Args:
            state: State dictionary to save
            checkpoint_name: Optional name for checkpoint
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{timestamp}"
        
        # Create checkpoint data
        checkpoint_data = {
            "timestamp": timestamp,
            "checkpoint_name": checkpoint_name,
            "state": state,
            "metadata": metadata or {},
            "experiment_folder": self.experiment_folder,
            "checkpoint_version": "1.0"
        }
        
        # Calculate state hash for validation
        state_hash = self._calculate_state_hash(state)
        checkpoint_data["state_hash"] = state_hash
        
        # Save checkpoint
        checkpoint_file = os.path.join(self.checkpoint_folder, f"{checkpoint_name}.pkl")
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        # Update history
        self.checkpoint_history.append({
            "timestamp": timestamp,
            "name": checkpoint_name,
            "file": checkpoint_file,
            "state_hash": state_hash,
            "metadata": metadata or {}
        })
        
        # Save history
        self.save_checkpoint_history()
        
        # Create latest symlink
        latest_file = os.path.join(self.checkpoint_folder, "latest.pkl")
        if os.path.exists(latest_file):
            os.remove(latest_file)
        shutil.copy2(checkpoint_file, latest_file)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint by name or latest.
        
        Args:
            checkpoint_name: Name of checkpoint to load, or None for latest
            
        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_name is None:
            # Load latest checkpoint
            latest_file = os.path.join(self.checkpoint_folder, "latest.pkl")
            if not os.path.exists(latest_file):
                return None
            checkpoint_file = latest_file
        else:
            checkpoint_file = os.path.join(self.checkpoint_folder, f"{checkpoint_name}.pkl")
            if not os.path.exists(checkpoint_file):
                return None
        
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint
            if self.validate_checkpoint(checkpoint_data):
                return checkpoint_data
            else:
                print(f"Warning: Checkpoint validation failed for {checkpoint_file}")
                return None
                
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_file}: {e}")
            return None
    
    def validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_data: Checkpoint data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["timestamp", "checkpoint_name", "state", "state_hash"]
            for field in required_fields:
                if field not in checkpoint_data:
                    print(f"Missing required field: {field}")
                    return False
            
            # Validate state hash
            current_hash = self._calculate_state_hash(checkpoint_data["state"])
            if current_hash != checkpoint_data["state_hash"]:
                print("State hash mismatch - checkpoint may be corrupted")
                return False
            
            return True
            
        except Exception as e:
            print(f"Checkpoint validation error: {e}")
            return False
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        return self.checkpoint_history.copy()
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to delete
            
        Returns:
            True if deleted, False if not found
        """
        checkpoint_file = os.path.join(self.checkpoint_folder, f"{checkpoint_name}.pkl")
        
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            
            # Remove from history
            self.checkpoint_history = [
                cp for cp in self.checkpoint_history 
                if cp["name"] != checkpoint_name
            ]
            self.save_checkpoint_history()
            
            return True
        
        return False
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and keep most recent
        sorted_checkpoints = sorted(
            self.checkpoint_history, 
            key=lambda x: x["timestamp"]
        )
        
        # Delete oldest checkpoints
        to_delete = sorted_checkpoints[:-self.max_checkpoints]
        for checkpoint in to_delete:
            try:
                if os.path.exists(checkpoint["file"]):
                    os.remove(checkpoint["file"])
                print(f"Deleted old checkpoint: {checkpoint['name']}")
            except Exception as e:
                print(f"Error deleting checkpoint {checkpoint['name']}: {e}")
        
        # Update history
        self.checkpoint_history = sorted_checkpoints[-self.max_checkpoints:]
        self.save_checkpoint_history()
    
    def save_checkpoint_history(self):
        """Save checkpoint history to disk."""
        history_file = os.path.join(self.checkpoint_folder, "checkpoint_history.json")
        with open(history_file, "w") as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def load_checkpoint_history(self):
        """Load checkpoint history from disk."""
        history_file = os.path.join(self.checkpoint_folder, "checkpoint_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    self.checkpoint_history = json.load(f)
            except Exception as e:
                print(f"Error loading checkpoint history: {e}")
                self.checkpoint_history = []
        else:
            self.checkpoint_history = []
    
    def _calculate_state_hash(self, state: Dict[str, Any]) -> str:
        """Calculate hash of state for validation."""
        # Convert state to string for hashing (simplified approach)
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about checkpoint system."""
        return {
            "experiment_folder": self.experiment_folder,
            "checkpoint_folder": self.checkpoint_folder,
            "num_checkpoints": len(self.checkpoint_history),
            "max_checkpoints": self.max_checkpoints,
            "latest_checkpoint": self.checkpoint_history[-1] if self.checkpoint_history else None,
            "total_size_mb": self._calculate_checkpoint_folder_size() / 1024**2
        }
    
    def _calculate_checkpoint_folder_size(self) -> int:
        """Calculate total size of checkpoint folder in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(self.checkpoint_folder):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size


class ProgressTracker:
    """
    Tracks progress of long-running experiments with automatic checkpointing.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 total_items: int,
                 checkpoint_manager: CheckpointManager,
                 checkpoint_frequency: int = 1000):
        """
        Initialize progress tracker.
        
        Args:
            experiment_name: Name of experiment
            total_items: Total number of items to process
            checkpoint_manager: CheckpointManager instance
            checkpoint_frequency: How often to checkpoint (number of items)
        """
        self.experiment_name = experiment_name
        self.total_items = total_items
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_frequency = checkpoint_frequency
        
        # Progress state
        self.completed_items = 0
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_items = 0
        
        # Statistics
        self.error_count = 0
        self.warning_count = 0
        self.checkpoint_count = 0
        
        # Load existing progress if available
        self.load_progress()
    
    def update_progress(self, items_completed: int = 1, **kwargs):
        """
        Update progress and checkpoint if needed.
        
        Args:
            items_completed: Number of items completed in this update
            **kwargs: Additional state to store in checkpoint
        """
        self.completed_items += items_completed
        
        # Check if we should checkpoint
        items_since_checkpoint = self.completed_items - self.last_checkpoint_items
        
        if (items_since_checkpoint >= self.checkpoint_frequency or 
            self.completed_items >= self.total_items):
            
            self.save_progress(**kwargs)
    
    def save_progress(self, **additional_state):
        """Save current progress to checkpoint."""
        current_time = time.time()
        
        progress_state = {
            "experiment_name": self.experiment_name,
            "completed_items": self.completed_items,
            "total_items": self.total_items,
            "start_time": self.start_time,
            "current_time": current_time,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "checkpoint_count": self.checkpoint_count,
            **additional_state
        }
        
        checkpoint_name = f"progress_{self.experiment_name}_{self.completed_items}"
        
        metadata = {
            "progress_percentage": (self.completed_items / self.total_items) * 100,
            "elapsed_time": current_time - self.start_time,
            "items_per_second": self.completed_items / (current_time - self.start_time) if current_time > self.start_time else 0,
            "estimated_completion": self.estimate_completion_time()
        }
        
        self.checkpoint_manager.save_checkpoint(
            state=progress_state,
            checkpoint_name=checkpoint_name,
            metadata=metadata
        )
        
        self.last_checkpoint_time = current_time
        self.last_checkpoint_items = self.completed_items
        self.checkpoint_count += 1
        
        print(f"Progress checkpoint: {self.completed_items}/{self.total_items} "
              f"({metadata['progress_percentage']:.1f}%) - "
              f"ETA: {metadata['estimated_completion']:.1f} minutes")
    
    def load_progress(self) -> bool:
        """
        Load progress from latest checkpoint.
        
        Returns:
            True if progress was loaded, False otherwise
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        
        if checkpoint_data and "state" in checkpoint_data:
            state = checkpoint_data["state"]
            
            # Check if this is our experiment
            if state.get("experiment_name") == self.experiment_name:
                self.completed_items = state.get("completed_items", 0)
                self.start_time = state.get("start_time", time.time())
                self.error_count = state.get("error_count", 0)
                self.warning_count = state.get("warning_count", 0)
                self.checkpoint_count = state.get("checkpoint_count", 0)
                
                print(f"Resumed experiment '{self.experiment_name}' from checkpoint:")
                print(f"  Completed: {self.completed_items}/{self.total_items}")
                print(f"  Progress: {(self.completed_items/self.total_items)*100:.1f}%")
                
                return True
        
        return False
    
    def estimate_completion_time(self) -> float:
        """
        Estimate time to completion in minutes.
        
        Returns:
            Estimated minutes to completion
        """
        if self.completed_items == 0:
            return float('inf')
        
        elapsed = time.time() - self.start_time
        rate = self.completed_items / elapsed
        remaining_items = self.total_items - self.completed_items
        
        if rate > 0:
            return remaining_items / rate / 60  # Convert to minutes
        else:
            return float('inf')
    
    def record_error(self, error_message: str = ""):
        """Record an error occurrence."""
        self.error_count += 1
        if error_message:
            print(f"Error recorded: {error_message}")
    
    def record_warning(self, warning_message: str = ""):
        """Record a warning occurrence."""
        self.warning_count += 1
        if warning_message:
            print(f"Warning recorded: {warning_message}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of current progress."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        return {
            "experiment_name": self.experiment_name,
            "completed_items": self.completed_items,
            "total_items": self.total_items,
            "progress_percentage": (self.completed_items / self.total_items) * 100,
            "elapsed_time_minutes": elapsed / 60,
            "items_per_second": self.completed_items / elapsed if elapsed > 0 else 0,
            "estimated_completion_minutes": self.estimate_completion_time(),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "checkpoint_count": self.checkpoint_count
        }


# Convenience functions for easy use
def create_experiment_checkpoint_manager(experiment_folder: str) -> CheckpointManager:
    """Create a checkpoint manager for an experiment."""
    return CheckpointManager(experiment_folder)


def create_progress_tracker(experiment_name: str, 
                          total_items: int, 
                          checkpoint_manager: CheckpointManager,
                          checkpoint_frequency: int = 1000) -> ProgressTracker:
    """Create a progress tracker with checkpointing."""
    return ProgressTracker(experiment_name, total_items, checkpoint_manager, checkpoint_frequency)


# Example usage
if __name__ == "__main__":
    # Example: Set up checkpointing for large experiment
    experiment_folder = "test_experiment"
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Create checkpoint manager
    checkpoint_manager = create_experiment_checkpoint_manager(experiment_folder)
    
    # Create progress tracker
    tracker = create_progress_tracker(
        experiment_name="large_scale_test",
        total_items=1000000,
        checkpoint_manager=checkpoint_manager,
        checkpoint_frequency=50000
    )
    
    print("Checkpoint system demo:")
    print(f"Checkpoint info: {checkpoint_manager.get_checkpoint_info()}")
    
    # Simulate progress
    for i in range(5):
        tracker.update_progress(10000, batch_number=i)
        time.sleep(0.1)  # Simulate work
    
    print(f"Final progress: {tracker.get_progress_summary()}")
    print(f"Checkpoints: {len(checkpoint_manager.list_checkpoints())}")