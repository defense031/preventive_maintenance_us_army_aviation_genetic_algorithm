"""
Checkpoint Manager for Genetic Algorithm

Handles saving and loading GA state for fault tolerance and resume capability.
Saves population, fitness scores, RNG state, and convergence history using pickle.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pickle
import numpy as np
from datetime import datetime
import shutil


class CheckpointManager:
    """Manage GA checkpoints for fault tolerance and resume capability.

    Saves complete GA state including population, fitness scores, generation number,
    RNG state, and convergence history. Supports automatic cleanup of old checkpoints.

    Attributes:
        save_dir: Directory to save checkpoints
        filename_pattern: Pattern for checkpoint filenames (e.g., "checkpoint_gen{generation:03d}.pkl")
        keep_last_n: Number of recent checkpoints to keep (0 = keep all)
        enabled: Whether checkpointing is enabled

    Example:
        >>> manager = CheckpointManager(save_dir="results/checkpoints", keep_last_n=3)
        >>> manager.save_checkpoint(generation=10, population=pop, rng=rng, ...)
        >>> state = manager.load_checkpoint("results/checkpoints/checkpoint_gen010.pkl")
    """

    def __init__(
        self,
        save_dir: str = "results/ga_checkpoints",
        filename_pattern: str = "checkpoint_gen{generation:03d}.pkl",
        keep_last_n: int = 3,
        enabled: bool = True
    ):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints (created if doesn't exist)
            filename_pattern: Filename pattern with {generation} placeholder
            keep_last_n: Keep only last N checkpoints (0 = keep all, default: 3)
            enabled: Enable checkpointing (default: True)

        Raises:
            ValueError: If keep_last_n < 0 or filename_pattern missing {generation}
        """
        if keep_last_n < 0:
            raise ValueError(f"keep_last_n must be non-negative, got {keep_last_n}")

        if "{generation" not in filename_pattern:
            raise ValueError(
                f"filename_pattern must contain '{{generation}}' placeholder, "
                f"got '{filename_pattern}'"
            )

        self.save_dir = Path(save_dir)
        self.filename_pattern = filename_pattern
        self.keep_last_n = keep_last_n
        self.enabled = enabled

        # Create directory if enabled
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        generation: int,
        population: Any,  # Population object
        fitness_scores: List[float],
        rng_state: Dict[str, Any],
        convergence_history: List[tuple],
        best_chromosome: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Save complete GA state to checkpoint file.

        Args:
            generation: Current generation number
            population: Population object with chromosomes
            fitness_scores: List of fitness scores for current population
            rng_state: NumPy RNG state dict (from rng.bit_generator.state)
            convergence_history: History from ConvergenceTracker
            best_chromosome: Best chromosome found so far (optional)
            metadata: Additional metadata to save (optional)

        Returns:
            Path to saved checkpoint file, or None if checkpointing disabled

        Example:
            >>> rng = np.random.default_rng(42)
            >>> manager.save_checkpoint(
            ...     generation=10,
            ...     population=pop,
            ...     fitness_scores=[0.8, 0.7, ...],
            ...     rng_state=rng.bit_generator.state,
            ...     convergence_history=tracker.get_history()
            ... )
        """
        if not self.enabled:
            return None

        # Create checkpoint dictionary
        checkpoint = {
            'generation': generation,
            'population': population,
            'fitness_scores': fitness_scores,
            'rng_state': rng_state,
            'convergence_history': convergence_history,
            'best_chromosome': best_chromosome,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        # Generate filename
        filename = self.filename_pattern.format(generation=generation)
        filepath = self.save_dir / filename

        # Save with pickle
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Failed to save checkpoint to {filepath}: {e}")

        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self.cleanup_old_checkpoints()

        return filepath

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load GA state from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing:
            - generation: Generation number
            - population: Population object
            - fitness_scores: List of fitness scores
            - rng_state: NumPy RNG state dict
            - convergence_history: Convergence history
            - best_chromosome: Best chromosome (if saved)
            - timestamp: Checkpoint creation timestamp
            - metadata: Additional metadata

        Raises:
            FileNotFoundError: If checkpoint file not found
            IOError: If checkpoint file is corrupted

        Example:
            >>> state = manager.load_checkpoint("results/checkpoints/checkpoint_gen010.pkl")
            >>> generation = state['generation']
            >>> population = state['population']
            >>> rng = np.random.default_rng()
            >>> rng.bit_generator.state = state['rng_state']
        """
        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

        # Validate checkpoint structure
        required_keys = ['generation', 'population', 'fitness_scores', 'rng_state', 'convergence_history']
        missing_keys = [key for key in required_keys if key not in checkpoint]

        if missing_keys:
            raise IOError(
                f"Corrupted checkpoint file (missing keys: {missing_keys}): {checkpoint_path}"
            )

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint file.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found

        Example:
            >>> latest = manager.get_latest_checkpoint()
            >>> if latest:
            ...     state = manager.load_checkpoint(latest)
        """
        if not self.save_dir.exists():
            return None

        # Find all checkpoint files matching pattern
        # Extract base pattern (everything before {generation})
        pattern_prefix = self.filename_pattern.split('{')[0]
        pattern_suffix = self.filename_pattern.split('}')[-1]

        checkpoint_files = list(self.save_dir.glob(f"{pattern_prefix}*{pattern_suffix}"))

        if not checkpoint_files:
            return None

        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return checkpoint_files[0]

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files sorted by generation (newest first).

        Returns:
            List of checkpoint file paths sorted by modification time

        Example:
            >>> checkpoints = manager.list_checkpoints()
            >>> print(f"Found {len(checkpoints)} checkpoints")
        """
        if not self.save_dir.exists():
            return []

        # Find all checkpoint files
        pattern_prefix = self.filename_pattern.split('{')[0]
        pattern_suffix = self.filename_pattern.split('}')[-1]

        checkpoint_files = list(self.save_dir.glob(f"{pattern_prefix}*{pattern_suffix}"))

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return checkpoint_files

    def cleanup_old_checkpoints(self) -> int:
        """Remove old checkpoints, keeping only the most recent N.

        Keeps the keep_last_n most recent checkpoints based on modification time.
        If keep_last_n=0, keeps all checkpoints.

        Returns:
            Number of checkpoints deleted

        Example:
            >>> manager = CheckpointManager(keep_last_n=3)
            >>> deleted = manager.cleanup_old_checkpoints()
            >>> print(f"Deleted {deleted} old checkpoints")
        """
        if self.keep_last_n == 0:
            return 0  # Keep all

        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= self.keep_last_n:
            return 0  # Nothing to delete

        # Delete older checkpoints
        to_delete = checkpoints[self.keep_last_n:]
        deleted_count = 0

        for checkpoint_path in to_delete:
            try:
                checkpoint_path.unlink()
                deleted_count += 1
            except Exception as e:
                # Log but don't fail on cleanup errors
                print(f"Warning: Failed to delete old checkpoint {checkpoint_path}: {e}")

        return deleted_count

    def delete_all_checkpoints(self) -> int:
        """Delete all checkpoints in save_dir.

        Use with caution - this removes all checkpoint files.

        Returns:
            Number of checkpoints deleted

        Example:
            >>> deleted = manager.delete_all_checkpoints()
        """
        checkpoints = self.list_checkpoints()
        deleted_count = 0

        for checkpoint_path in checkpoints:
            try:
                checkpoint_path.unlink()
                deleted_count += 1
            except Exception:
                pass

        return deleted_count

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get metadata about a checkpoint without loading full state.

        Loads only the metadata fields (generation, timestamp, etc.) without
        loading the full population and chromosomes.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata

        Example:
            >>> info = manager.get_checkpoint_info("checkpoint_gen010.pkl")
            >>> print(f"Generation {info['generation']} saved at {info['timestamp']}")
        """
        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load checkpoint info from {checkpoint_path}: {e}")

        # Return lightweight metadata only
        return {
            'generation': checkpoint.get('generation'),
            'timestamp': checkpoint.get('timestamp'),
            'metadata': checkpoint.get('metadata', {}),
            'has_best_chromosome': checkpoint.get('best_chromosome') is not None,
            'population_size': len(checkpoint.get('fitness_scores', [])),
            'file_size_mb': path.stat().st_size / (1024 * 1024)
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "enabled" if self.enabled else "disabled"
        return (
            f"CheckpointManager(save_dir='{self.save_dir}', "
            f"keep_last_n={self.keep_last_n}, {status})"
        )
