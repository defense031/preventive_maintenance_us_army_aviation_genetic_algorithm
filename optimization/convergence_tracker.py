"""
Convergence Tracker for Genetic Algorithm

Monitors fitness progression across generations and detects early stopping
conditions based on lack of improvement over a patience period.
"""

from typing import List, Tuple, Optional
import numpy as np


class ConvergenceTracker:
    """Track fitness progression and detect convergence for early stopping.

    Monitors best fitness over generations and triggers early stopping if no
    improvement is observed for a specified patience period.

    Attributes:
        patience: Number of generations without improvement before stopping
        min_delta: Minimum fitness improvement to count as progress
        best_fitness: Best fitness score observed so far
        best_generation: Generation where best fitness was achieved
        generations_since_improvement: Counter for early stopping
        history: List of (generation, best_fitness, mean_fitness, diversity) tuples

    Example:
        >>> tracker = ConvergenceTracker(patience=10, min_delta=0.001)
        >>> for gen in range(50):
        ...     should_stop = tracker.update(gen, best_fit, mean_fit, diversity)
        ...     if should_stop:
        ...         print(f"Early stopping at generation {gen}")
        ...         break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """Initialize convergence tracker.

        Args:
            patience: Generations without improvement before early stop (default: 10)
            min_delta: Minimum improvement threshold (default: 0.001)

        Raises:
            ValueError: If patience < 0 or min_delta < 0
        """
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")

        self.patience = patience
        self.min_delta = min_delta

        # Tracking state
        self.best_fitness: float = -np.inf
        self.best_generation: int = 0
        self.generations_since_improvement: int = 0

        # History: (generation, best_fitness, mean_fitness, diversity)
        self.history: List[Tuple[int, float, float, float]] = []

    def update(
        self,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        diversity: float
    ) -> bool:
        """Update tracker with current generation statistics.

        Args:
            generation: Current generation number
            best_fitness: Best fitness in current generation
            mean_fitness: Mean fitness across population
            diversity: Population diversity metric

        Returns:
            True if early stopping condition met, False otherwise

        Example:
            >>> tracker = ConvergenceTracker(patience=5)
            >>> should_stop = tracker.update(10, 0.85, 0.72, 0.35)
        """
        # Record history
        self.history.append((generation, best_fitness, mean_fitness, diversity))

        # Check for improvement
        improvement = best_fitness - self.best_fitness

        if improvement > self.min_delta:
            # Significant improvement found
            self.best_fitness = best_fitness
            self.best_generation = generation
            self.generations_since_improvement = 0
        else:
            # No significant improvement
            self.generations_since_improvement += 1

        # Check early stopping condition
        return self.has_converged()

    def has_converged(self) -> bool:
        """Check if optimization has converged (early stopping triggered).

        Returns:
            True if no improvement for patience generations, False otherwise

        Example:
            >>> if tracker.has_converged():
            ...     print("Optimization has converged")
        """
        return self.generations_since_improvement >= self.patience

    def get_history(self) -> List[Tuple[int, float, float, float]]:
        """Get complete convergence history.

        Returns:
            List of (generation, best_fitness, mean_fitness, diversity) tuples

        Example:
            >>> history = tracker.get_history()
            >>> for gen, best, mean, div in history:
            ...     print(f"Gen {gen}: Best={best:.4f}, Mean={mean:.4f}")
        """
        return self.history.copy()

    def get_improvement_summary(self) -> dict:
        """Get summary of fitness improvement over time.

        Returns:
            Dictionary containing:
            - best_fitness: Best fitness achieved
            - best_generation: When best fitness was found
            - total_generations: Total generations tracked
            - improvement_from_start: Fitness improvement from gen 0
            - generations_since_improvement: Stagnation count

        Example:
            >>> summary = tracker.get_improvement_summary()
            >>> print(f"Best: {summary['best_fitness']:.4f} at gen {summary['best_generation']}")
        """
        if not self.history:
            return {
                'best_fitness': -np.inf,
                'best_generation': 0,
                'total_generations': 0,
                'improvement_from_start': 0.0,
                'generations_since_improvement': 0
            }

        first_best = self.history[0][1]  # Best fitness from generation 0
        improvement = self.best_fitness - first_best

        return {
            'best_fitness': self.best_fitness,
            'best_generation': self.best_generation,
            'total_generations': len(self.history),
            'improvement_from_start': improvement,
            'generations_since_improvement': self.generations_since_improvement
        }

    def reset(self) -> None:
        """Reset tracker to initial state.

        Clears all history and resets counters.

        Example:
            >>> tracker.reset()
            >>> assert len(tracker.history) == 0
        """
        self.best_fitness = -np.inf
        self.best_generation = 0
        self.generations_since_improvement = 0
        self.history = []

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ConvergenceTracker(patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"best_fitness={self.best_fitness:.4f}, "
            f"generations_since_improvement={self.generations_since_improvement})"
        )
