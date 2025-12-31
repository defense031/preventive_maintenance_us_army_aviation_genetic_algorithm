#!/usr/bin/env python3
"""
Unit tests for ConvergenceTracker.

Tests convergence detection, early stopping, and fitness progression tracking.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.convergence_tracker import ConvergenceTracker


class TestConvergenceTrackerInitialization:
    """Test ConvergenceTracker initialization."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        tracker = ConvergenceTracker()

        assert tracker.patience == 10
        assert tracker.min_delta == 0.001
        assert tracker.best_fitness == float('-inf')
        assert tracker.best_generation == 0
        assert tracker.generations_since_improvement == 0
        assert len(tracker.history) == 0

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)

        assert tracker.patience == 5
        assert tracker.min_delta == 0.01

    def test_invalid_patience(self):
        """Test that negative patience raises error."""
        with pytest.raises(ValueError, match="patience must be non-negative"):
            ConvergenceTracker(patience=-1)

    def test_invalid_min_delta(self):
        """Test that negative min_delta raises error."""
        with pytest.raises(ValueError, match="min_delta must be non-negative"):
            ConvergenceTracker(min_delta=-0.001)


class TestConvergenceTracking:
    """Test convergence tracking and improvement detection."""

    def test_update_improves_best(self):
        """Test that update correctly tracks improvements."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)

        # First update
        should_stop = tracker.update(0, 0.50, 0.45, 0.8)

        assert tracker.best_fitness == 0.50
        assert tracker.best_generation == 0
        assert tracker.generations_since_improvement == 0
        assert not should_stop

    def test_update_no_improvement(self):
        """Test that stagnation is tracked correctly."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.01)

        # Initial improvement
        tracker.update(0, 0.50, 0.45, 0.8)

        # No significant improvements
        tracker.update(1, 0.505, 0.46, 0.75)  # < min_delta
        tracker.update(2, 0.502, 0.45, 0.78)  # < min_delta
        tracker.update(3, 0.501, 0.44, 0.80)  # < min_delta

        assert tracker.best_fitness == 0.50
        assert tracker.best_generation == 0
        assert tracker.generations_since_improvement == 3

    def test_update_with_significant_improvement(self):
        """Test that significant improvements reset counter."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)

        # Initial
        tracker.update(0, 0.50, 0.45, 0.8)

        # No improvement for 2 generations
        tracker.update(1, 0.505, 0.46, 0.75)
        tracker.update(2, 0.502, 0.47, 0.76)
        assert tracker.generations_since_improvement == 2

        # Significant improvement resets counter
        tracker.update(3, 0.65, 0.55, 0.70)
        assert tracker.best_fitness == 0.65
        assert tracker.best_generation == 3
        assert tracker.generations_since_improvement == 0

    def test_early_stopping_triggered(self):
        """Test that early stopping triggers after patience."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.01)

        # Initial
        tracker.update(0, 0.50, 0.45, 0.8)

        # No improvement
        should_stop = tracker.update(1, 0.505, 0.46, 0.75)
        assert not should_stop

        should_stop = tracker.update(2, 0.502, 0.47, 0.76)
        assert not should_stop

        should_stop = tracker.update(3, 0.501, 0.48, 0.77)
        assert should_stop  # Patience reached

        assert tracker.has_converged()

    def test_min_delta_threshold(self):
        """Test that min_delta correctly determines improvement."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.05)

        tracker.update(0, 0.50, 0.45, 0.8)

        # Small improvement (< min_delta)
        tracker.update(1, 0.53, 0.46, 0.75)
        assert tracker.generations_since_improvement == 1  # Not counted

        # Large improvement (> min_delta)
        tracker.update(2, 0.60, 0.50, 0.70)
        assert tracker.generations_since_improvement == 0  # Resets


class TestHistory:
    """Test history tracking."""

    def test_history_records_all_generations(self):
        """Test that history contains all generations."""
        tracker = ConvergenceTracker()

        for gen in range(5):
            tracker.update(gen, 0.5 + gen * 0.1, 0.4 + gen * 0.05, 0.8 - gen * 0.05)

        history = tracker.get_history()

        assert len(history) == 5

        # Check structure
        for i, (gen, best, mean, div) in enumerate(history):
            assert gen == i
            assert best == pytest.approx(0.5 + i * 0.1)
            assert mean == pytest.approx(0.4 + i * 0.05)
            assert div == pytest.approx(0.8 - i * 0.05)

    def test_history_is_copy(self):
        """Test that get_history returns a copy."""
        tracker = ConvergenceTracker()

        tracker.update(0, 0.5, 0.4, 0.8)
        history1 = tracker.get_history()

        tracker.update(1, 0.6, 0.5, 0.7)
        history2 = tracker.get_history()

        # Original history should not be modified
        assert len(history1) == 1
        assert len(history2) == 2


class TestImprovementSummary:
    """Test improvement summary generation."""

    def test_summary_empty(self):
        """Test summary with no updates."""
        tracker = ConvergenceTracker()

        summary = tracker.get_improvement_summary()

        assert summary['best_fitness'] == float('-inf')
        assert summary['best_generation'] == 0
        assert summary['total_generations'] == 0
        assert summary['improvement_from_start'] == 0.0
        assert summary['generations_since_improvement'] == 0

    def test_summary_with_improvement(self):
        """Test summary after improvements."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)

        tracker.update(0, 0.50, 0.45, 0.8)
        tracker.update(1, 0.55, 0.48, 0.75)
        tracker.update(2, 0.70, 0.60, 0.70)
        tracker.update(3, 0.69, 0.61, 0.72)

        summary = tracker.get_improvement_summary()

        assert summary['best_fitness'] == 0.70
        assert summary['best_generation'] == 2
        assert summary['total_generations'] == 4
        assert summary['improvement_from_start'] == pytest.approx(0.20)
        assert summary['generations_since_improvement'] == 1

    def test_summary_with_stagnation(self):
        """Test summary tracking stagnation."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)

        tracker.update(0, 0.50, 0.45, 0.8)

        # 5 generations without improvement
        for gen in range(1, 6):
            tracker.update(gen, 0.505, 0.46, 0.75)

        summary = tracker.get_improvement_summary()

        assert summary['best_fitness'] == 0.50
        assert summary['best_generation'] == 0
        assert summary['generations_since_improvement'] == 5


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)

        # Add some history
        for gen in range(3):
            tracker.update(gen, 0.5 + gen * 0.1, 0.4, 0.8)

        # Reset
        tracker.reset()

        assert tracker.best_fitness == float('-inf')
        assert tracker.best_generation == 0
        assert tracker.generations_since_improvement == 0
        assert len(tracker.history) == 0

    def test_reset_preserves_config(self):
        """Test that reset preserves patience and min_delta."""
        tracker = ConvergenceTracker(patience=7, min_delta=0.02)

        tracker.update(0, 0.5, 0.4, 0.8)
        tracker.reset()

        # Config should be preserved
        assert tracker.patience == 7
        assert tracker.min_delta == 0.02


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_patience(self):
        """Test that zero patience immediately triggers stopping."""
        tracker = ConvergenceTracker(patience=0, min_delta=0.01)

        tracker.update(0, 0.50, 0.45, 0.8)

        # Any non-improvement should trigger stop
        should_stop = tracker.update(1, 0.505, 0.46, 0.75)
        assert should_stop

    def test_large_patience(self):
        """Test with very large patience."""
        tracker = ConvergenceTracker(patience=100, min_delta=0.01)

        tracker.update(0, 0.50, 0.45, 0.8)

        # Many generations without stopping
        for gen in range(1, 50):
            should_stop = tracker.update(gen, 0.505, 0.46, 0.75)
            assert not should_stop

        assert tracker.generations_since_improvement == 49

    def test_negative_fitness(self):
        """Test with negative fitness values."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.1)

        tracker.update(0, -0.5, -0.6, 0.8)
        assert tracker.best_fitness == -0.5

        tracker.update(1, -0.3, -0.4, 0.75)
        assert tracker.best_fitness == -0.3
        assert tracker.generations_since_improvement == 0

    def test_very_small_min_delta(self):
        """Test with very small min_delta."""
        tracker = ConvergenceTracker(patience=5, min_delta=1e-10)

        tracker.update(0, 0.5, 0.4, 0.8)

        # Tiny improvement should count
        tracker.update(1, 0.5 + 1e-9, 0.4, 0.75)
        assert tracker.generations_since_improvement == 0


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        tracker = ConvergenceTracker(patience=10, min_delta=0.001)
        tracker.update(0, 0.75, 0.65, 0.8)

        repr_str = repr(tracker)

        assert "ConvergenceTracker" in repr_str
        assert "patience=10" in repr_str
        assert "min_delta=0.001" in repr_str
        assert "best_fitness=0.7500" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
