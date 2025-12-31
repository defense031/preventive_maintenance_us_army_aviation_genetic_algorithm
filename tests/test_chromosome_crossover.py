#!/usr/bin/env python3
"""
Unit tests for Chromosome crossover operator.

Tests the hybrid crossover implementation with component-specific strategies:
- feature_indices: Uniform crossover
- thresholds: Blend crossover (BLX-α)
- bucket_assignments: Uniform crossover
- tiebreak_feature: Random selection
- early_phase_window: Arithmetic mean ± noise
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.chromosome import Chromosome


class TestCrossoverBasic:
    """Basic crossover functionality tests."""

    def test_crossover_produces_valid_offspring(self):
        """Test that offspring pass validation."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        child1, child2 = parent1.crossover(parent2, crossover_rate=0.70, rng=rng)

        # Offspring should be valid Chromosome objects
        assert isinstance(child1, Chromosome)
        assert isinstance(child2, Chromosome)

        # Validate() is called automatically in __post_init__, but let's be explicit
        child1.validate()
        child2.validate()

    def test_crossover_maintains_structure(self):
        """Test that offspring have same structure as parents."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        child1, child2 = parent1.crossover(parent2, crossover_rate=0.70, rng=rng)

        # Check structure
        assert child1.tree_depth == parent1.tree_depth
        assert child1.n_features == parent1.n_features
        assert len(child1.feature_indices) == len(parent1.feature_indices)
        assert len(child1.thresholds) == len(parent1.thresholds)
        assert len(child1.bucket_assignments) == len(parent1.bucket_assignments)

        assert child2.tree_depth == parent2.tree_depth
        assert child2.n_features == parent2.n_features


class TestCrossoverGeneMixing:
    """Test that offspring inherit genes from both parents."""

    def test_genes_mixed_from_parents(self):
        """Test that offspring contain mixture of parent genes."""
        rng = np.random.default_rng(42)

        # Create diverse parents
        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Perform multiple crossovers and check gene mixing
        mixtures_found = 0
        for i in range(10):
            child1, child2 = parent1.crossover(parent2, crossover_rate=1.0, rng=rng)

            # Check if child has genes from both parents
            # For feature_indices: should have some from each parent (uniform crossover)
            matches_p1 = sum(1 for c, p in zip(child1.feature_indices, parent1.feature_indices) if c == p)
            matches_p2 = sum(1 for c, p in zip(child1.feature_indices, parent2.feature_indices) if c == p)

            # With uniform 50/50 crossover, we expect roughly half from each parent
            # But some might match both by chance, so check for non-trivial mixing
            if matches_p1 > 0 and matches_p2 > 0:
                mixtures_found += 1

        # At least some crossovers should show clear mixing
        assert mixtures_found >= 5, "Expected to see gene mixing from both parents"

    def test_blx_alpha_thresholds(self):
        """Test that threshold crossover uses BLX-α correctly."""
        rng = np.random.default_rng(42)

        # Create parents with distinct thresholds
        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        child1, child2 = parent1.crossover(parent2, crossover_rate=1.0, rng=rng)

        # Thresholds should be in [0, 1]
        assert all(0.0 <= t <= 1.0 for t in child1.thresholds)
        assert all(0.0 <= t <= 1.0 for t in child2.thresholds)

        # With BLX-α, offspring can be outside parent range (but clipped to [0,1])
        # Just verify they're valid
        assert len(child1.thresholds) == len(parent1.thresholds)
        assert len(child2.thresholds) == len(parent2.thresholds)


class TestCrossoverRate:
    """Test that crossover_rate controls operator behavior."""

    def test_crossover_rate_zero_returns_clones(self):
        """Test that crossover_rate=0.0 returns clones of parents."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # With crossover_rate=0.0, should always return clones
        child1, child2 = parent1.crossover(parent2, crossover_rate=0.0, rng=rng)

        # Child1 should be identical to parent1
        assert child1.feature_indices == parent1.feature_indices
        assert child1.thresholds == parent1.thresholds
        assert child1.bucket_assignments == parent1.bucket_assignments
        assert child1.tiebreak_feature == parent1.tiebreak_feature
        assert child1.early_phase_window == parent1.early_phase_window

        # Child2 should be identical to parent2
        assert child2.feature_indices == parent2.feature_indices
        assert child2.thresholds == parent2.thresholds
        assert child2.bucket_assignments == parent2.bucket_assignments
        assert child2.tiebreak_feature == parent2.tiebreak_feature
        assert child2.early_phase_window == parent2.early_phase_window

    def test_crossover_rate_one_always_crosses(self):
        """Test that crossover_rate=1.0 always performs crossover."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Perform multiple crossovers with rate=1.0
        crossovers_happened = 0
        for i in range(10):
            child1, child2 = parent1.crossover(parent2, crossover_rate=1.0, rng=rng)

            # Check if crossover happened (offspring different from parents)
            if (child1.feature_indices != parent1.feature_indices or
                child1.bucket_assignments != parent1.bucket_assignments):
                crossovers_happened += 1

        # With rate=1.0, crossover should happen every time (unless by chance identical)
        # Expect at least 9 out of 10 to show differences
        assert crossovers_happened >= 9, f"Expected crossover to happen consistently, got {crossovers_happened}/10"

    def test_crossover_rate_controls_frequency(self):
        """Test that crossover_rate affects crossover frequency."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # With low crossover_rate, should get more clones
        clones = 0
        for i in range(100):
            child1, child2 = parent1.crossover(parent2, crossover_rate=0.1, rng=rng)

            # Check if child1 is a clone of parent1
            if (child1.feature_indices == parent1.feature_indices and
                child1.bucket_assignments == parent1.bucket_assignments and
                child1.early_phase_window == parent1.early_phase_window):
                clones += 1

        # With rate=0.1, expect ~90% clones
        # Allow some variance (between 80-95%)
        assert 80 <= clones <= 95, f"Expected ~90 clones with crossover_rate=0.1, got {clones}"


class TestDeterminism:
    """Test that crossover is deterministic with fixed seed."""

    def test_same_seed_same_offspring(self):
        """Test that same seed produces same offspring."""
        # First run
        rng1 = np.random.default_rng(12345)
        parent1a = Chromosome.random(tree_depth=3, n_features=5, rng=rng1)
        parent2a = Chromosome.random(tree_depth=3, n_features=5, rng=rng1)
        child1a, child2a = parent1a.crossover(parent2a, crossover_rate=0.70, rng=rng1)

        # Second run with same seed
        rng2 = np.random.default_rng(12345)
        parent1b = Chromosome.random(tree_depth=3, n_features=5, rng=rng2)
        parent2b = Chromosome.random(tree_depth=3, n_features=5, rng=rng2)
        child1b, child2b = parent1b.crossover(parent2b, crossover_rate=0.70, rng=rng2)

        # Parents should be identical
        assert parent1a.feature_indices == parent1b.feature_indices
        assert parent2a.feature_indices == parent2b.feature_indices

        # Offspring should be identical
        assert child1a.feature_indices == child1b.feature_indices
        assert child1a.thresholds == child1b.thresholds
        assert child1a.bucket_assignments == child1b.bucket_assignments
        assert child1a.tiebreak_feature == child1b.tiebreak_feature
        assert child1a.early_phase_window == child1b.early_phase_window

        assert child2a.feature_indices == child2b.feature_indices
        assert child2a.thresholds == child2b.thresholds


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_incompatible_tree_depth(self):
        """Test that crossover raises error for incompatible tree depths."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=2, n_features=5, rng=rng)

        with pytest.raises(ValueError, match="different tree depths"):
            parent1.crossover(parent2, crossover_rate=0.70, rng=rng)

    def test_incompatible_n_features(self):
        """Test that crossover raises error for incompatible feature counts."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=4, rng=rng)

        with pytest.raises(ValueError, match="different feature counts"):
            parent1.crossover(parent2, crossover_rate=0.70, rng=rng)

    def test_identical_parents(self):
        """Test crossover with identical parents."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Clone parent1
        parent2 = Chromosome(
            feature_indices=parent1.feature_indices.copy(),
            thresholds=parent1.thresholds.copy(),
            bucket_assignments=parent1.bucket_assignments.copy(),
            tiebreak_feature=parent1.tiebreak_feature,
            early_phase_window=parent1.early_phase_window,
            tree_depth=parent1.tree_depth,
            n_features=parent1.n_features
        )

        # Crossover should still work
        child1, child2 = parent1.crossover(parent2, crossover_rate=1.0, rng=rng)

        # Offspring should be valid
        child1.validate()
        child2.validate()


class TestCrossoverComponents:
    """Test individual crossover component strategies."""

    def test_early_phase_window_arithmetic_mean(self):
        """Test that early_phase_window uses arithmetic mean ± noise."""
        rng = np.random.default_rng(42)

        # Create parents with specific windows
        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent1_dict = parent1.to_dict()
        parent1_dict['early_phase_window'] = 20
        parent1 = Chromosome.from_dict(parent1_dict)

        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2_dict = parent2.to_dict()
        parent2_dict['early_phase_window'] = 80
        parent2 = Chromosome.from_dict(parent2_dict)

        # Mean should be 50
        children_windows = []
        for i in range(20):
            child1, child2 = parent1.crossover(parent2, crossover_rate=1.0, rng=rng)
            children_windows.append(child1.early_phase_window)
            children_windows.append(child2.early_phase_window)

        # Windows should cluster around mean of 50 (with noise ±10)
        mean_window = np.mean(children_windows)
        assert 40 <= mean_window <= 60, f"Expected mean ~50, got {mean_window}"

        # All windows should be in valid range [0, 100]
        assert all(0 <= w <= 100 for w in children_windows)

    def test_tiebreak_feature_random_selection(self):
        """Test that tiebreak_feature is randomly selected from parents."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent1_dict = parent1.to_dict()
        parent1_dict['tiebreak_feature'] = 0
        parent1 = Chromosome.from_dict(parent1_dict)

        parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
        parent2_dict = parent2.to_dict()
        parent2_dict['tiebreak_feature'] = 4
        parent2 = Chromosome.from_dict(parent2_dict)

        # Collect tiebreak features from offspring
        tiebreaks = []
        for i in range(50):
            child1, child2 = parent1.crossover(parent2, crossover_rate=1.0, rng=rng)
            tiebreaks.append(child1.tiebreak_feature)
            tiebreaks.append(child2.tiebreak_feature)

        # Should see both 0 and 4 in offspring
        assert 0 in tiebreaks, "Expected some offspring with tiebreak=0"
        assert 4 in tiebreaks, "Expected some offspring with tiebreak=4"

        # Rough 50/50 split (allow 30-70% range)
        count_0 = sum(1 for t in tiebreaks if t == 0)
        ratio = count_0 / len(tiebreaks)
        assert 0.3 <= ratio <= 0.7, f"Expected ~50% split, got {ratio:.1%}"


def test_crossover_with_default_parameters():
    """Test crossover with default parameters (crossover_rate=0.70)."""
    rng = np.random.default_rng(42)

    parent1 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
    parent2 = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

    # Use default crossover_rate
    child1, child2 = parent1.crossover(parent2, rng=rng)

    # Should produce valid offspring
    child1.validate()
    child2.validate()

    # Should have expected structure
    assert child1.tree_depth == 3
    assert child1.n_features == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
