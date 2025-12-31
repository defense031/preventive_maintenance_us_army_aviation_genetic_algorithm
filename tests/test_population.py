#!/usr/bin/env python3
"""
Unit tests for Population class.

Tests population management functionality including:
- Initialization and random generation
- Fitness tracking and updates
- Elitism selection (top-k)
- Statistical computation
- Diversity measurement
- Population replacement
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.population import Population
from policy.chromosome import Chromosome


class TestPopulationInitialization:
    """Test population initialization and basic properties."""

    def test_population_creation(self):
        """Test basic population creation."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)

        assert pop.size == 50
        assert pop.tree_depth == 3
        assert pop.n_features == 5
        assert pop.seed == 42
        assert pop.generation == 0
        assert len(pop.chromosomes) == 0  # Not yet initialized

    def test_invalid_population_size(self):
        """Test that invalid population size raises error."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            Population(size=0, tree_depth=3, n_features=5)

        with pytest.raises(ValueError, match="Population size must be positive"):
            Population(size=-10, tree_depth=3, n_features=5)

    def test_invalid_tree_depth(self):
        """Test that invalid tree depth raises error."""
        with pytest.raises(ValueError, match="Tree depth must be positive"):
            Population(size=50, tree_depth=0, n_features=5)

    def test_invalid_n_features(self):
        """Test that invalid n_features raises error."""
        with pytest.raises(ValueError, match="Number of features must be positive"):
            Population(size=50, tree_depth=3, n_features=0)

    def test_initialize_random(self):
        """Test random population initialization."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        assert len(pop.chromosomes) == 50
        assert len(pop.fitness_scores) == 50
        assert all(f == 0.0 for f in pop.fitness_scores)
        assert pop.generation == 0

        # All chromosomes should be valid
        for chrom in pop.chromosomes:
            assert isinstance(chrom, Chromosome)
            assert chrom.tree_depth == 3
            assert chrom.n_features == 5
            chrom.validate()

    def test_deterministic_initialization(self):
        """Test that same seed produces same population."""
        pop1 = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop1.initialize_random()

        pop2 = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop2.initialize_random()

        # Same seed should produce identical chromosomes
        for chrom1, chrom2 in zip(pop1.chromosomes, pop2.chromosomes):
            assert chrom1.feature_indices == chrom2.feature_indices
            assert chrom1.thresholds == chrom2.thresholds
            assert chrom1.bucket_assignments == chrom2.bucket_assignments

    def test_len_operator(self):
        """Test __len__ operator."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        assert len(pop) == 0  # Empty before initialization

        pop.initialize_random()
        assert len(pop) == 50


class TestFitnessManagement:
    """Test fitness score tracking and updates."""

    def test_update_fitness(self):
        """Test updating fitness for individual."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        pop.update_fitness(0, 0.75)
        assert pop.fitness_scores[0] == 0.75

        pop.update_fitness(25, 0.92)
        assert pop.fitness_scores[25] == 0.92

    def test_update_fitness_out_of_bounds(self):
        """Test that out-of-bounds index raises error."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        with pytest.raises(IndexError, match="out of bounds"):
            pop.update_fitness(50, 0.5)

        with pytest.raises(IndexError, match="out of bounds"):
            pop.update_fitness(-1, 0.5)

    def test_update_fitness_invalid_value(self):
        """Test that invalid fitness values raise errors."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        with pytest.raises(ValueError, match="Invalid fitness value"):
            pop.update_fitness(0, float('nan'))

        with pytest.raises(ValueError, match="Invalid fitness value"):
            pop.update_fitness(0, float('inf'))


class TestElitismSelection:
    """Test elitism and top-k selection."""

    def test_get_top_k(self):
        """Test getting top k individuals."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign known fitness values
        for i in range(50):
            pop.update_fitness(i, float(i))  # Fitness = 0, 1, 2, ..., 49

        # Get top 4
        elite = pop.get_top_k(k=4)

        assert len(elite) == 4
        # Top 4 should be individuals with fitness 49, 48, 47, 46
        assert elite[0][1] == 49.0
        assert elite[1][1] == 48.0
        assert elite[2][1] == 47.0
        assert elite[3][1] == 46.0

        # Chromosomes should be valid
        for chrom, fitness in elite:
            assert isinstance(chrom, Chromosome)

    def test_get_top_k_sorted(self):
        """Test that top-k results are sorted by fitness."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Random fitness values
        rng = np.random.default_rng(42)
        for i in range(50):
            pop.update_fitness(i, rng.uniform(0, 1))

        elite = pop.get_top_k(k=10)

        # Check sorted descending
        for i in range(len(elite) - 1):
            assert elite[i][1] >= elite[i+1][1]

    def test_get_top_k_invalid(self):
        """Test that invalid k raises errors."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        with pytest.raises(ValueError, match="k must be positive"):
            pop.get_top_k(k=0)

        with pytest.raises(ValueError, match="cannot exceed population size"):
            pop.get_top_k(k=51)

    def test_get_best(self):
        """Test getting single best individual."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign known fitness
        for i in range(50):
            pop.update_fitness(i, float(i))

        best_chrom, best_fitness = pop.get_best()

        assert best_fitness == 49.0
        assert isinstance(best_chrom, Chromosome)

    def test_get_worst(self):
        """Test getting single worst individual."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign known fitness
        for i in range(50):
            pop.update_fitness(i, float(i))

        worst_chrom, worst_fitness = pop.get_worst()

        assert worst_fitness == 0.0
        assert isinstance(worst_chrom, Chromosome)


class TestStatistics:
    """Test fitness statistics computation."""

    def test_get_statistics_known_values(self):
        """Test statistics with known fitness values."""
        pop = Population(size=10, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign known fitness: 0, 1, 2, ..., 9
        for i in range(10):
            pop.update_fitness(i, float(i))

        stats = pop.get_statistics()

        assert stats['best'] == 9.0
        assert stats['worst'] == 0.0
        assert stats['mean'] == pytest.approx(4.5)
        assert stats['median'] == pytest.approx(4.5)
        assert stats['std'] == pytest.approx(np.std(range(10)))

    def test_get_statistics_empty(self):
        """Test statistics with empty population."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        # Don't initialize

        stats = pop.get_statistics()

        # Should return zeros for empty population
        assert stats['best'] == 0.0
        assert stats['worst'] == 0.0
        assert stats['mean'] == 0.0
        assert stats['median'] == 0.0
        assert stats['std'] == 0.0

    def test_get_statistics_uniform(self):
        """Test statistics when all fitness equal."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # All fitness = 0.5
        for i in range(50):
            pop.update_fitness(i, 0.5)

        stats = pop.get_statistics()

        assert stats['best'] == 0.5
        assert stats['worst'] == 0.5
        assert stats['mean'] == 0.5
        assert stats['median'] == 0.5
        assert stats['std'] == 0.0  # No variation


class TestDiversity:
    """Test population diversity measurement."""

    def test_compute_diversity_random(self):
        """Test diversity computation on random population."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        diversity = pop.compute_diversity()

        # Random population should have non-zero diversity
        assert diversity > 0.0
        assert diversity <= 1.0

    def test_compute_diversity_identical(self):
        """Test diversity is zero for identical chromosomes."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Replace all with clones of first
        first = pop.chromosomes[0]
        pop.chromosomes = [
            Chromosome(
                feature_indices=first.feature_indices.copy(),
                thresholds=first.thresholds.copy(),
                bucket_assignments=first.bucket_assignments.copy(),
                tiebreak_feature=first.tiebreak_feature,
                early_phase_window=first.early_phase_window,
                tree_depth=first.tree_depth,
                n_features=first.n_features
            )
            for _ in range(50)
        ]

        diversity = pop.compute_diversity()

        assert diversity == 0.0

    def test_compute_diversity_small_population(self):
        """Test diversity computation with small population."""
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        diversity = pop.compute_diversity()

        # Should work with just 2 chromosomes
        assert 0.0 <= diversity <= 1.0


class TestPopulationReplacement:
    """Test population replacement functionality."""

    def test_replace_population(self):
        """Test replacing entire population."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Update some fitness
        for i in range(50):
            pop.update_fitness(i, float(i))

        # Create new generation
        new_gen = [Chromosome.random(tree_depth=3, n_features=5, rng=pop.rng) for _ in range(50)]

        pop.replace(new_gen)

        # Check replacement
        assert len(pop.chromosomes) == 50
        assert pop.chromosomes == new_gen
        assert all(f == 0.0 for f in pop.fitness_scores)  # Fitness reset
        assert pop.generation == 1  # Generation incremented

    def test_replace_wrong_size(self):
        """Test that replacing with wrong size raises error."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        wrong_size = [Chromosome.random(tree_depth=3, n_features=5, rng=pop.rng) for _ in range(40)]

        with pytest.raises(ValueError, match="must match original size"):
            pop.replace(wrong_size)

    def test_generation_counter(self):
        """Test that generation counter increments correctly."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        assert pop.generation == 0

        for gen in range(1, 6):
            new_gen = [Chromosome.random(tree_depth=3, n_features=5, rng=pop.rng) for _ in range(50)]
            pop.replace(new_gen)
            assert pop.generation == gen


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        repr_str = repr(pop)

        assert "Population" in repr_str
        assert "size=50" in repr_str
        assert "gen=0" in repr_str
        assert "depth=3" in repr_str
        assert "features=5" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
