#!/usr/bin/env python3
"""
Unit tests for Selection strategies.

Tests tournament selection and roulette wheel selection for parent selection
in the genetic algorithm.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.selection import (
    tournament_selection,
    roulette_selection,
    select_parents
)
from policy.chromosome import Chromosome


class TestTournamentSelection:
    """Test tournament selection functionality."""

    def test_tournament_basic(self):
        """Test basic tournament selection."""
        rng = np.random.default_rng(42)

        # Create population
        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = [float(i) for i in range(10)]  # 0, 1, 2, ..., 9

        # Select parent
        parent = tournament_selection(pop, fitness, tournament_size=3, rng=rng)

        assert isinstance(parent, Chromosome)
        assert parent in pop

    def test_tournament_favors_high_fitness(self):
        """Test that tournament favors individuals with higher fitness."""
        rng = np.random.default_rng(42)

        # Create population with clear fitness hierarchy
        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(50)]
        fitness = list(range(50))  # 0 to 49

        # Run many tournaments and count selections
        selections = []
        for _ in range(1000):
            parent = tournament_selection(pop, fitness, tournament_size=3, rng=rng)
            selections.append(pop.index(parent))

        # Higher fitness individuals should be selected more often
        # Top 10 should be selected more than bottom 10
        top_10_count = sum(1 for idx in selections if idx >= 40)
        bottom_10_count = sum(1 for idx in selections if idx < 10)

        assert top_10_count > bottom_10_count

    def test_tournament_size_controls_pressure(self):
        """Test that larger tournament size increases selection pressure."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(50)]
        fitness = list(range(50))

        # Tournament size 2 (low pressure)
        selections_size2 = []
        for _ in range(500):
            rng_copy = np.random.default_rng(42 + len(selections_size2))
            parent = tournament_selection(pop, fitness, tournament_size=2, rng=rng_copy)
            selections_size2.append(pop.index(parent))

        # Tournament size 5 (high pressure)
        selections_size5 = []
        for _ in range(500):
            rng_copy = np.random.default_rng(42 + len(selections_size5))
            parent = tournament_selection(pop, fitness, tournament_size=5, rng=rng_copy)
            selections_size5.append(pop.index(parent))

        # Size 5 should select from top individuals more often
        avg_fitness_size2 = np.mean(selections_size2)
        avg_fitness_size5 = np.mean(selections_size5)

        assert avg_fitness_size5 > avg_fitness_size2

    def test_tournament_deterministic(self):
        """Test that same seed gives same results."""
        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=np.random.default_rng(i)) for i in range(10)]
        fitness = list(range(10))

        rng1 = np.random.default_rng(12345)
        parent1 = tournament_selection(pop, fitness, tournament_size=3, rng=rng1)

        rng2 = np.random.default_rng(12345)
        parent2 = tournament_selection(pop, fitness, tournament_size=3, rng=rng2)

        # Same seed should select same parent
        assert pop.index(parent1) == pop.index(parent2)

    def test_tournament_empty_population(self):
        """Test that empty population raises error."""
        with pytest.raises(ValueError, match="Population is empty"):
            tournament_selection([], [], tournament_size=3)

    def test_tournament_mismatched_sizes(self):
        """Test that mismatched population and fitness sizes raise error."""
        rng = np.random.default_rng(42)
        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = [0.5] * 5  # Wrong size

        with pytest.raises(ValueError, match="does not match fitness scores"):
            tournament_selection(pop, fitness, tournament_size=3, rng=rng)

    def test_tournament_invalid_size(self):
        """Test that invalid tournament size raises error."""
        rng = np.random.default_rng(42)
        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = [0.5] * 10

        with pytest.raises(ValueError, match="must be at least 1"):
            tournament_selection(pop, fitness, tournament_size=0, rng=rng)

        with pytest.raises(ValueError, match="cannot exceed population size"):
            tournament_selection(pop, fitness, tournament_size=11, rng=rng)


class TestRouletteSelection:
    """Test roulette wheel selection functionality."""

    def test_roulette_basic(self):
        """Test basic roulette selection."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = [float(i+1) for i in range(10)]  # 1 to 10

        parent = roulette_selection(pop, fitness, rng=rng)

        assert isinstance(parent, Chromosome)
        assert parent in pop

    def test_roulette_proportional_to_fitness(self):
        """Test that selection is proportional to fitness."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        # High contrast fitness: 1, 1, 1, ..., 1, 10
        fitness = [1.0] * 9 + [10.0]

        # Run many selections
        selections = []
        for _ in range(1000):
            parent = roulette_selection(pop, fitness, rng=rng)
            selections.append(pop.index(parent))

        # Last individual (fitness 10) should be selected much more than others
        count_last = sum(1 for idx in selections if idx == 9)
        count_others = len(selections) - count_last

        # Expect roughly 10:9 ratio (10/(10+9*1) = 10/19 ≈ 52%)
        ratio = count_last / len(selections)
        assert 0.45 <= ratio <= 0.60  # Allow some variance

    def test_roulette_handles_negative_fitness(self):
        """Test that roulette handles negative fitness by shifting."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

        # Should work without error (shifts to positive)
        parent = roulette_selection(pop, fitness, rng=rng)
        assert parent in pop

    def test_roulette_uniform_when_equal(self):
        """Test that roulette is uniform when all fitness equal."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = [0.5] * 10  # All equal

        # Run selections
        selections = []
        for _ in range(500):
            parent = roulette_selection(pop, fitness, rng=rng)
            selections.append(pop.index(parent))

        # Should have roughly uniform distribution
        counts = Counter(selections)
        # Each should be selected ~50 times (allow 30-70 range)
        for count in counts.values():
            assert 30 <= count <= 70

    def test_roulette_empty_population(self):
        """Test that empty population raises error."""
        with pytest.raises(ValueError, match="Population is empty"):
            roulette_selection([], [])


class TestSelectParents:
    """Test select_parents convenience function."""

    def test_select_parents_tournament(self):
        """Test selecting multiple parents with tournament."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(50)]
        fitness = list(range(50))

        parents = select_parents(pop, fitness, n_parents=20, method="tournament", tournament_size=3, rng=rng)

        assert len(parents) == 20
        assert all(isinstance(p, Chromosome) for p in parents)
        assert all(p in pop for p in parents)

    def test_select_parents_roulette(self):
        """Test selecting multiple parents with roulette."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(50)]
        fitness = list(range(1, 51))  # 1 to 50 (avoid zero)

        parents = select_parents(pop, fitness, n_parents=20, method="roulette", rng=rng)

        assert len(parents) == 20
        assert all(isinstance(p, Chromosome) for p in parents)

    def test_select_parents_invalid_method(self):
        """Test that invalid method raises error."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = list(range(10))

        with pytest.raises(ValueError, match="Unknown selection method"):
            select_parents(pop, fitness, n_parents=5, method="invalid", rng=rng)

    def test_select_parents_negative_count(self):
        """Test that negative n_parents raises error."""
        rng = np.random.default_rng(42)

        pop = [Chromosome.random(tree_depth=3, n_features=5, rng=rng) for _ in range(10)]
        fitness = list(range(10))

        with pytest.raises(ValueError, match="must be non-negative"):
            select_parents(pop, fitness, n_parents=-5, method="tournament", rng=rng)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
