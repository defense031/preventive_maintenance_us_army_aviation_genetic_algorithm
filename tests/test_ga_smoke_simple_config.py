#!/usr/bin/env python3
"""
Smoke Test for GA Operations with Simple Config (15 genes).

Tests end-to-end GA pipeline without full simulation:
1. Initialize population with simple config chromosomes
2. Assign mock fitness scores
3. Perform selection, crossover, mutation
4. Complete multiple generations without errors

This verifies the GA operators work correctly with the
corrected 15-gene simple config chromosomes.
"""

import pytest
import numpy as np
import copy
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.chromosome import Chromosome
from optimization.population import Population
from optimization.selection import select_parents, tournament_selection


class TestGASmokeSimpleConfig:
    """Smoke test for GA operations with simple config."""

    def test_population_initialization(self):
        """Test population initializes correctly with simple config."""
        pop = Population(size=20, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        assert len(pop.chromosomes) == 20
        assert len(pop.fitness_scores) == 20

        # All chromosomes should be simple config with 15 genes
        for chrom in pop.chromosomes:
            assert chrom.config_type == 'simple'
            assert chrom.n_genes == 15
            assert len(chrom.feature_indices) == 7
            assert len(chrom.thresholds) == 7

    def test_fitness_update(self):
        """Test fitness score updates work correctly."""
        pop = Population(size=10, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Update all fitness scores
        for i in range(10):
            pop.update_fitness(i, float(i) / 10.0)

        # Verify
        assert pop.fitness_scores[0] == 0.0
        assert pop.fitness_scores[9] == 0.9

    def test_elitism_selection(self):
        """Test elite selection returns top k individuals."""
        pop = Population(size=10, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign fitness scores
        for i in range(10):
            pop.update_fitness(i, float(i))  # 0, 1, 2, ..., 9

        # Get top 3
        elite = pop.get_top_k(k=3)

        assert len(elite) == 3
        # Elite should be indices 9, 8, 7 (highest fitness)
        assert elite[0][1] == 9.0
        assert elite[1][1] == 8.0
        assert elite[2][1] == 7.0

    def test_tournament_selection(self):
        """Test tournament selection works with simple config."""
        pop = Population(size=20, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign varied fitness scores
        for i in range(20):
            pop.update_fitness(i, np.random.random())

        rng = np.random.default_rng(123)

        # Select multiple parents
        for _ in range(10):
            parent = tournament_selection(
                population=pop.chromosomes,
                fitness_scores=pop.fitness_scores,
                tournament_size=3,
                rng=rng
            )
            assert isinstance(parent, Chromosome)
            assert parent.config_type == 'simple'
            assert parent.n_genes == 15

    def test_crossover_produces_valid_offspring(self):
        """Test crossover produces valid simple config offspring."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)

        child1, child2 = parent1.crossover(parent2, crossover_rate=0.7, rng=rng)

        # Both children should be valid simple config
        assert child1.config_type == 'simple'
        assert child2.config_type == 'simple'
        assert child1.n_genes == 15
        assert child2.n_genes == 15

        child1.validate()
        child2.validate()

    def test_mutation_produces_valid_offspring(self):
        """Test mutation produces valid simple config offspring."""
        rng = np.random.default_rng(42)

        parent = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)
        mutant = parent.mutate(mutation_rate=0.5, mutation_sigma=0.2, rng=rng)

        assert mutant.config_type == 'simple'
        assert mutant.n_genes == 15
        mutant.validate()

    def test_complete_generation_cycle(self):
        """Test complete generation cycle: selection → crossover → mutation."""
        rng = np.random.default_rng(42)

        pop = Population(size=20, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign random fitness scores
        for i in range(20):
            pop.update_fitness(i, rng.random())

        # Elite selection (keep top 2)
        elite_count = 2
        elite_pairs = pop.get_top_k(k=elite_count)
        elite_chromosomes = [chrom for chrom, fitness in elite_pairs]

        # Create offspring
        offspring = []
        n_offspring_needed = 20 - elite_count

        while len(offspring) < n_offspring_needed:
            # Select parents
            parents = select_parents(
                population=pop.chromosomes,
                fitness_scores=pop.fitness_scores,
                n_parents=2,
                method='tournament',
                tournament_size=3,
                rng=rng
            )
            parent1, parent2 = parents[0], parents[1]

            # Crossover
            child1, child2 = parent1.crossover(parent2, crossover_rate=0.7, rng=rng)

            # Mutation
            child1 = child1.mutate(mutation_rate=0.1, mutation_sigma=0.1, rng=rng)
            child2 = child2.mutate(mutation_rate=0.1, mutation_sigma=0.1, rng=rng)

            offspring.append(child1)
            if len(offspring) < n_offspring_needed:
                offspring.append(child2)

        # Combine elite + offspring
        new_population = elite_chromosomes + offspring[:n_offspring_needed]

        # Verify all chromosomes are valid
        assert len(new_population) == 20
        for chrom in new_population:
            assert chrom.config_type == 'simple'
            assert chrom.n_genes == 15
            chrom.validate()

        # Replace population
        pop.replace(new_population)
        assert pop.generation == 1

    def test_multi_generation_evolution(self):
        """Test multiple generations of evolution without errors."""
        rng = np.random.default_rng(42)
        pop = Population(size=20, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        n_generations = 5
        elite_count = 2
        crossover_rate = 0.7
        mutation_rate = 0.1
        mutation_sigma = 0.1
        tournament_size = 3

        best_fitness_history = []

        for gen in range(n_generations):
            # Simulate fitness evaluation (mock scores)
            for i in range(pop.size):
                # Use chromosome-based mock fitness (sum of feature indices)
                chrom = pop.chromosomes[i]
                mock_fitness = sum(chrom.feature_indices) / (7 * 4)  # Normalize to ~[0, 1]
                pop.update_fitness(i, mock_fitness)

            # Track best
            best_chrom, best_fitness = pop.get_best()
            best_fitness_history.append(best_fitness)

            # Evolve next generation
            elite_pairs = pop.get_top_k(k=elite_count)
            elite_chromosomes = [chrom for chrom, fitness in elite_pairs]

            offspring = []
            n_offspring_needed = pop.size - elite_count

            while len(offspring) < n_offspring_needed:
                parents = select_parents(
                    population=pop.chromosomes,
                    fitness_scores=pop.fitness_scores,
                    n_parents=2,
                    method='tournament',
                    tournament_size=tournament_size,
                    rng=rng
                )
                parent1, parent2 = parents[0], parents[1]

                child1, child2 = parent1.crossover(parent2, crossover_rate=crossover_rate, rng=rng)
                child1 = child1.mutate(mutation_rate=mutation_rate, mutation_sigma=mutation_sigma, rng=rng)
                child2 = child2.mutate(mutation_rate=mutation_rate, mutation_sigma=mutation_sigma, rng=rng)

                offspring.append(child1)
                if len(offspring) < n_offspring_needed:
                    offspring.append(child2)

            new_population = elite_chromosomes + offspring[:n_offspring_needed]
            pop.replace(new_population)

        # Verify generations completed
        assert pop.generation == n_generations
        assert len(best_fitness_history) == n_generations

        # All chromosomes should still be valid
        for chrom in pop.chromosomes:
            assert chrom.config_type == 'simple'
            assert chrom.n_genes == 15
            chrom.validate()


class TestPopulationStatistics:
    """Test population statistics with simple config."""

    def test_diversity_computation(self):
        """Test diversity computation works with simple config."""
        pop = Population(size=10, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        diversity = pop.compute_diversity()

        # Diversity should be between 0 and 1
        assert 0.0 <= diversity <= 1.0

    def test_statistics_computation(self):
        """Test statistics computation works correctly."""
        pop = Population(size=10, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Assign varied fitness
        for i in range(10):
            pop.update_fitness(i, float(i))

        stats = pop.get_statistics()

        assert stats['best'] == 9.0
        assert stats['worst'] == 0.0
        assert 4.0 <= stats['mean'] <= 5.0  # Should be 4.5
        assert 'std' in stats
        assert 'median' in stats


class TestDeterminism:
    """Test deterministic behavior with fixed seeds."""

    def test_population_determinism(self):
        """Same seed should produce identical populations."""
        pop1 = Population(size=10, tree_depth=3, n_features=5, seed=12345)
        pop1.initialize_random()

        pop2 = Population(size=10, tree_depth=3, n_features=5, seed=12345)
        pop2.initialize_random()

        for c1, c2 in zip(pop1.chromosomes, pop2.chromosomes):
            assert c1.feature_indices == c2.feature_indices
            assert c1.thresholds == c2.thresholds
            assert c1.tiebreak_feature == c2.tiebreak_feature

    def test_evolution_determinism(self):
        """Same seed should produce identical evolution."""
        def evolve_one_generation(seed):
            rng = np.random.default_rng(seed)
            pop = Population(size=10, tree_depth=3, n_features=5, seed=seed)
            pop.initialize_random()

            # Assign fitness
            for i in range(10):
                pop.update_fitness(i, rng.random())

            # Evolve
            elite = pop.get_top_k(k=2)
            elite_chroms = [c for c, f in elite]

            offspring = []
            while len(offspring) < 8:
                parents = select_parents(
                    pop.chromosomes, pop.fitness_scores, 2, 'tournament', 3, rng
                )
                c1, c2 = parents[0].crossover(parents[1], crossover_rate=0.7, rng=rng)
                c1 = c1.mutate(mutation_rate=0.1, rng=rng)
                c2 = c2.mutate(mutation_rate=0.1, rng=rng)
                offspring.extend([c1, c2])

            return elite_chroms + offspring[:8]

        result1 = evolve_one_generation(99999)
        result2 = evolve_one_generation(99999)

        for c1, c2 in zip(result1, result2):
            assert c1.feature_indices == c2.feature_indices
            assert c1.thresholds == c2.thresholds


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
