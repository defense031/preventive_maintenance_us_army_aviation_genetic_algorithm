#!/usr/bin/env python3
"""
Unit tests for ParallelEvaluator.

Tests parallel chromosome evaluation using multiprocessing, including
population evaluation, progress tracking, and worker coordination.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.parallel_evaluator import ParallelEvaluator, _evaluate_chromosome_worker
from policy.chromosome import Chromosome
from utils.config import load_config_from_yaml


class TestParallelEvaluatorInitialization:
    """Test ParallelEvaluator initialization and validation."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    def test_initialization_default(self, sim_config):
        """Test initialization with default parameters."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=4,
            n_episodes=10,
            seed=42
        )

        assert evaluator.sim_config == sim_config
        assert evaluator.n_workers == 4
        assert evaluator.evaluator_config['n_episodes'] == 10
        assert evaluator.evaluator_config['seed'] == 42

    def test_initialization_custom_workers(self, sim_config):
        """Test initialization with custom worker count."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=8
        )

        assert evaluator.n_workers == 8

    def test_invalid_n_workers(self, sim_config):
        """Test that invalid n_workers raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ParallelEvaluator(sim_config=sim_config, n_workers=0)

        with pytest.raises(ValueError, match="must be positive"):
            ParallelEvaluator(sim_config=sim_config, n_workers=-2)

    def test_evaluator_config_stored(self, sim_config):
        """Test that evaluator configuration is properly stored."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            weight_mission_success=0.60,
            weight_or=0.25,
            weight_flight_hours=0.15,
            n_episodes=50,
            n_workers=4
        )

        assert evaluator.evaluator_config['weight_mission_success'] == 0.60
        assert evaluator.evaluator_config['weight_or'] == 0.25
        assert evaluator.evaluator_config['weight_flight_hours'] == 0.15
        assert evaluator.evaluator_config['n_episodes'] == 50

    def test_repr(self, sim_config):
        """Test __repr__ output."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=6,
            n_episodes=100
        )

        repr_str = repr(evaluator)

        assert "ParallelEvaluator" in repr_str
        assert "n_workers=6" in repr_str
        assert "n_episodes=100" in repr_str


class TestWorkerFunction:
    """Test the worker function for chromosome evaluation."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    @pytest.mark.slow
    def test_worker_function_basic(self, sim_config):
        """Test worker function evaluates chromosome correctly."""
        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        evaluator_kwargs = {
            'feature_config_path': 'config/features/simple_dt.yaml',
            'weight_mission_success': 0.70,
            'weight_or': 0.15,
            'weight_flight_hours': 0.15,
            'baseline_max_flight_hours': 4563.55,
            'n_episodes': 1,  # Minimal for testing
            'seed': 42,
            'verbose': False
        }

        args = (0, chromosome, sim_config, evaluator_kwargs)

        # Call worker function
        idx, fitness, metrics = _evaluate_chromosome_worker(args)

        # Check results
        assert idx == 0
        assert 0.0 <= fitness <= 1.0
        assert 'mean_mission_success' in metrics
        assert 'fitness' in metrics

    @pytest.mark.slow
    def test_worker_function_different_indices(self, sim_config):
        """Test that different indices produce different results due to episode offset."""
        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        evaluator_kwargs = {
            'feature_config_path': 'config/features/simple_dt.yaml',
            'weight_mission_success': 0.70,
            'weight_or': 0.15,
            'weight_flight_hours': 0.15,
            'baseline_max_flight_hours': 4563.55,
            'n_episodes': 1,
            'seed': 42,
            'verbose': False
        }

        # Evaluate with index 0
        args1 = (0, chromosome, sim_config, evaluator_kwargs)
        idx1, fitness1, _ = _evaluate_chromosome_worker(args1)

        # Evaluate with index 5
        args2 = (5, chromosome, sim_config, evaluator_kwargs)
        idx2, fitness2, _ = _evaluate_chromosome_worker(args2)

        assert idx1 == 0
        assert idx2 == 5
        # Fitness may differ due to different episode offsets
        # (but could be same by chance, so we don't assert inequality)


class TestSingleEvaluation:
    """Test single chromosome evaluation (non-parallel)."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    @pytest.mark.slow
    def test_evaluate_single_basic(self, sim_config):
        """Test evaluating single chromosome without multiprocessing."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=2,
            n_episodes=1,  # Minimal for testing
            seed=42
        )

        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        fitness, metrics = evaluator.evaluate_single(chromosome, chromosome_idx=0)

        # Check fitness in valid range
        assert 0.0 <= fitness <= 1.0

        # Check metrics structure
        assert 'mean_mission_success' in metrics
        assert 'mean_or' in metrics
        assert 'fitness' in metrics

    @pytest.mark.slow
    def test_evaluate_single_deterministic(self, sim_config):
        """Test that same parameters produce same results."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=2,
            n_episodes=1,
            seed=42
        )

        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Evaluate twice with same index
        fitness1, _ = evaluator.evaluate_single(chromosome, chromosome_idx=0)
        fitness2, _ = evaluator.evaluate_single(chromosome, chromosome_idx=0)

        # Should be identical
        assert fitness1 == pytest.approx(fitness2)


class TestPopulationEvaluation:
    """Test parallel population evaluation."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    def test_evaluate_empty_population(self, sim_config):
        """Test evaluating empty population returns empty results."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=2,
            n_episodes=1
        )

        fitness_scores, metrics = evaluator.evaluate_population([])

        assert fitness_scores == []
        assert metrics == []

    @pytest.mark.slow
    def test_evaluate_population_basic(self, sim_config):
        """Test evaluating small population in parallel."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=2,
            n_episodes=1,  # Minimal for testing
            seed=42
        )

        # Create small population
        rng = np.random.default_rng(42)
        population = [
            Chromosome.random(tree_depth=3, n_features=5, rng=rng)
            for _ in range(4)
        ]

        # Evaluate population
        fitness_scores, all_metrics = evaluator.evaluate_population(population)

        # Check results
        assert len(fitness_scores) == 4
        assert len(all_metrics) == 4

        # Check all fitness scores are valid
        for fitness in fitness_scores:
            assert 0.0 <= fitness <= 1.0

        # Check all metrics are present
        for metrics in all_metrics:
            assert 'mean_mission_success' in metrics
            assert 'mean_or' in metrics
            assert 'fitness' in metrics

    @pytest.mark.slow
    def test_evaluate_population_progress_callback(self, sim_config):
        """Test progress callback is called correctly."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=2,
            n_episodes=1,
            seed=42
        )

        # Create small population
        rng = np.random.default_rng(42)
        population = [
            Chromosome.random(tree_depth=3, n_features=5, rng=rng)
            for _ in range(3)
        ]

        # Track progress
        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        # Evaluate with callback
        evaluator.evaluate_population(population, progress_callback=progress_callback)

        # Check callback was called for each chromosome
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)  # Final call

        # Check all calls had correct total
        for _, total in progress_calls:
            assert total == 3

    @pytest.mark.slow
    def test_evaluate_population_order_preserved(self, sim_config):
        """Test that results are returned in correct order despite parallel execution."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=2,
            n_episodes=1,
            seed=42
        )

        # Create population with distinct chromosomes
        rng = np.random.default_rng(42)
        population = []
        for i in range(4):
            chrom = Chromosome.random(tree_depth=3, n_features=5, rng=rng)
            population.append(chrom)

        # Evaluate population
        fitness_scores, _ = evaluator.evaluate_population(population)

        # Evaluate each individually for comparison
        individual_scores = []
        for idx, chrom in enumerate(population):
            fitness, _ = evaluator.evaluate_single(chrom, chromosome_idx=idx)
            individual_scores.append(fitness)

        # Check order is preserved (parallel results match sequential)
        for parallel, individual in zip(fitness_scores, individual_scores):
            assert parallel == pytest.approx(individual)


class TestVerboseMode:
    """Test verbose logging in parallel evaluation."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    @pytest.mark.slow
    def test_verbose_output(self, sim_config, capsys):
        """Test that verbose mode produces output."""
        evaluator = ParallelEvaluator(
            sim_config=sim_config,
            n_workers=2,
            n_episodes=1,
            seed=42,
            verbose=True
        )

        # Create small population
        rng = np.random.default_rng(42)
        population = [
            Chromosome.random(tree_depth=3, n_features=5, rng=rng)
            for _ in range(2)
        ]

        evaluator.evaluate_population(population)

        # Check that output was produced
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Starting parallel evaluation" in captured.out or "Population size" in captured.out


if __name__ == '__main__':
    # Mark slow tests to be run separately
    pytest.main([__file__, '-v', '-m', 'not slow'])
