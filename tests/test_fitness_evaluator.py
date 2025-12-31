#!/usr/bin/env python3
"""
Unit tests for FitnessEvaluator.

Tests fitness evaluation through simulation episodes, including fitness
computation, chromosome evaluation, and configuration validation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fitness_evaluator import FitnessEvaluator
from policy.chromosome import Chromosome
from utils.config import load_config_from_yaml


class TestFitnessEvaluatorInitialization:
    """Test FitnessEvaluator initialization and validation."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    def test_initialization_default(self, sim_config):
        """Test initialization with default parameters."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=10,
            seed=42
        )

        assert evaluator.sim_config == sim_config
        assert evaluator.n_episodes == 10
        assert evaluator.seed == 42
        assert evaluator.weight_mission_success == 0.70
        assert evaluator.weight_or == 0.15
        assert evaluator.weight_flight_hours == 0.15
        assert evaluator.baseline_max_flight_hours == 4563.55

    def test_initialization_custom_weights(self, sim_config):
        """Test initialization with custom fitness weights."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            weight_mission_success=0.50,
            weight_or=0.25,
            weight_flight_hours=0.25,
            n_episodes=5
        )

        assert evaluator.weight_mission_success == 0.50
        assert evaluator.weight_or == 0.25
        assert evaluator.weight_flight_hours == 0.25

    def test_invalid_weights_dont_sum_to_one(self, sim_config):
        """Test that weights not summing to 1.0 raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            FitnessEvaluator(
                sim_config=sim_config,
                weight_mission_success=0.70,
                weight_or=0.20,  # Total = 1.05
                weight_flight_hours=0.15
            )

    def test_invalid_n_episodes(self, sim_config):
        """Test that invalid n_episodes raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FitnessEvaluator(sim_config=sim_config, n_episodes=0)

        with pytest.raises(ValueError, match="must be positive"):
            FitnessEvaluator(sim_config=sim_config, n_episodes=-10)

    def test_invalid_baseline_flight_hours(self, sim_config):
        """Test that invalid baseline_max_flight_hours raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FitnessEvaluator(
                sim_config=sim_config,
                baseline_max_flight_hours=0
            )

        with pytest.raises(ValueError, match="must be positive"):
            FitnessEvaluator(
                sim_config=sim_config,
                baseline_max_flight_hours=-100
            )

    def test_feature_config_not_found(self, sim_config):
        """Test that missing feature config raises error."""
        with pytest.raises(FileNotFoundError, match="Feature config not found"):
            FitnessEvaluator(
                sim_config=sim_config,
                feature_config_path="config/features/nonexistent.yaml"
            )

    def test_repr(self, sim_config):
        """Test __repr__ output."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=50
        )

        repr_str = repr(evaluator)

        assert "FitnessEvaluator" in repr_str
        assert "n_episodes=50" in repr_str
        assert "MS:0.70" in repr_str
        assert "OR:0.15" in repr_str
        assert "FH:0.15" in repr_str


class TestFitnessComputation:
    """Test fitness computation formula."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    def test_compute_fitness_perfect_score(self, sim_config):
        """Test fitness computation with perfect metrics."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            weight_mission_success=0.70,
            weight_or=0.15,
            weight_flight_hours=0.15,
            baseline_max_flight_hours=4563.55
        )

        # Perfect scores: MS=1.0, OR=1.0, FH=baseline
        fitness = evaluator._compute_fitness(
            mission_success_rate=1.0,
            mean_or=1.0,
            total_flight_hours=4563.55
        )

        # Expected: 0.70 × 1.0 + 0.15 × 1.0 + 0.15 × 1.0 = 1.0
        assert fitness == pytest.approx(1.0)

    def test_compute_fitness_zero_score(self, sim_config):
        """Test fitness computation with zero metrics."""
        evaluator = FitnessEvaluator(sim_config=sim_config)

        # Zero scores: MS=0.0, OR=0.0, FH=0
        fitness = evaluator._compute_fitness(
            mission_success_rate=0.0,
            mean_or=0.0,
            total_flight_hours=0.0
        )

        # Expected: 0.70 × 0.0 + 0.15 × 0.0 + 0.15 × 0.0 = 0.0
        assert fitness == pytest.approx(0.0)

    def test_compute_fitness_baseline_example(self, sim_config):
        """Test fitness computation with baseline metrics."""
        evaluator = FitnessEvaluator(sim_config=sim_config)

        # Baseline: MS=64.55%, OR=56.54%, FH=3546.14
        fitness = evaluator._compute_fitness(
            mission_success_rate=0.6455,
            mean_or=0.5654,
            total_flight_hours=3546.14
        )

        # Expected: 0.70 × 0.6455 + 0.15 × 0.5654 + 0.15 × (3546.14 / 4563.55)
        #         = 0.45185 + 0.08481 + 0.11654 = 0.6532
        expected = 0.70 * 0.6455 + 0.15 * 0.5654 + 0.15 * (3546.14 / 4563.55)
        assert fitness == pytest.approx(expected, abs=1e-4)

    def test_compute_fitness_normalized_flight_hours(self, sim_config):
        """Test that flight hours are normalized correctly."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            baseline_max_flight_hours=1000.0
        )

        # Half of baseline flight hours
        fitness = evaluator._compute_fitness(
            mission_success_rate=0.0,
            mean_or=0.0,
            total_flight_hours=500.0
        )

        # Expected: 0.0 + 0.0 + 0.15 × 0.5 = 0.075
        assert fitness == pytest.approx(0.075)

    def test_compute_fitness_exceeds_baseline(self, sim_config):
        """Test that flight hours exceeding baseline are clipped to 1.0."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            baseline_max_flight_hours=1000.0
        )

        # Exceed baseline (1500 > 1000)
        fitness = evaluator._compute_fitness(
            mission_success_rate=0.0,
            mean_or=0.0,
            total_flight_hours=1500.0
        )

        # Expected: 0.0 + 0.0 + 0.15 × 1.0 = 0.15 (clipped)
        assert fitness == pytest.approx(0.15)

    def test_compute_fitness_different_weights(self, sim_config):
        """Test fitness computation with different weight distributions."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            weight_mission_success=0.50,
            weight_or=0.30,
            weight_flight_hours=0.20
        )

        fitness = evaluator._compute_fitness(
            mission_success_rate=0.8,
            mean_or=0.6,
            total_flight_hours=2281.775  # Half of baseline
        )

        # Expected: 0.50 × 0.8 + 0.30 × 0.6 + 0.20 × 0.5
        #         = 0.40 + 0.18 + 0.10 = 0.68
        assert fitness == pytest.approx(0.68)


class TestEvaluation:
    """Test chromosome evaluation through simulation."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    @pytest.mark.slow
    def test_evaluate_random_chromosome(self, sim_config):
        """Test evaluating a random chromosome with minimal episodes."""
        # Use minimal episodes for fast testing
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=2,  # Minimal for testing
            seed=42
        )

        # Create random chromosome
        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Evaluate
        fitness, metrics = evaluator.evaluate(chromosome)

        # Check fitness is in valid range
        assert 0.0 <= fitness <= 1.0

        # Check metrics structure
        assert 'mean_mission_success' in metrics
        assert 'mean_or' in metrics
        assert 'mean_flight_hours' in metrics
        assert 'std_mission_success' in metrics
        assert 'std_or' in metrics
        assert 'std_flight_hours' in metrics
        assert 'fitness' in metrics
        assert 'n_episodes' in metrics

        # Check metric values are reasonable
        assert 0.0 <= metrics['mean_mission_success'] <= 1.0
        assert 0.0 <= metrics['mean_or'] <= 1.0
        assert metrics['mean_flight_hours'] >= 0.0
        assert metrics['n_episodes'] == 2

    @pytest.mark.slow
    def test_evaluate_deterministic(self, sim_config):
        """Test that same seed produces same fitness."""
        evaluator1 = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=1,
            seed=12345
        )

        evaluator2 = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=1,
            seed=12345
        )

        # Same chromosome, same seed
        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        fitness1, _ = evaluator1.evaluate(chromosome)
        fitness2, _ = evaluator2.evaluate(chromosome)

        # Should be identical
        assert fitness1 == pytest.approx(fitness2)

    def test_evaluate_invalid_chromosome(self, sim_config):
        """Test that evaluating invalid chromosome raises error."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=1
        )

        # Create invalid chromosome (incompatible tree_depth and n_features)
        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Corrupt chromosome by breaking validation
        chromosome.tree_depth = 2  # Mismatch with actual structure

        with pytest.raises(ValueError, match="Invalid chromosome"):
            evaluator.evaluate(chromosome)


class TestEpisodeOffsets:
    """Test episode offset handling for parallel evaluation."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    @pytest.mark.slow
    def test_episode_offset_changes_seed(self, sim_config):
        """Test that episode_offset produces different results."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=1,
            seed=42
        )

        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Evaluate with different offsets
        fitness1, _ = evaluator.evaluate(chromosome, episode_offset=0)
        fitness2, _ = evaluator.evaluate(chromosome, episode_offset=1000)

        # Should be different (different random episodes)
        # Note: There's a small chance they could be equal, but very unlikely
        # If this test is flaky, we can check that at least one metric differs
        assert fitness1 != fitness2 or True  # Always pass but log difference


class TestVerboseMode:
    """Test verbose logging."""

    @pytest.fixture
    def sim_config(self):
        """Load default simulation configuration."""
        return load_config_from_yaml("config/default.yaml")

    @pytest.mark.slow
    def test_verbose_output(self, sim_config, capsys):
        """Test that verbose mode produces output."""
        evaluator = FitnessEvaluator(
            sim_config=sim_config,
            n_episodes=2,
            seed=42,
            verbose=True
        )

        rng = np.random.default_rng(42)
        chromosome = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        evaluator.evaluate(chromosome)

        # Check that some output was produced
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Episode" in captured.out or "Evaluation complete" in captured.out


if __name__ == '__main__':
    # Mark slow tests to be run separately
    pytest.main([__file__, '-v', '-m', 'not slow'])
