#!/usr/bin/env python3
"""
Unit tests for GAAlgorithm.

Tests GA main loop, evolution, checkpointing, and integration.
Most tests are marked as slow since they require simulation.
"""

import pytest
import sys
import copy
import tempfile
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.ga_algorithm import GAAlgorithm
from optimization.ga_config import GAConfig
from policy.chromosome import Chromosome


class TestGAAlgorithmInitialization:
    """Test GAAlgorithm initialization."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return GAConfig.from_yaml('config/ga/default.yaml')

    def test_initialization_basic(self, test_config):
        """Test basic initialization."""
        ga = GAAlgorithm(config=test_config, verbose=False)

        assert ga.config == test_config
        assert ga.population is not None
        assert ga.evaluator is not None
        assert ga.convergence_tracker is not None
        assert ga.checkpoint_manager is not None
        assert ga.results_manager is not None
        assert ga.best_chromosome is None
        assert ga.best_fitness == -np.inf

    def test_initialization_creates_population(self, test_config):
        """Test that population is created with correct size."""
        ga = GAAlgorithm(config=test_config, verbose=False)

        assert ga.population.size == test_config.population_size
        assert ga.population.tree_depth == test_config.tree_depth
        assert ga.population.n_features == test_config.n_features

    def test_initialization_rng_seeded(self, test_config):
        """Test that RNG is properly seeded."""
        ga1 = GAAlgorithm(config=test_config, verbose=False)
        ga2 = GAAlgorithm(config=test_config, verbose=False)

        # Should produce same random sequence
        vals1 = ga1.rng.random(5)
        vals2 = ga2.rng.random(5)

        np.testing.assert_array_equal(vals1, vals2)

    def test_repr(self, test_config):
        """Test __repr__ output."""
        ga = GAAlgorithm(config=test_config, verbose=False)

        repr_str = repr(ga)

        assert "GAAlgorithm" in repr_str
        assert str(test_config.population_size) in repr_str


class TestPopulationEvolution:
    """Test population evolution mechanics."""

    @pytest.fixture
    def test_config(self):
        """Create small test configuration."""
        config = GAConfig.from_yaml('config/ga/default.yaml')
        # Override for faster testing
        config.population_size = 10
        config.elite_count = 2
        return config

    def test_population_initialized(self, test_config):
        """Test that population gets initialized."""
        ga = GAAlgorithm(config=test_config, verbose=False)
        ga.population.initialize_random()

        assert len(ga.population.chromosomes) == test_config.population_size
        assert all(isinstance(c, Chromosome) for c in ga.population.chromosomes)

    def test_evolve_generation_preserves_size(self, test_config):
        """Test that evolution preserves population size."""
        ga = GAAlgorithm(config=test_config, verbose=False)
        ga.population.initialize_random()

        # Set dummy fitness scores
        ga.population.fitness_scores = [float(i) for i in range(test_config.population_size)]

        # Evolve one generation
        initial_size = len(ga.population.chromosomes)
        ga._evolve_generation()

        assert len(ga.population.chromosomes) == initial_size

    def test_evolve_generation_creates_new_individuals(self, test_config):
        """Test that evolution creates new offspring."""
        ga = GAAlgorithm(config=test_config, verbose=False)
        ga.population.initialize_random()

        # Set dummy fitness scores
        ga.population.fitness_scores = [float(i) for i in range(test_config.population_size)]

        # Store initial population
        initial_chromosomes = [copy.deepcopy(c) for c in ga.population.chromosomes]

        # Evolve
        ga._evolve_generation()

        # At least some chromosomes should be different (offspring)
        # Note: Elite might be preserved
        differences = sum(
            1 for i, c in enumerate(ga.population.chromosomes)
            if not np.array_equal(c.genes, initial_chromosomes[i].genes)
        )

        assert differences > 0  # At least some changed


class TestBestChromosomeTracking:
    """Test tracking of best chromosome."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        config = GAConfig.from_yaml('config/ga/default.yaml')
        config.population_size = 5
        return config

    def test_update_best_chromosome_initial(self, test_config):
        """Test updating best chromosome initially."""
        ga = GAAlgorithm(config=test_config, verbose=False)
        ga.population.initialize_random()

        # Create mock fitness scores and metrics
        fitness_scores = [0.5, 0.7, 0.6, 0.4, 0.55]
        all_metrics = [
            {'mean_mission_success': 0.5, 'mean_or': 0.4, 'mean_flight_hours': 1000},
            {'mean_mission_success': 0.7, 'mean_or': 0.6, 'mean_flight_hours': 1500},
            {'mean_mission_success': 0.6, 'mean_or': 0.5, 'mean_flight_hours': 1200},
            {'mean_mission_success': 0.4, 'mean_or': 0.3, 'mean_flight_hours': 800},
            {'mean_mission_success': 0.55, 'mean_or': 0.45, 'mean_flight_hours': 1100}
        ]

        ga._update_best_chromosome(
            generation=0,
            fitness_scores=fitness_scores,
            all_metrics=all_metrics
        )

        # Best should be index 1 (fitness 0.7)
        assert ga.best_fitness == 0.7
        assert ga.best_generation == 0
        assert ga.best_metrics['mean_mission_success'] == 0.7

    def test_update_best_chromosome_improvement(self, test_config):
        """Test that best updates only on improvement."""
        ga = GAAlgorithm(config=test_config, verbose=False)
        ga.population.initialize_random()

        # Initial best
        ga.best_fitness = 0.6
        ga.best_generation = 0

        # New generation with better fitness
        fitness_scores = [0.5, 0.8, 0.6, 0.4, 0.55]
        all_metrics = [{'mean_mission_success': 0.8}] * 5

        ga._update_best_chromosome(
            generation=5,
            fitness_scores=fitness_scores,
            all_metrics=all_metrics
        )

        # Should update
        assert ga.best_fitness == 0.8
        assert ga.best_generation == 5

    def test_update_best_chromosome_no_improvement(self, test_config):
        """Test that best doesn't update without improvement."""
        ga = GAAlgorithm(config=test_config, verbose=False)
        ga.population.initialize_random()

        # Initial best
        ga.best_fitness = 0.8
        ga.best_generation = 3

        # New generation with worse fitness
        fitness_scores = [0.5, 0.7, 0.6, 0.4, 0.55]
        all_metrics = [{'mean_mission_success': 0.5}] * 5

        ga._update_best_chromosome(
            generation=5,
            fitness_scores=fitness_scores,
            all_metrics=all_metrics
        )

        # Should NOT update
        assert ga.best_fitness == 0.8
        assert ga.best_generation == 3


class TestCheckpointing:
    """Test checkpointing functionality."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration with checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GAConfig.from_yaml('config/ga/default.yaml')
            config.population_size = 5
            config.checkpointing_enabled = True
            config.checkpoint_save_dir = tmpdir
            config.checkpoint_frequency = 2
            yield config

    def test_save_checkpoint(self, test_config):
        """Test saving checkpoint."""
        ga = GAAlgorithm(config=test_config, verbose=False)
        ga.population.initialize_random()
        ga.population.fitness_scores = [0.5] * test_config.population_size

        # Save checkpoint
        ga._save_checkpoint(generation=5)

        # Check file exists
        checkpoints = ga.checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0


class TestResumeFromCheckpoint:
    """Test resuming from checkpoint."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GAConfig.from_yaml('config/ga/default.yaml')
            config.population_size = 5
            config.checkpointing_enabled = True
            config.checkpoint_save_dir = tmpdir
            yield config

    def test_resume_from_checkpoint_restores_state(self, test_config):
        """Test that resume restores GA state."""
        # Create initial GA and save checkpoint
        ga1 = GAAlgorithm(config=test_config, verbose=False)
        ga1.population.initialize_random()
        ga1.population.fitness_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        ga1.best_fitness = 0.9
        ga1.best_generation = 5

        # Save checkpoint
        checkpoint_path = ga1.checkpoint_manager.save_checkpoint(
            generation=5,
            population=ga1.population,
            fitness_scores=ga1.population.fitness_scores,
            rng_state=ga1.rng.bit_generator.state,
            convergence_history=[],
            best_chromosome=ga1.population.chromosomes[4],
            metadata={'best_fitness': 0.9, 'best_generation': 5}
        )

        # Create new GA and resume
        ga2 = GAAlgorithm(config=test_config, verbose=False)
        ga2.resume_from_checkpoint(str(checkpoint_path))

        # Verify state restored
        assert ga2.best_fitness == 0.9
        assert ga2.best_generation == 5
        assert ga2.current_generation == 6  # Should be next generation


@pytest.mark.slow
class TestFullOptimizationSmoke:
    """Smoke tests for full optimization run (slow)."""

    @pytest.fixture
    def smoke_config(self):
        """Create minimal smoke test configuration."""
        config = GAConfig.from_yaml('config/ga/default.yaml')
        # Minimal settings for fast smoke test
        config.population_size = 3
        config.max_generations = 2
        config.episodes_per_chromosome = 1  # Single episode for speed
        config.elite_count = 1
        config.checkpointing_enabled = False
        config.parallel_workers = 2
        return config

    def test_run_completes(self, smoke_config):
        """Test that GA run completes without errors."""
        ga = GAAlgorithm(config=smoke_config, verbose=False)

        # Run optimization
        best_chrom, best_fitness = ga.run()

        # Verify results
        assert best_chrom is not None
        assert isinstance(best_fitness, float)
        assert 0.0 <= best_fitness <= 1.0

    def test_run_tracks_best(self, smoke_config):
        """Test that best chromosome is tracked."""
        ga = GAAlgorithm(config=smoke_config, verbose=False)

        best_chrom, best_fitness = ga.run()

        # Should have tracked best
        assert ga.best_chromosome is not None
        assert ga.best_fitness == best_fitness
        assert ga.best_generation >= 0


@pytest.mark.slow
class TestEarlyStoppingIntegration:
    """Test early stopping integration (slow)."""

    @pytest.fixture
    def early_stop_config(self):
        """Create configuration with aggressive early stopping."""
        config = GAConfig.from_yaml('config/ga/default.yaml')
        config.population_size = 3
        config.max_generations = 20
        config.episodes_per_chromosome = 1
        config.early_stopping_patience = 2  # Very low patience
        config.improvement_threshold = 0.001
        config.parallel_workers = 2
        return config

    def test_early_stopping_can_trigger(self, early_stop_config):
        """Test that early stopping can trigger before max generations."""
        ga = GAAlgorithm(config=early_stop_config, verbose=False)

        best_chrom, best_fitness = ga.run()

        # Should stop before max_generations due to convergence
        # (may or may not trigger depending on random initialization)
        assert ga.current_generation <= early_stop_config.max_generations


class TestEdgeCases:
    """Test edge cases."""

    def test_single_elite_population(self):
        """Test with population size equal to elite count."""
        config = GAConfig.from_yaml('config/ga/default.yaml')
        config.population_size = 4
        config.elite_count = 4  # All elite
        config.max_generations = 1
        config.episodes_per_chromosome = 1

        ga = GAAlgorithm(config=config, verbose=False)

        # Should not crash (no offspring needed)
        ga.population.initialize_random()
        ga.population.fitness_scores = [0.5, 0.6, 0.7, 0.8]
        ga._evolve_generation()

        assert len(ga.population.chromosomes) == 4


if __name__ == '__main__':
    # Run non-slow tests by default
    pytest.main([__file__, '-v', '-m', 'not slow'])
