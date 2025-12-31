#!/usr/bin/env python3
"""
Integration tests for complete GA pipeline.

Tests end-to-end GA optimization with full smoke run to ensure all components
work together correctly.
"""

import pytest
import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.ga_config import GAConfig
from optimization.ga_algorithm import GAAlgorithm


@pytest.mark.slow
@pytest.mark.integration
class TestGAIntegration:
    """Integration tests for complete GA pipeline."""

    def test_smoke_run_completes(self):
        """Test that a complete smoke run finishes without errors.

        This is the critical end-to-end test that validates the entire
        GA pipeline from initialization through evolution to final results.
        """
        # Load smoke test configuration
        config = GAConfig.from_yaml('config/ga/smoke_test.yaml')

        # Override to be even faster for testing
        config.population_size = 3
        config.max_generations = 2
        config.episodes_per_chromosome = 5
        config.parallel_workers = 2
        config.checkpointing_enabled = False  # Disable for clean test

        # Run optimization
        ga = GAAlgorithm(config=config, verbose=False)
        best_chromosome, best_fitness = ga.run()

        # Verify results
        assert best_chromosome is not None, "Best chromosome should not be None"
        assert isinstance(best_fitness, float), "Best fitness should be a float"
        assert 0.0 <= best_fitness <= 1.0, f"Best fitness {best_fitness} out of valid range [0,1]"

        # Verify that best was tracked
        assert ga.best_chromosome is not None
        assert ga.best_fitness == best_fitness
        assert ga.best_generation >= 0

    def test_smoke_run_produces_results(self):
        """Test that smoke run produces expected output artifacts."""
        # Load configuration
        config = GAConfig.from_yaml('config/ga/smoke_test.yaml')

        # Override for testing
        config.population_size = 3
        config.max_generations = 2
        config.episodes_per_chromosome = 5
        config.parallel_workers = 2

        # Clean results directory if exists
        results_dir = Path(config.results_dir)
        if results_dir.exists():
            shutil.rmtree(results_dir)

        # Run optimization
        ga = GAAlgorithm(config=config, verbose=False)
        best_chromosome, best_fitness = ga.run()

        # Check that results directory was created
        assert results_dir.exists(), f"Results directory {results_dir} was not created"

        # Check subdirectories
        assert (results_dir / "chromosomes").exists()
        assert (results_dir / "plots").exists()
        assert (results_dir / "data").exists()

        # Check key output files
        assert (results_dir / "chromosomes" / "best_chromosome.json").exists()
        assert (results_dir / "chromosomes" / "best_chromosome.pkl").exists()
        assert (results_dir / "chromosomes" / "final_population.pkl").exists()
        assert (results_dir / "data" / "fitness_history.csv").exists()
        assert (results_dir / "summary_report.json").exists()

        # Check plots
        assert (results_dir / "plots" / "convergence_plot.png").exists()
        assert (results_dir / "plots" / "diversity_plot.png").exists()
        assert (results_dir / "plots" / "combined_plot.png").exists()

    def test_convergence_tracking_works(self):
        """Test that convergence tracking functions correctly."""
        config = GAConfig.from_yaml('config/ga/smoke_test.yaml')

        config.population_size = 3
        config.max_generations = 5
        config.episodes_per_chromosome = 5
        config.early_stopping_patience = 2
        config.checkpointing_enabled = False

        ga = GAAlgorithm(config=config, verbose=False)
        best_chromosome, best_fitness = ga.run()

        # Check convergence tracker has history
        history = ga.convergence_tracker.get_history()
        assert len(history) > 0, "Convergence history should not be empty"

        # Each entry should be a tuple of (gen, best, mean, diversity)
        for entry in history:
            assert len(entry) == 4
            gen, best, mean, div = entry
            assert isinstance(gen, int)
            assert isinstance(best, (int, float))
            assert isinstance(mean, (int, float))
            assert isinstance(div, (int, float))

    @pytest.mark.slow
    def test_checkpointing_integration(self):
        """Test that checkpointing and resume work together."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GAConfig.from_yaml('config/ga/smoke_test.yaml')

            # Configure for checkpoint testing
            config.population_size = 3
            config.max_generations = 4
            config.episodes_per_chromosome = 5
            config.checkpointing_enabled = True
            config.checkpoint_save_dir = tmpdir
            config.checkpoint_frequency = 2  # Save every 2 generations

            # Run first GA (will create checkpoints)
            ga1 = GAAlgorithm(config=config, verbose=False)
            best1, fitness1 = ga1.run()

            # Verify checkpoints were created
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.pkl"))
            assert len(checkpoints) > 0, "No checkpoints were created"

            # Get latest checkpoint
            latest = ga1.checkpoint_manager.get_latest_checkpoint()
            assert latest is not None, "No latest checkpoint found"

            # Create new GA and resume from checkpoint
            ga2 = GAAlgorithm(config=config, verbose=False)
            ga2.resume_from_checkpoint(str(latest))

            # Verify state was restored
            assert ga2.best_fitness > -np.inf  # Should have restored best
            assert ga2.current_generation > 0  # Should have restored generation

    def test_population_evolution_preserves_diversity(self):
        """Test that population maintains some diversity during evolution."""
        config = GAConfig.from_yaml('config/ga/smoke_test.yaml')

        config.population_size = 5
        config.max_generations = 3
        config.episodes_per_chromosome = 5
        config.checkpointing_enabled = False

        ga = GAAlgorithm(config=config, verbose=False)
        best_chromosome, best_fitness = ga.run()

        # Check diversity was tracked
        history = ga.convergence_tracker.get_history()
        diversities = [entry[3] for entry in history]

        # Diversity should be in [0, 1]
        assert all(0.0 <= d <= 1.0 for d in diversities)

        # At least some diversity should remain (not complete convergence)
        # (This may fail occasionally with very small populations/generations)
        assert diversities[-1] > 0.0, "Population completely converged (zero diversity)"


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v', '-m', 'integration'])
