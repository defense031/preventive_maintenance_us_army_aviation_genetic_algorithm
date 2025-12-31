#!/usr/bin/env python3
"""
Unit tests for GAConfig dataclass.

Tests GA configuration loading from YAML, validation, and parameter access.
"""

import pytest
import sys
import tempfile
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.ga_config import GAConfig


class TestGAConfigLoading:
    """Test loading GA configuration from YAML."""

    def test_load_default_config(self):
        """Test loading default GA configuration."""
        config = GAConfig.from_yaml('config/ga/default.yaml')

        # Check metadata
        assert config.name == "ga_default"
        assert config.version == "1.0.0"

        # Check population
        assert config.population_size == 50
        assert config.elite_count == 4

        # Check operators
        assert config.crossover_rate == 0.70
        assert config.mutation_rate == 0.30
        assert config.blx_alpha == 0.5

        # Check selection
        assert config.selection_method == "tournament"
        assert config.tournament_size == 3

        # Check fitness
        assert config.fitness_weight_mission_success == 0.70
        assert config.fitness_weight_or == 0.15
        assert config.fitness_weight_flight_hours == 0.15
        assert config.baseline_max_flight_hours == 4563.55

        # Check evaluation
        assert config.episodes_per_chromosome == 100
        assert config.parallel_workers == 6

        # Check convergence
        assert config.max_generations == 50
        assert config.early_stopping_patience == 10
        assert config.improvement_threshold == 0.001

        # Check checkpointing
        assert config.checkpointing_enabled == True
        assert config.checkpoint_frequency == 10

        # Check chromosome
        assert config.tree_depth == 3
        assert config.n_features == 5
        assert config.n_splits == 7
        assert config.n_leaves == 8

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="GA config file not found"):
            GAConfig.from_yaml('config/ga/nonexistent.yaml')

    def test_load_empty_yaml(self):
        """Test loading empty YAML file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Empty or invalid YAML"):
                GAConfig.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_partial_config_uses_defaults(self):
        """Test that partial config uses default values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'name': 'test_config',
                'population': {'size': 30},
                'operators': {'crossover_rate': 0.80}
            }, f)
            temp_path = f.name

        try:
            config = GAConfig.from_yaml(temp_path)

            # Check custom values
            assert config.name == 'test_config'
            assert config.population_size == 30
            assert config.crossover_rate == 0.80

            # Check defaults are still applied
            assert config.elite_count == 4  # Default
            assert config.mutation_rate == 0.30  # Default
            assert config.tournament_size == 3  # Default
        finally:
            Path(temp_path).unlink()


class TestGAConfigValidation:
    """Test GAConfig validation."""

    def test_valid_config_passes(self):
        """Test that valid configuration passes validation."""
        config = GAConfig()
        # Should not raise
        config.validate()

    def test_invalid_population_size(self):
        """Test that invalid population_size fails validation."""
        config = GAConfig(population_size=0)
        with pytest.raises(ValueError, match="population_size must be positive"):
            config.validate()

        config = GAConfig(population_size=-10)
        with pytest.raises(ValueError, match="population_size must be positive"):
            config.validate()

    def test_invalid_elite_count(self):
        """Test that invalid elite_count fails validation."""
        config = GAConfig(elite_count=-1)
        with pytest.raises(ValueError, match="elite_count must be non-negative"):
            config.validate()

        config = GAConfig(elite_count=60)  # Exceeds population_size=50
        with pytest.raises(ValueError, match="cannot exceed population_size"):
            config.validate()

    def test_invalid_crossover_rate(self):
        """Test that invalid crossover_rate fails validation."""
        config = GAConfig(crossover_rate=-0.1)
        with pytest.raises(ValueError, match="crossover_rate must be in"):
            config.validate()

        config = GAConfig(crossover_rate=1.5)
        with pytest.raises(ValueError, match="crossover_rate must be in"):
            config.validate()

    def test_invalid_mutation_rate(self):
        """Test that invalid mutation_rate fails validation."""
        config = GAConfig(mutation_rate=-0.1)
        with pytest.raises(ValueError, match="mutation_rate must be in"):
            config.validate()

        config = GAConfig(mutation_rate=1.5)
        with pytest.raises(ValueError, match="mutation_rate must be in"):
            config.validate()

    def test_invalid_blx_alpha(self):
        """Test that negative blx_alpha fails validation."""
        config = GAConfig(blx_alpha=-0.5)
        with pytest.raises(ValueError, match="blx_alpha must be non-negative"):
            config.validate()

    def test_invalid_selection_method(self):
        """Test that invalid selection_method fails validation."""
        config = GAConfig(selection_method="invalid")
        with pytest.raises(ValueError, match="must be 'tournament' or 'roulette'"):
            config.validate()

    def test_invalid_tournament_size(self):
        """Test that invalid tournament_size fails validation."""
        config = GAConfig(tournament_size=0)
        with pytest.raises(ValueError, match="tournament_size must be at least 1"):
            config.validate()

        config = GAConfig(tournament_size=100)  # Exceeds population_size=50
        with pytest.raises(ValueError, match="cannot exceed population_size"):
            config.validate()

    def test_invalid_fitness_weights(self):
        """Test that fitness weights not summing to 1.0 fails validation."""
        config = GAConfig(
            fitness_weight_mission_success=0.70,
            fitness_weight_or=0.20,  # Total = 1.05
            fitness_weight_flight_hours=0.15
        )
        with pytest.raises(ValueError, match="Fitness weights must sum to 1.0"):
            config.validate()

    def test_invalid_baseline_max_flight_hours(self):
        """Test that invalid baseline_max_flight_hours fails validation."""
        config = GAConfig(baseline_max_flight_hours=0)
        with pytest.raises(ValueError, match="baseline_max_flight_hours must be positive"):
            config.validate()

        config = GAConfig(baseline_max_flight_hours=-100)
        with pytest.raises(ValueError, match="baseline_max_flight_hours must be positive"):
            config.validate()

    def test_invalid_episodes_per_chromosome(self):
        """Test that invalid episodes_per_chromosome fails validation."""
        config = GAConfig(episodes_per_chromosome=0)
        with pytest.raises(ValueError, match="episodes_per_chromosome must be positive"):
            config.validate()

    def test_invalid_parallel_workers(self):
        """Test that invalid parallel_workers fails validation."""
        config = GAConfig(parallel_workers=0)
        with pytest.raises(ValueError, match="parallel_workers must be positive"):
            config.validate()

    def test_invalid_max_generations(self):
        """Test that invalid max_generations fails validation."""
        config = GAConfig(max_generations=0)
        with pytest.raises(ValueError, match="max_generations must be positive"):
            config.validate()

    def test_invalid_early_stopping_patience(self):
        """Test that negative early_stopping_patience fails validation."""
        config = GAConfig(early_stopping_patience=-1)
        with pytest.raises(ValueError, match="early_stopping_patience must be non-negative"):
            config.validate()

    def test_invalid_improvement_threshold(self):
        """Test that negative improvement_threshold fails validation."""
        config = GAConfig(improvement_threshold=-0.001)
        with pytest.raises(ValueError, match="improvement_threshold must be non-negative"):
            config.validate()

    def test_invalid_checkpoint_frequency(self):
        """Test that invalid checkpoint_frequency fails validation."""
        config = GAConfig(checkpoint_frequency=0)
        with pytest.raises(ValueError, match="checkpoint_frequency must be positive"):
            config.validate()

    def test_invalid_log_frequency(self):
        """Test that invalid log_frequency fails validation."""
        config = GAConfig(log_frequency=0)
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            config.validate()

    def test_invalid_tree_depth(self):
        """Test that invalid tree_depth fails validation."""
        config = GAConfig(tree_depth=0)
        with pytest.raises(ValueError, match="tree_depth must be positive"):
            config.validate()

    def test_invalid_n_features(self):
        """Test that invalid n_features fails validation."""
        config = GAConfig(n_features=0)
        with pytest.raises(ValueError, match="n_features must be positive"):
            config.validate()

    def test_inconsistent_tree_structure(self):
        """Test that inconsistent tree structure fails validation."""
        # n_splits inconsistent with tree_depth
        config = GAConfig(tree_depth=3, n_splits=5)  # Should be 7
        with pytest.raises(ValueError, match="n_splits.*inconsistent with tree_depth"):
            config.validate()

        # n_leaves inconsistent with tree_depth
        config = GAConfig(tree_depth=3, n_leaves=10)  # Should be 8
        with pytest.raises(ValueError, match="n_leaves.*inconsistent with tree_depth"):
            config.validate()


class TestGAConfigAccessors:
    """Test GAConfig parameter access."""

    def test_repr(self):
        """Test __repr__ output."""
        config = GAConfig(name='test', population_size=30)

        repr_str = repr(config)

        assert "GAConfig" in repr_str
        assert "name='test'" in repr_str
        assert "pop=30" in repr_str

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = GAConfig()

        # Verify some defaults
        assert config.population_size == 50
        assert config.crossover_rate == 0.70
        assert config.selection_method == "tournament"
        assert config.seed == 42

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = GAConfig(
            name='custom',
            population_size=100,
            crossover_rate=0.60,
            mutation_rate=0.40,
            selection_method='roulette'
        )

        assert config.name == 'custom'
        assert config.population_size == 100
        assert config.crossover_rate == 0.60
        assert config.mutation_rate == 0.40
        assert config.selection_method == 'roulette'

    def test_raw_data_stored(self):
        """Test that raw YAML data is stored."""
        config = GAConfig.from_yaml('config/ga/default.yaml')

        # Check raw data is present
        assert isinstance(config._raw_data, dict)
        assert len(config._raw_data) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
