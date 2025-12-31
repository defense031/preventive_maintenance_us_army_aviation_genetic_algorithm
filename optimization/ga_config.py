"""
Genetic Algorithm Configuration Management

Provides GAConfig dataclass for loading, validating, and accessing GA parameters
from YAML configuration files. Centralizes all GA hyperparameters with validation
and convenient accessors.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class GAConfig:
    """Genetic Algorithm configuration loaded from YAML.

    Centralizes all GA hyperparameters with validation and convenient access.
    Supports nested configuration structure matching config/ga/default.yaml.

    Attributes:
        name: Configuration name/identifier
        description: Human-readable description
        version: Configuration version

        # Population
        population_size: Number of individuals in population
        elite_count: Number of elite individuals to preserve

        # Genetic operators
        crossover_rate: Probability of crossover (vs returning clones)
        mutation_rate: Probability of mutation per gene
        blx_alpha: Blend crossover alpha parameter

        # Selection
        selection_method: Selection method ("tournament" or "roulette")
        tournament_size: Tournament size (if using tournament selection)

        # Fitness
        fitness_weight_mission_success: Weight for mission success component
        fitness_weight_or: Weight for operational readiness component
        fitness_weight_flight_hours: Weight for flight hours component
        baseline_max_flight_hours: Maximum flight hours for normalization

        # Evaluation
        episodes_per_chromosome: Number of episodes to evaluate each chromosome
        parallel_workers: Number of CPU cores for parallel evaluation

        # Convergence
        max_generations: Maximum number of generations to evolve
        early_stopping_patience: Stop if no improvement for N generations
        improvement_threshold: Minimum fitness improvement to count

        # Checkpointing
        checkpointing_enabled: Enable checkpoint saving
        checkpoint_frequency: Save checkpoint every N generations
        checkpoint_save_dir: Directory for checkpoint files
        checkpoint_filename_pattern: Filename pattern for checkpoints

        # Logging
        logging_verbose: Enable detailed progress logging
        log_dir: Directory for log files
        log_frequency: Log statistics every N generations
        track_diversity: Track population diversity
        track_statistics: Track fitness statistics

        # Chromosome
        tree_depth: Decision tree depth
        n_features: Number of per-aircraft features
        n_splits: Number of decision splits
        n_leaves: Number of terminal leaves
        n_buckets: Number of FMC buckets

        # Paths
        feature_config_path: Path to feature encoder configuration
        simulation_config_path: Path to simulation configuration
        results_dir: Base directory for GA results

        # Random seed
        seed: Base random seed for reproducibility

        # Output options
        save_best_chromosome: Save best chromosome from each generation
        save_final_population: Save entire final population
        save_fitness_history: Save fitness progression over generations
        generate_plots: Generate fitness convergence plots

    Example:
        >>> config = GAConfig.from_yaml('config/ga/default.yaml')
        >>> print(f"Population size: {config.population_size}")
        >>> print(f"Crossover rate: {config.crossover_rate}")
    """

    # Metadata
    name: str = "ga_default"
    description: str = ""
    version: str = "1.0.0"

    # Population settings
    population_size: int = 50
    elite_count: int = 4

    # Genetic operators
    crossover_rate: float = 0.70
    mutation_rate: float = 0.30  # Legacy - kept for backwards compat
    blx_alpha: float = 0.5

    # Adaptive mutation parameters
    mutation_start_rate: float = 0.50  # Initial mutation rate (50%)
    mutation_min_rate: float = 0.05  # Minimum mutation rate after decay (5%)
    mutation_start_sigma: float = 0.50  # Initial mutation sigma (50% of range)
    mutation_min_sigma: float = 0.10  # Minimum mutation sigma after decay (10%)
    mutation_decay_point: float = 0.75  # Fraction of max_gen to reach min

    # Selection strategy
    selection_method: str = "tournament"
    tournament_size: int = 3

    # Fitness function
    fitness_weight_mission_success: float = 0.70
    fitness_weight_or: float = 0.15
    fitness_weight_flight_hours: float = 0.15
    baseline_max_flight_hours: float = 4563.55

    # Evaluation settings
    episodes_per_chromosome: int = 100
    parallel_workers: int = 6

    # Convergence & stopping criteria
    max_generations: int = 50
    early_stopping_patience: int = 10
    improvement_threshold: float = 0.001

    # Checkpointing
    checkpointing_enabled: bool = True
    checkpoint_frequency: int = 10
    checkpoint_save_dir: str = "results/ga_checkpoints"
    checkpoint_filename_pattern: str = "checkpoint_gen{generation:03d}.pkl"

    # Logging
    logging_verbose: bool = True
    log_dir: str = "results/ga_logs"
    log_frequency: int = 1
    track_diversity: bool = True
    track_statistics: bool = True

    # Chromosome structure
    config_type: str = 'simple'
    tree_depth: int = 3
    n_features: int = 4  # Updated: da_line_deviation is now single signed feature
    n_fleet_features: int = 0  # 0 for simple config, 8 for medium/full
    n_splits: int = 7
    n_leaves: int = 8
    n_buckets: int = 4
    trainable_rul_threshold: bool = False  # If True, rul_threshold becomes a trainable gene (5-100h)

    # Configuration paths
    feature_config_path: str = "config/features/simple_dt.yaml"
    simulation_config_path: str = "config/default.yaml"
    results_dir: str = "results/ga_optimization"

    # Random seed
    seed: int = 42

    # Output options
    save_best_chromosome: bool = True
    save_final_population: bool = True
    save_fitness_history: bool = True
    generate_plots: bool = True

    # Raw YAML data (for debugging)
    _raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GAConfig':
        """Load GA configuration from YAML file.

        Args:
            yaml_path: Path to GA configuration YAML file

        Returns:
            GAConfig instance with parameters loaded from YAML

        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If YAML is malformed or missing required fields

        Example:
            >>> config = GAConfig.from_yaml('config/ga/default.yaml')
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"GA config file not found: {yaml_path}")

        # Load YAML
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

        # Extract nested configuration
        config_kwargs = {
            # Metadata
            'name': data.get('name', 'ga_default'),
            'description': data.get('description', ''),
            'version': data.get('version', '1.0.0'),

            # Population
            'population_size': data.get('population', {}).get('size', 50),
            'elite_count': data.get('population', {}).get('elite_count', 4),

            # Operators
            'crossover_rate': data.get('operators', {}).get('crossover_rate', 0.70),
            'mutation_rate': data.get('operators', {}).get('mutation_rate', 0.30),
            'blx_alpha': data.get('operators', {}).get('blx_alpha', 0.5),

            # Adaptive mutation
            'mutation_start_rate': data.get('operators', {}).get('mutation_start_rate', 0.50),
            'mutation_min_rate': data.get('operators', {}).get('mutation_min_rate', 0.05),
            'mutation_start_sigma': data.get('operators', {}).get('mutation_start_sigma', 0.50),
            'mutation_min_sigma': data.get('operators', {}).get('mutation_min_sigma', 0.10),
            'mutation_decay_point': data.get('operators', {}).get('mutation_decay_point', 0.75),

            # Selection
            'selection_method': data.get('selection', {}).get('method', 'tournament'),
            'tournament_size': data.get('selection', {}).get('tournament_size', 3),

            # Fitness
            'fitness_weight_mission_success': data.get('fitness', {}).get('weights', {}).get('mission_success', 0.70),
            'fitness_weight_or': data.get('fitness', {}).get('weights', {}).get('operational_readiness', 0.15),
            'fitness_weight_flight_hours': data.get('fitness', {}).get('weights', {}).get('flight_hours', 0.15),
            'baseline_max_flight_hours': data.get('fitness', {}).get('baseline_max_flight_hours', 4563.55),

            # Evaluation
            'episodes_per_chromosome': data.get('evaluation', {}).get('episodes_per_chromosome', 100),
            'parallel_workers': data.get('evaluation', {}).get('parallel_workers', 6),

            # Convergence
            'max_generations': data.get('convergence', {}).get('max_generations', 50),
            'early_stopping_patience': data.get('convergence', {}).get('early_stopping_patience', 10),
            'improvement_threshold': data.get('convergence', {}).get('improvement_threshold', 0.001),

            # Checkpointing
            'checkpointing_enabled': data.get('checkpointing', {}).get('enabled', True),
            'checkpoint_frequency': data.get('checkpointing', {}).get('frequency', 10),
            'checkpoint_save_dir': data.get('checkpointing', {}).get('save_dir', 'results/ga_checkpoints'),
            'checkpoint_filename_pattern': data.get('checkpointing', {}).get('filename_pattern', 'checkpoint_gen{generation:03d}.pkl'),

            # Logging
            'logging_verbose': data.get('logging', {}).get('verbose', True),
            'log_dir': data.get('logging', {}).get('log_dir', 'results/ga_logs'),
            'log_frequency': data.get('logging', {}).get('log_frequency', 1),
            'track_diversity': data.get('logging', {}).get('track_diversity', True),
            'track_statistics': data.get('logging', {}).get('track_statistics', True),

            # Chromosome
            'config_type': data.get('chromosome', {}).get('config_type', 'simple'),
            'tree_depth': data.get('chromosome', {}).get('tree_depth', 3),
            'n_features': data.get('chromosome', {}).get('n_features', 4),
            'n_fleet_features': data.get('chromosome', {}).get('n_fleet_features', 0),
            'n_splits': data.get('chromosome', {}).get('n_splits', 7),
            'n_leaves': data.get('chromosome', {}).get('n_leaves', 8),
            'n_buckets': data.get('chromosome', {}).get('n_buckets', 4),
            'trainable_rul_threshold': data.get('chromosome', {}).get('trainable_rul_threshold', False),

            # Paths
            'feature_config_path': data.get('feature_encoder', {}).get('config_path', 'config/features/simple_dt.yaml'),
            'simulation_config_path': data.get('simulation', {}).get('config_path', 'config/default.yaml'),
            'results_dir': data.get('output', {}).get('results_dir', 'results/ga_optimization'),

            # Seed
            'seed': data.get('simulation', {}).get('seed', 42),

            # Output
            'save_best_chromosome': data.get('output', {}).get('save_best_chromosome', True),
            'save_final_population': data.get('output', {}).get('save_final_population', True),
            'save_fitness_history': data.get('output', {}).get('save_fitness_history', True),
            'generate_plots': data.get('output', {}).get('generate_plots', True),

            # Store raw data
            '_raw_data': data
        }

        # Create instance
        config = cls(**config_kwargs)

        # Validate
        config.validate()

        return config

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameters are invalid
        """
        # Population validation
        if self.population_size <= 0:
            raise ValueError(f"population_size must be positive, got {self.population_size}")

        if self.elite_count < 0:
            raise ValueError(f"elite_count must be non-negative, got {self.elite_count}")

        if self.elite_count > self.population_size:
            raise ValueError(
                f"elite_count ({self.elite_count}) cannot exceed population_size ({self.population_size})"
            )

        # Operators validation
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0,1], got {self.crossover_rate}")

        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0,1], got {self.mutation_rate}")

        if self.blx_alpha < 0:
            raise ValueError(f"blx_alpha must be non-negative, got {self.blx_alpha}")

        # Adaptive mutation validation
        if self.mutation_min_rate > self.mutation_start_rate:
            raise ValueError(
                f"mutation_min_rate ({self.mutation_min_rate}) cannot exceed "
                f"mutation_start_rate ({self.mutation_start_rate})"
            )
        if self.mutation_min_sigma > self.mutation_start_sigma:
            raise ValueError(
                f"mutation_min_sigma ({self.mutation_min_sigma}) cannot exceed "
                f"mutation_start_sigma ({self.mutation_start_sigma})"
            )
        if not 0.0 < self.mutation_decay_point <= 1.0:
            raise ValueError(
                f"mutation_decay_point must be in (0, 1], got {self.mutation_decay_point}"
            )

        # Selection validation
        if self.selection_method not in ['tournament', 'roulette']:
            raise ValueError(
                f"selection_method must be 'tournament' or 'roulette', got '{self.selection_method}'"
            )

        if self.tournament_size < 1:
            raise ValueError(f"tournament_size must be at least 1, got {self.tournament_size}")

        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size ({self.tournament_size}) cannot exceed population_size ({self.population_size})"
            )

        # Fitness validation
        total_weight = (
            self.fitness_weight_mission_success +
            self.fitness_weight_or +
            self.fitness_weight_flight_hours
        )
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Fitness weights must sum to 1.0, got {total_weight:.6f}"
            )

        if self.baseline_max_flight_hours <= 0:
            raise ValueError(
                f"baseline_max_flight_hours must be positive, got {self.baseline_max_flight_hours}"
            )

        # Evaluation validation
        if self.episodes_per_chromosome <= 0:
            raise ValueError(
                f"episodes_per_chromosome must be positive, got {self.episodes_per_chromosome}"
            )

        if self.parallel_workers <= 0:
            raise ValueError(
                f"parallel_workers must be positive, got {self.parallel_workers}"
            )

        # Convergence validation
        if self.max_generations <= 0:
            raise ValueError(f"max_generations must be positive, got {self.max_generations}")

        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}"
            )

        if self.improvement_threshold < 0:
            raise ValueError(
                f"improvement_threshold must be non-negative, got {self.improvement_threshold}"
            )

        # Checkpointing validation
        if self.checkpoint_frequency <= 0:
            raise ValueError(
                f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}"
            )

        # Logging validation
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be positive, got {self.log_frequency}")

        # Chromosome validation
        if self.config_type not in ['simple', 'medium', 'full']:
            raise ValueError(
                f"config_type must be 'simple', 'medium', or 'full', got '{self.config_type}'"
            )

        if self.tree_depth <= 0:
            raise ValueError(f"tree_depth must be positive, got {self.tree_depth}")

        if self.n_features <= 0:
            raise ValueError(f"n_features must be positive, got {self.n_features}")

        if self.n_fleet_features < 0:
            raise ValueError(f"n_fleet_features must be non-negative, got {self.n_fleet_features}")

        # For simple config, n_fleet_features should be 0
        # For medium/full config, n_fleet_features should be positive
        if self.config_type == 'simple' and self.n_fleet_features != 0:
            raise ValueError(f"simple config should have n_fleet_features=0, got {self.n_fleet_features}")
        if self.config_type in ['medium', 'full'] and self.n_fleet_features <= 0:
            raise ValueError(f"{self.config_type} config requires n_fleet_features > 0, got {self.n_fleet_features}")

        # Validate tree structure consistency
        expected_splits = 2 ** self.tree_depth - 1
        if self.n_splits != expected_splits:
            raise ValueError(
                f"n_splits ({self.n_splits}) inconsistent with tree_depth ({self.tree_depth}). "
                f"Expected {expected_splits}"
            )

        expected_leaves = 2 ** self.tree_depth
        if self.n_leaves != expected_leaves:
            raise ValueError(
                f"n_leaves ({self.n_leaves}) inconsistent with tree_depth ({self.tree_depth}). "
                f"Expected {expected_leaves}"
            )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"GAConfig(name='{self.name}', "
            f"pop={self.population_size}, "
            f"elite={self.elite_count}, "
            f"cx={self.crossover_rate}, "
            f"mut={self.mutation_rate}, "
            f"selection={self.selection_method}, "
            f"episodes={self.episodes_per_chromosome}, "
            f"workers={self.parallel_workers}, "
            f"max_gen={self.max_generations})"
        )
