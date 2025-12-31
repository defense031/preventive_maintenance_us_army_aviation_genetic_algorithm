"""
Population Management for Genetic Algorithm

Manages a population of chromosomes throughout the GA evolutionary process.
Provides functionality for:
- Population initialization
- Fitness tracking
- Elitism selection
- Statistical analysis
- Diversity measurement
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from policy.chromosome import Chromosome


class Population:
    """Manages a population of chromosomes for genetic algorithm optimization.

    This class maintains a collection of chromosomes and their associated fitness
    scores, providing operations for initialization, elitism, statistics, and
    diversity tracking.

    Attributes:
        size: Number of individuals in the population
        tree_depth: Depth of decision trees (all chromosomes must match)
        n_features: Number of features (all chromosomes must match)
        seed: Random seed for reproducibility
        chromosomes: List of Chromosome objects
        fitness_scores: Parallel list of fitness values
        rng: NumPy random generator for deterministic operations
        generation: Current generation number

    Example:
        >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
        >>> pop.initialize_random()
        >>> pop.update_fitness(0, 0.85)
        >>> best_chrom, best_fitness = pop.get_best()
        >>> elite = pop.get_top_k(k=4)
    """

    def __init__(
        self,
        size: int = 50,
        tree_depth: int = 3,
        n_features: int = 5,
        seed: int = 42,
        config_type: str = 'simple',
        n_fleet_features: int = 7,
        trainable_rul_threshold: bool = False,
        feature_bounds: Optional[List[Tuple[float, float]]] = None
    ):
        """Initialize population with given parameters.

        Args:
            size: Population size (default: 50)
            tree_depth: Tree depth for chromosomes (default: 3)
            n_features: Number of features for chromosomes (default: 5)
            seed: Random seed for reproducibility (default: 42)
            config_type: Chromosome config type ('simple', 'medium', 'full')
            n_fleet_features: Number of fleet-level features (default: 8)
            trainable_rul_threshold: If True, rul_threshold becomes a trainable gene (5-100h)
            feature_bounds: Bounds for each feature [(min, max), ...]. Required for initialization.

        Raises:
            ValueError: If size, tree_depth, or n_features <= 0
        """
        if size <= 0:
            raise ValueError(f"Population size must be positive, got {size}")
        if tree_depth <= 0:
            raise ValueError(f"Tree depth must be positive, got {tree_depth}")
        if n_features <= 0:
            raise ValueError(f"Number of features must be positive, got {n_features}")
        if config_type not in ['simple', 'medium', 'full']:
            raise ValueError(f"config_type must be 'simple', 'medium', or 'full', got {config_type}")

        self.size = size
        self.tree_depth = tree_depth
        self.n_features = n_features
        self.seed = seed
        self.config_type = config_type
        self.n_fleet_features = n_fleet_features
        self.trainable_rul_threshold = trainable_rul_threshold
        self.feature_bounds = feature_bounds

        # Population state
        self.chromosomes: List[Chromosome] = []
        self.fitness_scores: List[float] = []
        self.generation = 0

        # Random number generator for determinism
        self.rng = np.random.default_rng(seed)

    def initialize_random(self) -> None:
        """Initialize population with random chromosomes.

        Generates `size` random chromosomes and initializes fitness scores to 0.0.
        This should be called once after construction before starting evolution.

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> len(pop.chromosomes)
            50
        """
        self.chromosomes = [
            Chromosome.random(
                tree_depth=self.tree_depth,
                n_features=self.n_features,
                feature_bounds=self.feature_bounds,
                config_type=self.config_type,
                n_fleet_features=self.n_fleet_features,
                trainable_rul_threshold=self.trainable_rul_threshold,
                rng=self.rng
            )
            for _ in range(self.size)
        ]

        # Initialize fitness scores to 0.0
        self.fitness_scores = [0.0] * self.size

        self.generation = 0

    def update_fitness(self, idx: int, fitness: float) -> None:
        """Update fitness score for a specific individual.

        Args:
            idx: Index of individual to update (0-indexed)
            fitness: New fitness value (higher is better)

        Raises:
            IndexError: If idx is out of bounds
            ValueError: If fitness is NaN or infinite

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> pop.update_fitness(0, 0.75)
            >>> pop.fitness_scores[0]
            0.75
        """
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for population size {self.size}")

        if np.isnan(fitness) or np.isinf(fitness):
            raise ValueError(f"Invalid fitness value: {fitness}")

        self.fitness_scores[idx] = fitness

    def get_top_k(self, k: int = 4) -> List[Tuple[Chromosome, float]]:
        """Get the k best individuals (elite selection).

        Returns chromosomes with highest fitness scores. Used for elitism to
        preserve best solutions across generations.

        Args:
            k: Number of top individuals to return (default: 4 for 8% elitism with size=50)

        Returns:
            List of (chromosome, fitness) tuples, sorted by fitness (descending)

        Raises:
            ValueError: If k <= 0 or k > population size

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> # ... evaluate fitness ...
            >>> elite = pop.get_top_k(k=4)
            >>> len(elite)
            4
            >>> elite[0][1] >= elite[1][1]  # First is best
            True
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > self.size:
            raise ValueError(f"k ({k}) cannot exceed population size ({self.size})")

        # Pair chromosomes with fitness scores
        pairs = list(zip(self.chromosomes, self.fitness_scores))

        # Sort by fitness (descending = best first)
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        # Return top k
        return pairs_sorted[:k]

    def get_best(self) -> Tuple[Chromosome, float]:
        """Get the single best individual in the population.

        Returns:
            Tuple of (best_chromosome, best_fitness)

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> # ... evaluate fitness ...
            >>> best_chrom, best_fitness = pop.get_best()
        """
        best_idx = np.argmax(self.fitness_scores)
        return self.chromosomes[best_idx], self.fitness_scores[best_idx]

    def get_statistics(self) -> Dict[str, float]:
        """Compute fitness statistics for the population.

        Returns:
            Dictionary with keys:
            - 'best': Maximum fitness
            - 'mean': Mean fitness
            - 'worst': Minimum fitness
            - 'std': Standard deviation of fitness
            - 'median': Median fitness

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> # ... evaluate fitness ...
            >>> stats = pop.get_statistics()
            >>> print(f"Best: {stats['best']:.3f}, Mean: {stats['mean']:.3f}")
        """
        if len(self.fitness_scores) == 0:
            return {
                'best': 0.0,
                'mean': 0.0,
                'worst': 0.0,
                'std': 0.0,
                'median': 0.0
            }

        fitness_array = np.array(self.fitness_scores)

        return {
            'best': float(np.max(fitness_array)),
            'mean': float(np.mean(fitness_array)),
            'worst': float(np.min(fitness_array)),
            'std': float(np.std(fitness_array)),
            'median': float(np.median(fitness_array))
        }

    def compute_diversity(self) -> float:
        """Compute population diversity based on feature_indices.

        Measures average pairwise Hamming distance on feature_indices genes.
        Higher diversity means more varied decision tree structures.

        Returns:
            Average Hamming distance (normalized to [0, 1])
            - 0.0: All chromosomes identical
            - 1.0: Maximum diversity

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> diversity = pop.compute_diversity()
            >>> 0.0 <= diversity <= 1.0
            True
        """
        if len(self.chromosomes) < 2:
            return 0.0

        # Extract feature_indices from all chromosomes
        features = np.array([chrom.feature_indices for chrom in self.chromosomes])

        # Compute pairwise Hamming distances
        n = len(self.chromosomes)
        n_genes = len(self.chromosomes[0].feature_indices)

        total_distance = 0.0
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Hamming distance: count of differing genes
                distance = np.sum(features[i] != features[j])
                total_distance += distance
                pair_count += 1

        # Normalize by maximum possible distance (all genes different)
        # and number of pairs
        if pair_count == 0:
            return 0.0

        avg_distance = total_distance / pair_count
        max_distance = n_genes

        return avg_distance / max_distance

    def replace(self, new_chromosomes: List[Chromosome]) -> None:
        """Replace the entire population with new chromosomes.

        Used to update population after selection, crossover, and mutation.
        Fitness scores are reset to 0.0 and must be re-evaluated.

        Args:
            new_chromosomes: List of new chromosomes (must have length == size)

        Raises:
            ValueError: If new population size doesn't match

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> # ... perform selection/crossover/mutation ...
            >>> pop.replace(new_generation)
            >>> pop.generation
            1
        """
        if len(new_chromosomes) != self.size:
            raise ValueError(
                f"New population size ({len(new_chromosomes)}) "
                f"must match original size ({self.size})"
            )

        self.chromosomes = new_chromosomes
        self.fitness_scores = [0.0] * self.size
        self.generation += 1

    def get_worst(self) -> Tuple[Chromosome, float]:
        """Get the worst individual in the population.

        Returns:
            Tuple of (worst_chromosome, worst_fitness)

        Example:
            >>> pop = Population(size=50, tree_depth=3, n_features=5, seed=42)
            >>> pop.initialize_random()
            >>> # ... evaluate fitness ...
            >>> worst_chrom, worst_fitness = pop.get_worst()
        """
        worst_idx = np.argmin(self.fitness_scores)
        return self.chromosomes[worst_idx], self.fitness_scores[worst_idx]

    def __len__(self) -> int:
        """Return population size."""
        return len(self.chromosomes)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Population(size={self.size}, "
            f"gen={self.generation}, "
            f"depth={self.tree_depth}, "
            f"features={self.n_features}, "
            f"config_type='{self.config_type}')"
        )
