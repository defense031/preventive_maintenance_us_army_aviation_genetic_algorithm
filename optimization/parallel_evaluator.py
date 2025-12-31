"""
Parallel Fitness Evaluator for Genetic Algorithm

Distributes chromosome fitness evaluation across multiple CPU cores using
multiprocessing. Enables efficient population-wide evaluation by parallelizing
the computationally expensive simulation episodes.

Performance:
    - Single-threaded: 50 chromosomes × 100 episodes ≈ 25 minutes
    - 6 cores: ≈ 4-5 minutes (near-linear speedup)
"""

from typing import List, Tuple, Dict, Optional
import multiprocessing as mp
from functools import partial
import numpy as np

from policy.chromosome import Chromosome
from optimization.fitness_evaluator import FitnessEvaluator
from utils.config import SimulationConfig


def _evaluate_chromosome_worker(
    args: Tuple[int, Chromosome, SimulationConfig, Dict]
) -> Tuple[int, float, Dict]:
    """Worker function to evaluate a single chromosome.

    This function runs in a separate process. It creates a FitnessEvaluator
    and evaluates the given chromosome.

    Args:
        args: Tuple of (chromosome_idx, chromosome, sim_config, evaluator_kwargs)

    Returns:
        Tuple of (chromosome_idx, fitness, metrics)

    Note:
        This function must be at module level (not nested) to be picklable
        by multiprocessing.
    """
    chromosome_idx, chromosome, sim_config, evaluator_kwargs = args

    # Create evaluator in worker process
    evaluator = FitnessEvaluator(
        sim_config=sim_config,
        **evaluator_kwargs
    )

    # Compute episode offset to ensure different random seeds per chromosome
    episode_offset = chromosome_idx * evaluator_kwargs.get('n_episodes', 100)

    # Evaluate chromosome
    fitness, metrics = evaluator.evaluate(chromosome, episode_offset=episode_offset)

    return (chromosome_idx, fitness, metrics)


class ParallelEvaluator:
    """Parallel chromosome fitness evaluator using multiprocessing.

    Distributes fitness evaluation across multiple CPU cores for efficient
    population-wide evaluation. Each worker process evaluates one chromosome
    independently with its own simulation environment.

    Attributes:
        evaluator_config: Configuration dict for FitnessEvaluator
        sim_config: Simulation configuration
        n_workers: Number of parallel worker processes
        verbose: Enable detailed logging

    Example:
        >>> parallel_eval = ParallelEvaluator(
        ...     sim_config=config,
        ...     n_workers=6,
        ...     n_episodes=100,
        ...     seed=42
        ... )
        >>> population = [Chromosome.random(...) for _ in range(50)]
        >>> fitness_scores, all_metrics = parallel_eval.evaluate_population(population)
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        feature_config_path: str = "config/features/simple_dt.yaml",
        weight_mission_success: float = 0.70,
        weight_or: float = 0.15,
        weight_flight_hours: float = 0.15,
        baseline_max_flight_hours: float = 4563.55,
        n_episodes: int = 100,
        n_workers: int = 6,
        seed: int = 42,
        verbose: bool = False
    ):
        """Initialize parallel evaluator.

        Args:
            sim_config: Simulation configuration object
            feature_config_path: Path to feature encoder YAML (default: simple_dt.yaml)
            weight_mission_success: Weight for mission success (default: 0.70)
            weight_or: Weight for mean OR (default: 0.15)
            weight_flight_hours: Weight for flight hours (default: 0.15)
            baseline_max_flight_hours: Normalizer for flight hours (default: 4563.55)
            n_episodes: Number of evaluation episodes per chromosome (default: 100)
            n_workers: Number of parallel workers (default: 6)
            seed: Base random seed for reproducibility
            verbose: Enable detailed logging

        Raises:
            ValueError: If n_workers <= 0
        """
        if n_workers <= 0:
            raise ValueError(f"n_workers must be positive, got {n_workers}")

        self.sim_config = sim_config
        self.n_workers = n_workers
        self.verbose = verbose

        # Store evaluator configuration (passed to worker processes)
        self.evaluator_config = {
            'feature_config_path': feature_config_path,
            'weight_mission_success': weight_mission_success,
            'weight_or': weight_or,
            'weight_flight_hours': weight_flight_hours,
            'baseline_max_flight_hours': baseline_max_flight_hours,
            'n_episodes': n_episodes,
            'seed': seed,
            'verbose': False  # Disable per-worker logging to avoid clutter
        }

    def evaluate_population(
        self,
        population: List[Chromosome],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[float], List[Dict]]:
        """Evaluate fitness for entire population in parallel.

        Distributes chromosome evaluation across worker processes. Each worker
        evaluates one chromosome at a time using a unique episode offset for
        random seed diversity.

        Args:
            population: List of chromosomes to evaluate
            progress_callback: Optional callback function(completed, total) for progress tracking

        Returns:
            Tuple of (fitness_scores, detailed_metrics)

            fitness_scores: List of fitness values (parallel to population)
            detailed_metrics: List of metric dicts (parallel to population)

        Example:
            >>> def progress(completed, total):
            ...     print(f"Progress: {completed}/{total}")
            >>> fitness_scores, metrics = evaluator.evaluate_population(
            ...     population,
            ...     progress_callback=progress
            ... )

        Performance:
            - Population=50, Episodes=100, Workers=6: ~4-5 minutes
            - Near-linear speedup with more cores (up to I/O bottleneck)
        """
        if len(population) == 0:
            return [], []

        if self.verbose:
            print(f"\n🚀 Starting parallel evaluation:")
            print(f"   Population size: {len(population)}")
            print(f"   Workers: {self.n_workers}")
            print(f"   Episodes per chromosome: {self.evaluator_config['n_episodes']}")
            print(f"   Total episodes: {len(population) * self.evaluator_config['n_episodes']}")

        # Prepare work items: (idx, chromosome, sim_config, evaluator_kwargs)
        work_items = [
            (idx, chrom, self.sim_config, self.evaluator_config)
            for idx, chrom in enumerate(population)
        ]

        # Create process pool and evaluate in parallel
        with mp.Pool(processes=self.n_workers) as pool:
            # Use imap_unordered for better memory efficiency and progress tracking
            results = []

            for completed, result in enumerate(
                pool.imap_unordered(_evaluate_chromosome_worker, work_items),
                start=1
            ):
                results.append(result)

                # Progress callback
                if progress_callback is not None:
                    progress_callback(completed, len(population))

                # Progress logging
                if self.verbose and completed % 10 == 0:
                    print(f"   Completed: {completed}/{len(population)} chromosomes")

        if self.verbose:
            print(f"✅ Parallel evaluation complete: {len(population)} chromosomes")

        # Sort results by chromosome index (imap_unordered doesn't preserve order)
        results.sort(key=lambda x: x[0])

        # Extract fitness scores and metrics
        fitness_scores = [fitness for _, fitness, _ in results]
        detailed_metrics = [metrics for _, _, metrics in results]

        return fitness_scores, detailed_metrics

    def evaluate_single(
        self,
        chromosome: Chromosome,
        chromosome_idx: int = 0
    ) -> Tuple[float, Dict]:
        """Evaluate a single chromosome (non-parallel).

        Useful for debugging or evaluating individual chromosomes without
        the overhead of multiprocessing.

        Args:
            chromosome: Chromosome to evaluate
            chromosome_idx: Index for episode offset (default: 0)

        Returns:
            Tuple of (fitness, metrics)

        Example:
            >>> fitness, metrics = evaluator.evaluate_single(chromosome)
        """
        # Create evaluator (in main process)
        evaluator = FitnessEvaluator(
            sim_config=self.sim_config,
            **self.evaluator_config
        )

        # Compute episode offset
        episode_offset = chromosome_idx * self.evaluator_config['n_episodes']

        # Evaluate
        return evaluator.evaluate(chromosome, episode_offset=episode_offset)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ParallelEvaluator("
            f"n_workers={self.n_workers}, "
            f"n_episodes={self.evaluator_config['n_episodes']}, "
            f"weights=[MS:{self.evaluator_config['weight_mission_success']:.2f}, "
            f"OR:{self.evaluator_config['weight_or']:.2f}, "
            f"FH:{self.evaluator_config['weight_flight_hours']:.2f}])"
        )
