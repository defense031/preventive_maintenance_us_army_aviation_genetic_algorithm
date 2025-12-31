"""
Selection Strategies for Genetic Algorithm

Provides parent selection methods for generating the next generation:
- Tournament selection (primary, recommended)
- Roulette wheel selection (fitness-proportional, alternative)

Tournament selection is preferred for this application because:
1. More robust to fitness scaling issues
2. No need to shift fitness to positive range
3. Easier to tune selection pressure via tournament size
4. Works well with small populations (20-50)
5. More parallelizable than roulette
"""

from typing import List
import numpy as np
from policy.chromosome import Chromosome


def tournament_selection(
    population: List[Chromosome],
    fitness_scores: List[float],
    tournament_size: int = 3,
    rng: np.random.Generator = None
) -> Chromosome:
    """Select a parent using tournament selection.

    Randomly samples `tournament_size` individuals from the population and
    returns the one with the highest fitness. Higher tournament size increases
    selection pressure (favors best individuals more strongly).

    Args:
        population: List of chromosomes to select from
        fitness_scores: Parallel list of fitness values (higher is better)
        tournament_size: Number of individuals to compete (default: 3)
        rng: Random number generator for determinism

    Returns:
        Selected chromosome (not a copy)

    Raises:
        ValueError: If tournament_size invalid or population too small

    Example:
        >>> pop = [Chromosome.random(...) for _ in range(50)]
        >>> fitness = [0.5, 0.7, 0.3, ...]  # 50 values
        >>> rng = np.random.default_rng(42)
        >>> parent = tournament_selection(pop, fitness, tournament_size=3, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(population)

    if n == 0:
        raise ValueError("Population is empty")

    if n != len(fitness_scores):
        raise ValueError(
            f"Population size ({n}) does not match fitness scores ({len(fitness_scores)})"
        )

    if tournament_size < 1:
        raise ValueError(f"Tournament size must be at least 1, got {tournament_size}")

    if tournament_size > n:
        raise ValueError(
            f"Tournament size ({tournament_size}) cannot exceed population size ({n})"
        )

    # Sample tournament_size individuals without replacement
    tournament_indices = rng.choice(n, size=tournament_size, replace=False)

    # Find the one with highest fitness
    best_idx = tournament_indices[0]
    best_fitness = fitness_scores[best_idx]

    for idx in tournament_indices[1:]:
        if fitness_scores[idx] > best_fitness:
            best_fitness = fitness_scores[idx]
            best_idx = idx

    return population[best_idx]


def roulette_selection(
    population: List[Chromosome],
    fitness_scores: List[float],
    rng: np.random.Generator = None
) -> Chromosome:
    """Select a parent using fitness-proportional (roulette wheel) selection.

    Probability of selecting an individual is proportional to its fitness.
    Requires all fitness scores to be non-negative (will shift if needed).

    Tournament selection is generally preferred over roulette for this
    application, but roulette is provided as an alternative.

    Args:
        population: List of chromosomes to select from
        fitness_scores: Parallel list of fitness values (higher is better)
        rng: Random number generator for determinism

    Returns:
        Selected chromosome (not a copy)

    Raises:
        ValueError: If population empty or fitness values invalid

    Example:
        >>> pop = [Chromosome.random(...) for _ in range(50)]
        >>> fitness = [0.5, 0.7, 0.3, ...]
        >>> rng = np.random.default_rng(42)
        >>> parent = roulette_selection(pop, fitness, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(population)

    if n == 0:
        raise ValueError("Population is empty")

    if n != len(fitness_scores):
        raise ValueError(
            f"Population size ({n}) does not match fitness scores ({len(fitness_scores)})"
        )

    # Convert to numpy array for easier manipulation
    fitness_array = np.array(fitness_scores)

    # Shift to ensure all positive (if any negative)
    min_fitness = np.min(fitness_array)
    if min_fitness < 0:
        fitness_array = fitness_array - min_fitness + 1e-6

    # Handle case where all fitness equal (uniform random selection)
    if np.all(fitness_array == fitness_array[0]):
        return population[rng.integers(0, n)]

    # Compute selection probabilities
    total_fitness = np.sum(fitness_array)

    if total_fitness == 0:
        # All fitness scores are zero, use uniform random selection
        return population[rng.integers(0, n)]

    probabilities = fitness_array / total_fitness

    # Sample according to probabilities
    selected_idx = rng.choice(n, p=probabilities)

    return population[selected_idx]


def select_parents(
    population: List[Chromosome],
    fitness_scores: List[float],
    n_parents: int,
    method: str = "tournament",
    tournament_size: int = 3,
    rng: np.random.Generator = None
) -> List[Chromosome]:
    """Select multiple parents for breeding.

    Convenience function to select n_parents using the specified method.

    Args:
        population: List of chromosomes to select from
        fitness_scores: Parallel list of fitness values
        n_parents: Number of parents to select
        method: Selection method ("tournament" or "roulette")
        tournament_size: Tournament size (if method=="tournament")
        rng: Random number generator

    Returns:
        List of n_parents selected chromosomes

    Raises:
        ValueError: If method invalid or n_parents invalid

    Example:
        >>> pop = [Chromosome.random(...) for _ in range(50)]
        >>> fitness = [...]  # 50 values
        >>> parents = select_parents(pop, fitness, n_parents=20, method="tournament")
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_parents < 0:
        raise ValueError(f"n_parents must be non-negative, got {n_parents}")

    if method == "tournament":
        return [
            tournament_selection(population, fitness_scores, tournament_size, rng)
            for _ in range(n_parents)
        ]
    elif method == "roulette":
        return [
            roulette_selection(population, fitness_scores, rng)
            for _ in range(n_parents)
        ]
    else:
        raise ValueError(
            f"Unknown selection method: '{method}'. "
            "Valid options: 'tournament', 'roulette'"
        )
