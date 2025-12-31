"""
Island Model Genetic Algorithm for Decision Tree Policy Optimization.

Implements a heterogeneous island model with three islands in a pipeline:

- Oahu (Explorer): High mutation, weak selection - generates diverse experiments
- Maui (Refiner): High crossover, moderate selection - combines good traits
- Big Island (Validator): Moderate selection - validates quality solutions

Pipeline migration every N generations:
Oahu (4) → Maui (2) → Big Island (1) → Oahu (feedback)
"""

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import yaml

from policy.chromosome import Chromosome
from optimization.parallel_evaluator import ParallelEvaluator
from optimization.ga_config import GAConfig
from utils.config import load_config_from_yaml


def load_feature_bounds(feature_config_path: str) -> List[Tuple[float, float]]:
    """Extract feature bounds from feature config YAML.

    Args:
        feature_config_path: Path to feature config YAML file

    Returns:
        List of (min, max) tuples for each per-aircraft feature

    Raises:
        ValueError: If feature config is missing required bounds
    """
    with open(feature_config_path, 'r') as f:
        config = yaml.safe_load(f)

    bounds = []
    for feature in config.get('per_aircraft_features', []):
        feature_bounds = feature.get('bounds')
        if feature_bounds is None:
            raise ValueError(f"Feature '{feature.get('name')}' missing 'bounds' in {feature_config_path}")
        bounds.append(tuple(feature_bounds))

    if not bounds:
        raise ValueError(f"No per_aircraft_features found in {feature_config_path}")

    return bounds


@dataclass
class IslandConfig:
    """Configuration for a single island."""
    name: str
    population_size: int
    elite_count: int
    tournament_size: int
    mutation_start_rate: float
    mutation_min_rate: float
    mutation_start_sigma: float
    mutation_min_sigma: float
    crossover_rate: float
    description: str = ""


# Pre-defined island personalities
ISLAND_CONFIGS = {
    "oahu": IslandConfig(
        name="Oahu",
        population_size=20,         # Exploration lab - generates experiments
        elite_count=2,              # Small buffer - preserve discoveries until migration
        tournament_size=2,          # Weak selection - preserve diversity
        mutation_start_rate=0.70,
        mutation_min_rate=0.30,
        mutation_start_sigma=0.60,
        mutation_min_sigma=0.25,
        crossover_rate=0.50,        # Lower - preserve diversity from mutations
        description="Laboratory - small, high exploration, receives proven winners"
    ),
    "maui": IslandConfig(
        name="Maui",
        population_size=20,         # Medium - polishing ground
        elite_count=3,              # Moderate - preserve good combinations
        tournament_size=3,          # Moderate selection
        mutation_start_rate=0.20,
        mutation_min_rate=0.05,
        mutation_start_sigma=0.25,
        mutation_min_sigma=0.08,
        crossover_rate=0.85,        # High - combine traits aggressively
        description="Refiner - combines and polishes promising variations"
    ),
    "big_island": IslandConfig(
        name="Big Island",
        population_size=25,         # Validation ground - reduced pressure
        elite_count=4,              # High - preserve champions
        tournament_size=3,          # Moderate selection - avoid premature convergence
        mutation_start_rate=0.35,
        mutation_min_rate=0.10,
        mutation_start_sigma=0.35,
        mutation_min_sigma=0.10,
        crossover_rate=0.70,
        description="Brawl - intense competition, only the fittest survive"
    )
}

# Scaled island configs for full config (72 genes vs 31 for medium)
# Populations ~2.2x to handle larger search space while maintaining pipeline ratios
ISLAND_CONFIGS_FULL = {
    "oahu": IslandConfig(
        name="Oahu",
        population_size=45,         # Scaled 2.25x for larger search space
        elite_count=4,              # Scaled proportionally
        tournament_size=2,          # Keep weak selection
        mutation_start_rate=0.70,
        mutation_min_rate=0.30,
        mutation_start_sigma=0.60,
        mutation_min_sigma=0.25,
        crossover_rate=0.50,
        description="Laboratory - scaled for full config search space"
    ),
    "maui": IslandConfig(
        name="Maui",
        population_size=45,         # Scaled 2.25x
        elite_count=6,              # Scaled proportionally
        tournament_size=3,
        mutation_start_rate=0.20,
        mutation_min_rate=0.05,
        mutation_start_sigma=0.25,
        mutation_min_sigma=0.08,
        crossover_rate=0.85,
        description="Refiner - scaled for full config search space"
    ),
    "big_island": IslandConfig(
        name="Big Island",
        population_size=55,         # Scaled 2.2x
        elite_count=8,              # Scaled proportionally
        tournament_size=3,
        mutation_start_rate=0.35,
        mutation_min_rate=0.10,
        mutation_start_sigma=0.35,
        mutation_min_sigma=0.10,
        crossover_rate=0.70,
        description="Brawl - scaled for full config search space"
    )
}

# Pipeline migration: Explore -> Refine -> Validate (with small feedback)
MIGRATION_MAP = {
    "oahu": ("maui", 4),            # Experiments to refiner
    "maui": ("big_island", 2),      # Polished solutions to competition
    "big_island": ("oahu", 1),      # Small feedback to lab
}


class Island:
    """A single island with its own population and evolutionary parameters."""

    def __init__(
        self,
        config: IslandConfig,
        tree_depth: int,
        n_features: int,
        feature_bounds: List[Tuple[float, float]],
        config_type: str = 'simple',
        n_fleet_features: int = 0,
        trainable_rul_threshold: bool = False,
        seed: int = 42
    ):
        self.config = config
        self.name = config.name
        self.tree_depth = tree_depth
        self.n_features = n_features
        self.feature_bounds = feature_bounds
        self.config_type = config_type
        self.n_fleet_features = n_fleet_features
        self.trainable_rul_threshold = trainable_rul_threshold

        # Create RNG with island-specific seed
        self.rng = np.random.default_rng(seed)

        # Initialize population
        self.chromosomes: List[Chromosome] = [
            Chromosome.random(
                tree_depth=tree_depth,
                n_features=n_features,
                feature_bounds=feature_bounds,
                config_type=config_type,
                n_fleet_features=n_fleet_features,
                trainable_rul_threshold=trainable_rul_threshold,
                rng=self.rng
            )
            for _ in range(config.population_size)
        ]

        # Fitness tracking
        self.fitness_scores: List[float] = [0.0] * config.population_size
        self.metrics_list: List[Dict] = [{}] * config.population_size
        self.best_fitness: float = 0.0
        self.best_chromosome: Optional[Chromosome] = None
        self.best_generation: int = 0
        self.best_metrics: Dict = {}

        # History
        self.fitness_history: List[float] = []

    def get_adaptive_mutation_params(self, generation: int, max_generations: int) -> Tuple[float, float]:
        """Calculate adaptive mutation rate and sigma with exponential decay."""
        k = 6.13 / max_generations

        rate = self.config.mutation_min_rate + \
               (self.config.mutation_start_rate - self.config.mutation_min_rate) * \
               np.exp(-k * generation)

        sigma = self.config.mutation_min_sigma + \
                (self.config.mutation_start_sigma - self.config.mutation_min_sigma) * \
                np.exp(-k * generation)

        return max(rate, self.config.mutation_min_rate), max(sigma, self.config.mutation_min_sigma)

    def tournament_select(self) -> Chromosome:
        """Select a chromosome using tournament selection."""
        indices = self.rng.choice(
            len(self.chromosomes),
            size=min(self.config.tournament_size, len(self.chromosomes)),
            replace=False
        )
        best_idx = max(indices, key=lambda i: self.fitness_scores[i])
        return self.chromosomes[best_idx]

    def update_best(self, generation: int) -> None:
        """Update best chromosome tracking after fitness evaluation."""
        if not self.fitness_scores:
            return

        best_idx = int(np.argmax(self.fitness_scores))
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_chromosome = deepcopy(self.chromosomes[best_idx])
            self.best_generation = generation
            if self.metrics_list and best_idx < len(self.metrics_list):
                self.best_metrics = self.metrics_list[best_idx]

        self.fitness_history.append(self.best_fitness)

    def evolve_one_generation(self, generation: int, max_generations: int) -> None:
        """Perform one generation of evolution on this island."""

        # Sort by fitness (descending)
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        sorted_chromosomes = [self.chromosomes[i] for i in sorted_indices]

        # Elitism - keep top individuals
        new_population = [deepcopy(c) for c in sorted_chromosomes[:self.config.elite_count]]

        # Get adaptive mutation parameters
        mut_rate, mut_sigma = self.get_adaptive_mutation_params(generation, max_generations)

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child1, child2 = parent1.crossover(parent2, rng=self.rng)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)

            # Mutation
            child1 = child1.mutate(mutation_rate=mut_rate, mutation_sigma=mut_sigma, rng=self.rng)
            child2 = child2.mutate(mutation_rate=mut_rate, mutation_sigma=mut_sigma, rng=self.rng)

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        self.chromosomes = new_population
        self.fitness_scores = [0.0] * len(new_population)
        self.metrics_list = [{}] * len(new_population)

    def receive_migrants(self, migrants: List[Chromosome]) -> None:
        """Receive migrating chromosomes, replacing worst individuals."""
        if not migrants:
            return

        # Sort by fitness and replace worst
        sorted_indices = np.argsort(self.fitness_scores)
        for i, migrant in enumerate(migrants):
            if i < len(sorted_indices):
                worst_idx = sorted_indices[i]
                self.chromosomes[worst_idx] = deepcopy(migrant)

    def get_emigrants(self, n: int = 2) -> List[Chromosome]:
        """Get top n chromosomes to migrate to another island."""
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        return [deepcopy(self.chromosomes[i]) for i in sorted_indices[:n]]


class IslandGA:
    """
    Island Model Genetic Algorithm.

    Manages multiple islands with different evolutionary characteristics
    and handles migration between them.
    """

    def __init__(
        self,
        ga_config: GAConfig,
        migration_frequency: int = 40
    ):
        self.ga_config = ga_config
        self.migration_frequency = migration_frequency

        # Load feature bounds from feature config (fail-fast if missing)
        self.feature_bounds = load_feature_bounds(ga_config.feature_config_path)

        # Create islands with different seeds
        base_seed = ga_config.seed if ga_config.seed else 42
        self.islands: Dict[str, Island] = {}

        # Select island configs based on config type (full needs larger populations)
        island_configs = ISLAND_CONFIGS_FULL if ga_config.config_type == "full" else ISLAND_CONFIGS

        for i, (key, island_config) in enumerate(island_configs.items()):
            self.islands[key] = Island(
                config=island_config,
                tree_depth=ga_config.tree_depth,
                n_features=ga_config.n_features,
                feature_bounds=self.feature_bounds,
                config_type=ga_config.config_type,
                n_fleet_features=ga_config.n_fleet_features,
                trainable_rul_threshold=ga_config.trainable_rul_threshold,
                seed=base_seed + i * 1000
            )

        # Load simulation config
        self.sim_config = load_config_from_yaml(ga_config.simulation_config_path)

        # Create parallel evaluator
        self.evaluator = ParallelEvaluator(
            sim_config=self.sim_config,
            n_workers=ga_config.parallel_workers,
            n_episodes=ga_config.episodes_per_chromosome,
            seed=base_seed,
            feature_config_path=ga_config.feature_config_path,
            weight_mission_success=ga_config.fitness_weight_mission_success,
            weight_or=ga_config.fitness_weight_or,
            weight_flight_hours=ga_config.fitness_weight_flight_hours,
            baseline_max_flight_hours=ga_config.baseline_max_flight_hours
        )

        # Global tracking
        self.current_generation = 0
        self.global_best_fitness = 0.0
        self.global_best_chromosome: Optional[Chromosome] = None
        self.global_best_generation = 0
        self.global_best_island = ""
        self.global_best_metrics: Dict = {}

        # History
        self.generation_history: List[Dict] = []

    def _evaluate_all_islands(self) -> None:
        """Evaluate fitness for all chromosomes across all islands."""
        # Collect all chromosomes with their island/index info
        all_chromosomes = []
        island_indices = []

        for island_key, island in self.islands.items():
            for i, chrom in enumerate(island.chromosomes):
                all_chromosomes.append(chrom)
                island_indices.append((island_key, i))

        # Evaluate all at once (parallel)
        all_fitness, all_metrics = self.evaluator.evaluate_population(all_chromosomes)

        # Distribute scores back to islands
        for idx, ((island_key, chrom_idx), fitness) in enumerate(zip(island_indices, all_fitness)):
            self.islands[island_key].fitness_scores[chrom_idx] = fitness
            if all_metrics and idx < len(all_metrics):
                self.islands[island_key].metrics_list[chrom_idx] = all_metrics[idx]

    def _migrate(self) -> None:
        """Perform asymmetric migration between islands using MIGRATION_MAP."""
        # Collect emigrants from each source island based on MIGRATION_MAP
        for source_key, (dest_key, n_migrants) in MIGRATION_MAP.items():
            emigrants = self.islands[source_key].get_emigrants(n_migrants)
            self.islands[dest_key].receive_migrants(emigrants)

    def _update_global_best(self) -> None:
        """Update global best across all islands."""
        for key, island in self.islands.items():
            if island.best_fitness > self.global_best_fitness:
                self.global_best_fitness = island.best_fitness
                self.global_best_chromosome = deepcopy(island.best_chromosome) if island.best_chromosome else None
                self.global_best_generation = self.current_generation
                self.global_best_island = key
                self.global_best_metrics = island.best_metrics

    def run(self) -> Dict:
        """Run the island GA optimization."""

        print("\n" + "=" * 70)
        print("  Island Model Genetic Algorithm")
        print("=" * 70)
        print(f"\nIslands:")
        for key, island in self.islands.items():
            print(f"  {island.name:12s} (pop={island.config.population_size:2d}): "
                  f"{island.config.description}")
        print(f"\nMigration: Every {self.migration_frequency} generations")
        print("Asymmetric transfers:")
        for source, (dest, n) in MIGRATION_MAP.items():
            src_name = self.islands[source].name
            dest_name = self.islands[dest].name
            print(f"  {src_name} -> {dest_name}: {n} migrants")
        print("=" * 70 + "\n")

        # Initial evaluation
        print("Evaluating initial populations...")
        self._evaluate_all_islands()

        # Update best for each island
        for island in self.islands.values():
            island.update_best(0)
        self._update_global_best()

        # Track for early stopping - per-island stalls
        # Stop only when ALL islands have stagnated
        island_stalls = {key: 0 for key in self.islands}
        island_last_best = {key: island.best_fitness for key, island in self.islands.items()}

        # Progress bar - wider to show more metrics
        pbar = tqdm(
            total=self.ga_config.max_generations,
            desc="Island GA",
            unit="gen",
            ncols=140
        )

        start_time = time.time()

        for gen in range(self.ga_config.max_generations):
            self.current_generation = gen

            # Evolve each island
            for island in self.islands.values():
                island.evolve_one_generation(gen, self.ga_config.max_generations)

            # Evaluate all
            self._evaluate_all_islands()

            # Update best for each island
            for island in self.islands.values():
                island.update_best(gen)

            # Migration
            if gen > 0 and gen % self.migration_frequency == 0:
                self._migrate()
                pbar.write(f"  Migration at generation {gen}")

            # Update global best
            self._update_global_best()

            # Check for improvement - per island
            for key, island in self.islands.items():
                if island.best_fitness > island_last_best[key] + self.ga_config.improvement_threshold:
                    island_stalls[key] = 0
                    island_last_best[key] = island.best_fitness
                else:
                    island_stalls[key] += 1

            # Minimum stall = most recently improved island
            min_stall = min(island_stalls.values())

            # Record history (with per-island diversity)
            self.generation_history.append({
                'generation': gen,
                'global_best': self.global_best_fitness,
                'oahu_best': self.islands['oahu'].best_fitness,
                'maui_best': self.islands['maui'].best_fitness,
                'big_island_best': self.islands['big_island'].best_fitness,
                # Per-island fitness std (diversity measure)
                'oahu_std': float(np.std(self.islands['oahu'].fitness_scores)),
                'maui_std': float(np.std(self.islands['maui'].fitness_scores)),
                'big_island_std': float(np.std(self.islands['big_island'].fitness_scores)),
            })

            # Update progress with per-island fitness and global metrics
            o_fit = self.islands['oahu'].best_fitness
            m_fit = self.islands['maui'].best_fitness
            b_fit = self.islands['big_island'].best_fitness

            # Get mission success and OR from global best metrics
            ms_pct = 0.0
            or_pct = 0.0
            if self.global_best_metrics:
                ms_pct = self.global_best_metrics.get('mean_mission_success', 0.0) * 100
                or_pct = self.global_best_metrics.get('mean_or', 0.0) * 100

            pbar.set_postfix_str(
                f"O={o_fit:.3f} M={m_fit:.3f} B={b_fit:.3f} | "
                f"MS={ms_pct:.1f}% OR={or_pct:.1f}% Stall={min_stall}"
            )
            pbar.update(1)

            # Log periodically with full details
            if gen % 25 == 0 and gen > 0:
                pbar.write(f"  Gen {gen:4d}: Best={self.global_best_fitness:.4f} ({self.global_best_island}) | "
                          f"O={o_fit:.4f} M={m_fit:.4f} B={b_fit:.4f} | "
                          f"MS={ms_pct:.1f}% OR={or_pct:.1f}%")

            # Early stopping - only when ALL islands have stagnated
            if min_stall >= self.ga_config.early_stopping_patience:
                pbar.write(f"\n  Early stopping: All islands stagnated for {min_stall} generations")
                pbar.write(f"    Per-island stalls: O={island_stalls['oahu']} M={island_stalls['maui']} B={island_stalls['big_island']}")
                break

        pbar.close()
        elapsed = time.time() - start_time

        # Final summary
        print("\n" + "=" * 70)
        print("  Island GA Complete")
        print("=" * 70)
        print(f"\nFinal Results:")
        print(f"  Global Best Fitness: {self.global_best_fitness:.4f}")
        print(f"  Best Island: {self.global_best_island}")
        print(f"  Best Generation: {self.global_best_generation}")
        print(f"  Total Generations: {self.current_generation + 1}")
        print(f"  Runtime: {elapsed/60:.1f} minutes")

        if self.global_best_metrics:
            print(f"\nBest Policy Metrics:")
            print(f"  Mission Success:      {100*self.global_best_metrics.get('mean_mission_success', 0):.1f}%")
            print(f"  Operational Readiness: {100*self.global_best_metrics.get('mean_or', 0):.1f}%")
            print(f"  Flight Hours:         {self.global_best_metrics.get('mean_flight_hours', 0):.1f}")

        print("\nPer-Island Results:")
        for key, island in self.islands.items():
            print(f"  {island.name:12s}: Best={island.best_fitness:.4f} (gen {island.best_generation})")

        return {
            'best_fitness': self.global_best_fitness,
            'best_chromosome': self.global_best_chromosome,
            'best_island': self.global_best_island,
            'best_generation': self.global_best_generation,
            'final_generation': self.current_generation,
            'runtime_minutes': elapsed / 60,
            'best_metrics': self.global_best_metrics,
            'history': self.generation_history
        }

    def save_results(self, output_dir: str) -> None:
        """Save results to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save best chromosome
        if self.global_best_chromosome:
            chrom_path = output_path / "best_chromosome.json"
            with open(chrom_path, 'w') as f:
                json.dump(self.global_best_chromosome.to_dict(), f, indent=2)

        # Save history
        history_path = output_path / "island_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.generation_history, f, indent=2)

        # Save summary
        summary = {
            'best_fitness': self.global_best_fitness,
            'best_island': self.global_best_island,
            'best_generation': self.global_best_generation,
            'final_generation': self.current_generation,
            'best_metrics': self.global_best_metrics,
            'islands': {
                key: {
                    'name': island.name,
                    'best_fitness': island.best_fitness,
                    'best_generation': island.best_generation,
                    'population_size': island.config.population_size
                }
                for key, island in self.islands.items()
            }
        }
        summary_path = output_path / "summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}")
