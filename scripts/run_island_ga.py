#!/usr/bin/env python3
"""
Run Island Model Genetic Algorithm for Decision Tree Policy Optimization.

This script runs a heterogeneous island model with three islands:
- Oahu (Laboratory): Small, high exploration, receives proven winners
- Maui (Refiner): Combines and polishes promising variations
- Big Island (Brawl): Intense competition, only the fittest survive

Migration is asymmetric:
- Big Island -> Oahu: 4 migrants (proven winners to lab)
- Oahu -> Maui: 2 migrants (explorations to refine)
- Maui -> Big Island: 2 migrants (polished solutions to compete)

Usage:
    python scripts/run_island_ga.py --config config/ga/medium_500.yaml
    python scripts/run_island_ga.py --config experiments/ga_configs/exp_simple_very_high_ms80.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.ga_config import GAConfig
from optimization.island_ga import IslandGA, ISLAND_CONFIGS, ISLAND_CONFIGS_FULL, MIGRATION_MAP


def print_header():
    """Print startup header."""
    print("\n" + "=" * 70)
    print("  Island Model Genetic Algorithm - Decision Tree Policy")
    print("=" * 70)


def print_config_summary(config: GAConfig, migration_freq: int):
    """Print configuration summary."""
    print(f"\nConfiguration: {config.name}")
    print("-" * 50)

    # Select island configs based on config type (same logic as IslandGA)
    island_configs = ISLAND_CONFIGS_FULL if config.config_type == "full" else ISLAND_CONFIGS

    # Island info
    total_pop = sum(ic.population_size for ic in island_configs.values())
    print(f"\nIsland Model:")
    for key, ic in island_configs.items():
        print(f"  {ic.name:12s}: pop={ic.population_size:2d}, "
              f"elite={ic.elite_count}, "
              f"tourn={ic.tournament_size}, "
              f"cx={ic.crossover_rate:.0%}")
    print(f"  Total Population: {total_pop}")

    # Asymmetric migration
    print(f"\nMigration (every {migration_freq} generations):")
    for source, (dest, n) in MIGRATION_MAP.items():
        src_name = island_configs[source].name
        dest_name = island_configs[dest].name
        print(f"  {src_name:12s} -> {dest_name:12s}: {n} migrants")

    # Fitness weights
    print(f"\nFitness Weights:")
    print(f"  Mission Success:       {config.fitness_weight_mission_success:.0%}")
    print(f"  Operational Readiness: {config.fitness_weight_or:.0%}")
    print(f"  Flight Hours:          {config.fitness_weight_flight_hours:.0%}")

    # Evaluation
    print(f"\nEvaluation:")
    print(f"  Episodes/Chromosome:  {config.episodes_per_chromosome}")
    print(f"  Parallel Workers:     {config.parallel_workers}")

    # Convergence
    print(f"\nConvergence:")
    print(f"  Max Generations:      {config.max_generations}")
    print(f"  Early Stop Patience:  {config.early_stopping_patience}")
    print(f"  Improvement Threshold: {config.improvement_threshold}")

    # Chromosome
    print(f"\nChromosome:")
    print(f"  Config Type:          {config.config_type}")
    print(f"  Tree Depth:           {config.tree_depth}")
    print(f"  N Features:           {config.n_features}")


def main():
    parser = argparse.ArgumentParser(description="Run Island Model GA")
    parser.add_argument("--config", required=True, help="Path to GA config YAML")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--migration-freq", type=int, default=40, help="Migration frequency (generations)")
    parser.add_argument("--parallel-workers", type=int, default=None, help="Override parallel workers from config")
    args = parser.parse_args()

    print_header()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"\nConfig not found: {config_path}")
        sys.exit(1)

    print(f"\nLoading configuration from: {config_path}")
    config = GAConfig.from_yaml(str(config_path))

    # Override parallel workers if specified
    if args.parallel_workers is not None:
        config.parallel_workers = args.parallel_workers

    print_config_summary(config, args.migration_freq)

    # Output directory
    output_dir = Path(config.results_dir)
    print(f"\nOutput Directory: {output_dir}")

    # Confirmation
    if not args.no_confirm:
        response = input("\nStart Island GA optimization? [Y/n]: ")
        if response.lower() == 'n':
            print("Aborted.")
            sys.exit(0)

    print("\n" + "=" * 70)

    # Create and run Island GA
    island_ga = IslandGA(
        ga_config=config,
        migration_frequency=args.migration_freq
    )

    # Run optimization
    results = island_ga.run()

    # Save results
    island_ga.save_results(str(output_dir))

    # Generate plots automatically
    from optimization.island_plot_generator import generate_island_plots
    print("\nGenerating plots...")
    generate_island_plots(str(output_dir), run_name=config.name)

    # Print final summary
    print("\n" + "=" * 70)
    print("  Optimization Complete!")
    print("=" * 70)
    print(f"\nBest Fitness: {results['best_fitness']:.4f}")
    print(f"Best Island:  {results['best_island']}")
    print(f"Generation:   {results['best_generation']}")
    print(f"Runtime:      {results['runtime_minutes']:.1f} minutes")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
