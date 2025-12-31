#!/usr/bin/env python3
"""
Genetic Algorithm Optimization Runner

Command-line interface for running GA optimization of decision tree policies.
Similar to pipeline.R but for GA optimization.

Usage:
    python scripts/run_ga.py --config config/ga/default.yaml
    python scripts/run_ga.py --config config/ga/smoke_test.yaml --no-confirm
    python scripts/run_ga.py --resume results/ga_checkpoints/checkpoint_gen020.pkl

Examples:
    # Run with default configuration
    python scripts/run_ga.py

    # Run with custom configuration
    python scripts/run_ga.py --config config/ga/quick_test.yaml

    # Override specific parameters
    python scripts/run_ga.py --config config/ga/default.yaml \\
        --population-size 100 --max-generations 100

    # Resume from checkpoint
    python scripts/run_ga.py --resume results/ga_checkpoints/checkpoint_gen020.pkl

    # Skip confirmation (useful for automated runs)
    python scripts/run_ga.py --config config/ga/smoke_test.yaml --no-confirm
"""

import argparse
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.ga_config import GAConfig
from optimization.ga_algorithm import GAAlgorithm


def estimate_runtime(config: GAConfig) -> tuple:
    """Estimate optimization runtime.

    Args:
        config: GAConfig with optimization parameters

    Returns:
        Tuple of (min_minutes, max_minutes, expected_minutes)
    """
    # Rough estimates (adjust based on empirical measurements)
    # Assume ~4-6 seconds per chromosome evaluation (100 episodes)
    seconds_per_chromosome = 5.0  # Conservative estimate

    # Parallel speedup factor
    speedup = min(config.parallel_workers, config.population_size)
    effective_time_per_gen = (config.population_size * seconds_per_chromosome) / speedup

    # Pessimistic: all generations
    max_time = config.max_generations * effective_time_per_gen / 60  # minutes

    # Optimistic: early stopping kicks in
    expected_gens = min(config.max_generations, config.max_generations * 0.6)
    expected_time = expected_gens * effective_time_per_gen / 60

    # Best case: very early stopping
    min_gens = min(config.max_generations, config.early_stopping_patience * 2)
    min_time = min_gens * effective_time_per_gen / 60

    return min_time, max_time, expected_time


def display_config_summary(config: GAConfig):
    """Display configuration summary.

    Args:
        config: GAConfig to display
    """
    print(f"\n{'='*70}")
    print(f"  GA Configuration: {config.name}")
    print(f"{'='*70}")
    print(f"\nPopulation Settings:")
    print(f"  Population Size:      {config.population_size}")
    print(f"  Elite Count:          {config.elite_count} ({config.elite_count/config.population_size*100:.1f}%)")

    print(f"\nGenetic Operators:")
    print(f"  Crossover Rate:       {config.crossover_rate:.2f}")
    print(f"  Mutation Rate:        {config.mutation_rate:.2f}")
    print(f"  Selection Method:     {config.selection_method}")
    if config.selection_method == "tournament":
        print(f"  Tournament Size:      {config.tournament_size}")

    print(f"\nFitness Function:")
    print(f"  Mission Success:      {config.fitness_weight_mission_success:.2f}")
    print(f"  Operational Ready:    {config.fitness_weight_or:.2f}")
    print(f"  Flight Hours:         {config.fitness_weight_flight_hours:.2f}")

    print(f"\nEvaluation:")
    print(f"  Episodes/Chromosome:  {config.episodes_per_chromosome}")
    print(f"  Parallel Workers:     {config.parallel_workers}")

    print(f"\nConvergence:")
    print(f"  Max Generations:      {config.max_generations}")
    print(f"  Early Stop Patience:  {config.early_stopping_patience}")
    print(f"  Improvement Threshold: {config.improvement_threshold}")

    print(f"\nCheckpointing:")
    print(f"  Enabled:              {config.checkpointing_enabled}")
    if config.checkpointing_enabled:
        print(f"  Frequency:            Every {config.checkpoint_frequency} generations")
        print(f"  Save Directory:       {config.checkpoint_save_dir}")

    print(f"\nOutput:")
    print(f"  Results Directory:    {config.results_dir}")
    print(f"  Generate Plots:       {config.generate_plots}")
    print(f"{'='*70}\n")


def display_runtime_estimate(config: GAConfig):
    """Display estimated runtime.

    Args:
        config: GAConfig for runtime estimation
    """
    min_time, max_time, expected_time = estimate_runtime(config)

    print(f"{'='*70}")
    print(f"  Runtime Estimate")
    print(f"{'='*70}")
    print(f"  Best Case (early convergence):  {min_time:.1f} minutes")
    print(f"  Expected (typical):             {expected_time:.1f} minutes")
    print(f"  Worst Case (no early stop):     {max_time:.1f} minutes")

    # Display expected completion time
    expected_completion = datetime.now() + timedelta(minutes=expected_time)
    print(f"\n  Expected completion: {expected_completion.strftime('%I:%M %p')}")
    print(f"{'='*70}\n")


def confirm_run() -> bool:
    """Prompt user to confirm optimization run.

    Returns:
        True if user confirms, False otherwise
    """
    print("⚠️  Ready to start optimization. This may take a while.")
    response = input("   Continue? [y/N]: ").strip().lower()
    print()
    return response in ['y', 'yes']


def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Genetic Algorithm optimization for decision tree policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python scripts/run_ga.py

  # Run with custom configuration
  python scripts/run_ga.py --config config/ga/quick_test.yaml

  # Override parameters
  python scripts/run_ga.py --population-size 100 --max-generations 100

  # Resume from checkpoint
  python scripts/run_ga.py --resume results/ga_checkpoints/checkpoint_gen020.pkl

  # Skip confirmation
  python scripts/run_ga.py --config config/ga/smoke_test.yaml --no-confirm
        """
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/ga/default.yaml',
        help='Path to GA configuration file (default: config/ga/default.yaml)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume from'
    )

    # Parameter overrides
    parser.add_argument(
        '--population-size',
        type=int,
        default=None,
        help='Override population size'
    )

    parser.add_argument(
        '--max-generations',
        type=int,
        default=None,
        help='Override maximum generations'
    )

    parser.add_argument(
        '--episodes-per-chromosome',
        type=int,
        default=None,
        help='Override episodes per chromosome'
    )

    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=None,
        help='Override number of parallel workers'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
    )

    # Execution options
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    return parser.parse_args()


def apply_overrides(config: GAConfig, args):
    """Apply command-line parameter overrides to config.

    Args:
        config: GAConfig to modify
        args: Parsed command-line arguments

    Returns:
        Modified config
    """
    if args.population_size is not None:
        config.population_size = args.population_size

    if args.max_generations is not None:
        config.max_generations = args.max_generations

    if args.episodes_per_chromosome is not None:
        config.episodes_per_chromosome = args.episodes_per_chromosome

    if args.parallel_workers is not None:
        config.parallel_workers = args.parallel_workers

    if args.seed is not None:
        config.seed = args.seed

    # Re-validate after overrides
    config.validate()

    return config


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Display header
    print("\n" + "="*70)
    print("  Genetic Algorithm Optimization - Decision Tree Policy")
    print("="*70)

    try:
        # Load configuration
        if not args.quiet:
            print(f"\nLoading configuration from: {args.config}")

        config = GAConfig.from_yaml(args.config)

        # Apply overrides
        if any([args.population_size, args.max_generations,
                args.episodes_per_chromosome, args.parallel_workers, args.seed]):
            if not args.quiet:
                print("Applying command-line parameter overrides...")
            config = apply_overrides(config, args)

        # Display configuration
        if not args.quiet:
            display_config_summary(config)
            display_runtime_estimate(config)

        # Confirm run (unless --no-confirm)
        if not args.no_confirm:
            if not confirm_run():
                print("❌ Optimization cancelled by user.\n")
                sys.exit(0)

        # Create GA algorithm
        verbose = not args.quiet
        ga = GAAlgorithm(config=config, verbose=verbose)

        # Resume from checkpoint if specified
        if args.resume:
            if not args.quiet:
                print(f"Resuming from checkpoint: {args.resume}\n")
            ga.resume_from_checkpoint(args.resume)

        # Run optimization
        start_time = time.time()
        best_chromosome, best_fitness = ga.run()
        elapsed_time = time.time() - start_time

        # Display final results
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Final Results")
            print(f"{'='*70}")
            print(f"Best Fitness:        {best_fitness:.4f}")
            print(f"Found at Generation: {ga.best_generation}")
            print(f"Total Time:          {elapsed_time/60:.1f} minutes")
            print(f"\nResults saved to:    {config.results_dir}")
            print(f"{'='*70}\n")

        print("✅ Optimization completed successfully!\n")
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please check that the configuration file exists.\n")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("   Please check your configuration parameters.\n")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user.")
        print("   Progress has been saved to checkpoints.\n")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
