"""
Genetic Algorithm Main Orchestrator

Coordinates the complete GA optimization pipeline including population evolution,
fitness evaluation, convergence tracking, checkpointing, and results management.
Provides beautiful tqdm progress display with live metrics.
"""

from typing import Optional, Dict, Any, Tuple, List
import copy
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import yaml

from optimization.ga_config import GAConfig
from optimization.population import Population
from optimization.parallel_evaluator import ParallelEvaluator
from optimization.selection import select_parents
from optimization.convergence_tracker import ConvergenceTracker
from optimization.checkpoint_manager import CheckpointManager
from optimization.results_manager import ResultsManager
from policy.chromosome import Chromosome
from utils.config import load_config_from_yaml


def load_feature_bounds(feature_config_path: str) -> List[Tuple[float, float]]:
    """Extract feature bounds from feature config YAML.

    Args:
        feature_config_path: Path to feature config YAML file

    Returns:
        List of (min, max) tuples for each per-aircraft feature
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


class GAAlgorithm:
    """Main Genetic Algorithm orchestrator.

    Coordinates the complete optimization pipeline with all GA components.
    Provides beautiful progress display and comprehensive result tracking.

    Attributes:
        config: GAConfig with all hyperparameters
        population: Population of chromosomes
        evaluator: ParallelEvaluator for fitness computation
        convergence_tracker: ConvergenceTracker for early stopping
        checkpoint_manager: CheckpointManager for fault tolerance
        results_manager: ResultsManager for output artifacts
        rng: NumPy random number generator
        best_chromosome: Best chromosome found so far
        best_fitness: Best fitness score achieved
        best_generation: Generation where best was found
        best_metrics: Detailed metrics for best chromosome

    Example:
        >>> config = GAConfig.from_yaml('config/ga/default.yaml')
        >>> ga = GAAlgorithm(config)
        >>> best_chromosome, best_fitness = ga.run()
    """

    def __init__(self, config: GAConfig, verbose: bool = True):
        """Initialize GA algorithm with configuration.

        Args:
            config: GAConfig with all hyperparameters
            verbose: Enable progress display (default: True)
        """
        self.config = config
        self.verbose = verbose

        # Initialize RNG
        self.rng = np.random.default_rng(config.seed)

        # Load simulation config
        self.sim_config = load_config_from_yaml(config.simulation_config_path)

        # Load feature bounds from feature config
        self.feature_bounds = load_feature_bounds(config.feature_config_path)

        # Initialize population
        self.population = Population(
            size=config.population_size,
            tree_depth=config.tree_depth,
            n_features=config.n_features,
            seed=config.seed,
            config_type=config.config_type,
            n_fleet_features=config.n_fleet_features,
            feature_bounds=self.feature_bounds
        )

        # Initialize evaluator
        self.evaluator = ParallelEvaluator(
            sim_config=self.sim_config,
            feature_config_path=config.feature_config_path,
            weight_mission_success=config.fitness_weight_mission_success,
            weight_or=config.fitness_weight_or,
            weight_flight_hours=config.fitness_weight_flight_hours,
            baseline_max_flight_hours=config.baseline_max_flight_hours,
            n_episodes=config.episodes_per_chromosome,
            n_workers=config.parallel_workers,
            seed=config.seed,
            verbose=False  # Disable evaluator verbosity for clean progress bar
        )

        # Initialize convergence tracker
        self.convergence_tracker = ConvergenceTracker(
            patience=config.early_stopping_patience,
            min_delta=config.improvement_threshold
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.checkpoint_save_dir,
            filename_pattern=config.checkpoint_filename_pattern,
            keep_last_n=3,  # Keep last 3 checkpoints
            enabled=config.checkpointing_enabled
        )

        # Initialize results manager
        self.results_manager = ResultsManager(
            results_dir=config.results_dir,
            run_name=config.name,
            enabled=True
        )

        # Track best chromosome
        self.best_chromosome: Optional[Chromosome] = None
        self.best_fitness: float = -np.inf
        self.best_generation: int = 0
        self.best_metrics: Dict[str, Any] = {}

        # Current generation
        self.current_generation: int = 0

    def run(self) -> Tuple[Chromosome, float]:
        """Run complete GA optimization.

        Returns:
            Tuple of (best_chromosome, best_fitness)

        Example:
            >>> ga = GAAlgorithm(config)
            >>> best_chrom, best_fit = ga.run()
            >>> print(f"Best fitness: {best_fit:.4f}")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Genetic Algorithm Optimization - {self.config.name}")
            print(f"{'='*70}")
            print(f"Population: {self.config.population_size} | "
                  f"Elite: {self.config.elite_count} | "
                  f"Generations: {self.config.max_generations}")
            print(f"Config Type: {self.config.config_type} | "
                  f"Episodes/Chromosome: {self.config.episodes_per_chromosome} | "
                  f"Workers: {self.config.parallel_workers}")
            print(f"{'='*70}\n")

        # Initialize population
        if self.verbose:
            print("Initializing random population...")
        self.population.initialize_random()

        # Create tqdm progress bar (include {postfix} to show metrics)
        pbar = tqdm(
            total=self.config.max_generations,
            desc="GA Optimization",
            disable=not self.verbose,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )

        # Main evolution loop
        for generation in range(self.config.max_generations):
            self.current_generation = generation

            # Evaluate population
            fitness_scores, all_metrics = self._evaluate_population(generation)

            # Update population fitness
            self.population.fitness_scores = fitness_scores

            # Track best chromosome
            self._update_best_chromosome(generation, fitness_scores, all_metrics)

            # Compute population statistics
            mean_fitness = np.mean(fitness_scores)
            diversity = self.population.compute_diversity()

            # Update convergence tracker
            should_stop = self.convergence_tracker.update(
                generation=generation,
                best_fitness=self.best_fitness,
                mean_fitness=mean_fitness,
                diversity=diversity
            )

            # Update progress bar with enhanced metrics
            self._update_progress_display(pbar, generation)

            # Save checkpoint if needed
            if (generation + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(generation)

            # Check early stopping
            if should_stop:
                if self.verbose:
                    pbar.write(f"\n⚠️  Early stopping triggered at generation {generation}")
                    pbar.write(f"    No improvement for {self.config.early_stopping_patience} generations\n")
                break

            # Evolve next generation (if not last generation)
            if generation < self.config.max_generations - 1:
                self._evolve_generation()

            # Increment progress bar
            pbar.update(1)

        pbar.close()

        # Finalize and save results
        self._finalize_results()

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Optimization Complete!")
            print(f"{'='*70}")
            print(f"Best Fitness: {self.best_fitness:.4f} (Generation {self.best_generation})")
            if self.best_metrics:
                print(f"  Mission Success: {self.best_metrics['mean_mission_success']*100:.1f}%")
                print(f"  Operational Readiness: {self.best_metrics['mean_or']*100:.1f}%")
                print(f"  Flight Hours: {self.best_metrics['mean_flight_hours']:.1f}")
            print(f"{'='*70}\n")

        return self.best_chromosome, self.best_fitness

    def _evaluate_population(self, generation: int) -> Tuple[list, list]:
        """Evaluate entire population in parallel.

        Args:
            generation: Current generation number

        Returns:
            Tuple of (fitness_scores, all_metrics)
        """
        # Progress callback for parallel evaluation
        def progress_callback(completed, total):
            pass  # Silent during parallel eval to keep progress bar clean

        # Evaluate in parallel
        fitness_scores, all_metrics = self.evaluator.evaluate_population(
            population=self.population.chromosomes,
            progress_callback=progress_callback
        )

        return fitness_scores, all_metrics

    def _update_best_chromosome(
        self,
        generation: int,
        fitness_scores: list,
        all_metrics: list
    ):
        """Update best chromosome if new best found.

        Args:
            generation: Current generation
            fitness_scores: Fitness scores for current population
            all_metrics: Detailed metrics for all chromosomes
        """
        # Find best in current generation
        best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[best_idx]

        # Update global best if improved
        if gen_best_fitness > self.best_fitness:
            self.best_fitness = gen_best_fitness
            self.best_chromosome = copy.deepcopy(self.population.chromosomes[best_idx])
            self.best_generation = generation
            self.best_metrics = all_metrics[best_idx].copy()

    def _update_progress_display(self, pbar: tqdm, generation: int):
        """Update tqdm progress bar with enhanced metrics display.

        Shows format in postfix:
        Gen 34 | Fit: 0.6834 | MS: 67.3% | OR: 58.2% | FH: 3892

        Also prints detailed summary every 10 generations.

        Args:
            pbar: tqdm progress bar
            generation: Current generation
        """
        if not self.best_metrics:
            return

        # Extract metrics
        ms = self.best_metrics.get('mean_mission_success', 0.0)
        or_val = self.best_metrics.get('mean_or', 0.0)
        fh = self.best_metrics.get('mean_flight_hours', 0.0)

        # Format compact display string for progress bar postfix
        display = (
            f"Gen {self.best_generation} | "
            f"Fit: {self.best_fitness:.4f} | "
            f"MS: {ms*100:.1f}% | "
            f"OR: {or_val*100:.1f}% | "
            f"FH: {fh:.0f}"
        )

        # Write to progress bar postfix
        pbar.set_postfix_str(display)

        # Print detailed summary every 25 generations (or first generation)
        if generation == 0 or (generation + 1) % 25 == 0:
            pbar.write(f"\n{'─'*70}")
            pbar.write(f"  Generation {generation} Summary (Best found at Gen {self.best_generation})")
            pbar.write(f"{'─'*70}")
            pbar.write(f"  Fitness:              {self.best_fitness:.4f}")
            pbar.write(f"  Mission Success:      {ms*100:.2f}%")
            pbar.write(f"  Operational Readiness: {or_val*100:.2f}%")
            pbar.write(f"  Flight Hours:         {fh:.1f}")
            pbar.write(f"{'─'*70}\n")

    def _save_checkpoint(self, generation: int):
        """Save checkpoint if enabled.

        Args:
            generation: Current generation number
        """
        if not self.config.checkpointing_enabled:
            return

        self.checkpoint_manager.save_checkpoint(
            generation=generation,
            population=self.population,
            fitness_scores=self.population.fitness_scores,
            rng_state=self.rng.bit_generator.state,
            convergence_history=self.convergence_tracker.get_history(),
            best_chromosome=self.best_chromosome,
            metadata={
                'best_fitness': self.best_fitness,
                'best_generation': self.best_generation,
                'config_name': self.config.name
            }
        )

    def _get_adaptive_mutation_params(self, generation: int) -> Tuple[float, float]:
        """Calculate adaptive mutation rate and sigma with exponential decay.

        Both rate and sigma decay from start values to min values, reaching
        approximately the minimum at mutation_decay_point fraction of max_generations.

        Args:
            generation: Current generation number

        Returns:
            Tuple of (current_rate, current_sigma)
        """
        # Exponential decay: k chosen so value ≈ min at decay_point (75% of max_gen by default)
        k = 6.13 / self.config.max_generations

        # Decay rate: 0.50 → 0.05 (default)
        rate = self.config.mutation_min_rate + \
               (self.config.mutation_start_rate - self.config.mutation_min_rate) * \
               np.exp(-k * generation)

        # Decay sigma: 0.50 → 0.10 (default)
        sigma = self.config.mutation_min_sigma + \
                (self.config.mutation_start_sigma - self.config.mutation_min_sigma) * \
                np.exp(-k * generation)

        return max(rate, self.config.mutation_min_rate), max(sigma, self.config.mutation_min_sigma)

    def _evolve_generation(self):
        """Evolve population to create next generation.

        Applies elitism, selection, crossover, and mutation.
        """
        # Get elite individuals (extract chromosomes from (chrom, fitness) tuples)
        elite_pairs = self.population.get_top_k(k=self.config.elite_count)
        elite_chromosomes = [chrom for chrom, fitness in elite_pairs]

        # Create offspring
        offspring = []
        n_offspring_needed = self.config.population_size - self.config.elite_count

        while len(offspring) < n_offspring_needed:
            # Select parents (returns chromosomes directly, not indices)
            parent1, parent2 = select_parents(
                population=self.population.chromosomes,
                fitness_scores=self.population.fitness_scores,
                n_parents=2,
                method=self.config.selection_method,
                tournament_size=self.config.tournament_size,
                rng=self.rng
            )

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child1, child2 = parent1.crossover(
                    other=parent2,
                    rng=self.rng
                )
            else:
                # No crossover - return clones
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Adaptive mutation - rate and sigma decay over generations
            current_rate, current_sigma = self._get_adaptive_mutation_params(self.current_generation)
            child1 = child1.mutate(
                mutation_rate=current_rate,
                mutation_sigma=current_sigma,
                rng=self.rng
            )
            child2 = child2.mutate(
                mutation_rate=current_rate,
                mutation_sigma=current_sigma,
                rng=self.rng
            )

            # Add to offspring
            offspring.append(child1)
            if len(offspring) < n_offspring_needed:
                offspring.append(child2)

        # Combine elite + offspring to form new generation
        new_population = elite_chromosomes + offspring[:n_offspring_needed]

        # Replace population
        self.population.replace(new_population)

    def _finalize_results(self):
        """Save all final results using ResultsManager."""
        if not self.config.save_best_chromosome and not self.config.save_final_population:
            return

        # Prepare best chromosome metadata
        best_metadata = {
            'fitness': self.best_fitness,
            'mean_mission_success': self.best_metrics.get('mean_mission_success', 0.0),
            'mean_or': self.best_metrics.get('mean_or', 0.0),
            'mean_flight_hours': self.best_metrics.get('mean_flight_hours', 0.0),
            'std_mission_success': self.best_metrics.get('std_mission_success', 0.0),
            'std_or': self.best_metrics.get('std_or', 0.0),
            'std_flight_hours': self.best_metrics.get('std_flight_hours', 0.0)
        }

        # Get convergence summary
        convergence_summary = self.convergence_tracker.get_improvement_summary()

        # Save all results
        self.results_manager.save_all_results(
            best_chromosome=self.best_chromosome,
            best_fitness=self.best_fitness,
            best_generation=self.best_generation,
            population=self.population,
            fitness_scores=self.population.fitness_scores,
            final_generation=self.current_generation,
            convergence_history=self.convergence_tracker.get_history(),
            convergence_summary=convergence_summary,
            best_chromosome_metadata=best_metadata,
            config=self.config._raw_data,
            generate_plots=self.config.generate_plots
        )

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume optimization from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file to resume from

        Example:
            >>> ga = GAAlgorithm(config)
            >>> ga.resume_from_checkpoint("results/checkpoints/checkpoint_gen020.pkl")
            >>> ga.run()
        """
        # Load checkpoint
        state = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Restore population
        self.population = state['population']

        # Restore RNG state
        self.rng.bit_generator.state = state['rng_state']

        # Restore convergence history
        for gen, best, mean, div in state['convergence_history']:
            self.convergence_tracker.update(gen, best, mean, div)

        # Restore best chromosome
        if state['best_chromosome'] is not None:
            self.best_chromosome = state['best_chromosome']
            self.best_fitness = state['metadata'].get('best_fitness', -np.inf)
            self.best_generation = state['metadata'].get('best_generation', 0)

        # Set starting generation
        self.current_generation = state['generation'] + 1

        if self.verbose:
            print(f"\n✅ Resumed from checkpoint: generation {state['generation']}")
            print(f"   Best fitness so far: {self.best_fitness:.4f}\n")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"GAAlgorithm(config='{self.config.name}', "
            f"population_size={self.config.population_size}, "
            f"max_generations={self.config.max_generations}, "
            f"best_fitness={self.best_fitness:.4f})"
        )
