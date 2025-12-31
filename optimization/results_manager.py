"""
Results Manager for Genetic Algorithm

Handles saving GA optimization results including best chromosomes, fitness history,
convergence plots, and summary reports. Provides comprehensive output artifacts
for analysis and reproducibility.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import pickle
import csv
from datetime import datetime
import numpy as np

# Matplotlib imports (with Agg backend for headless environments)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class ResultsManager:
    """Manage GA optimization results and output artifacts.

    Handles saving best chromosomes, fitness history, convergence plots,
    diversity plots, and summary reports. Creates comprehensive output
    for analysis and reproducibility.

    Attributes:
        results_dir: Directory to save all results
        run_name: Name/identifier for this optimization run
        enabled: Whether results saving is enabled

    Example:
        >>> manager = ResultsManager(results_dir="results/ga_run1")
        >>> manager.save_best_chromosome(chromosome, fitness=0.85, generation=25)
        >>> manager.save_fitness_history(history)
        >>> manager.generate_convergence_plot(history)
    """

    def __init__(
        self,
        results_dir: str = "results/ga_optimization",
        run_name: Optional[str] = None,
        enabled: bool = True
    ):
        """Initialize results manager.

        Args:
            results_dir: Base directory for saving results
            run_name: Name for this run (default: timestamp-based)
            enabled: Enable results saving (default: True)
        """
        self.results_dir = Path(results_dir)
        self.run_name = run_name or f"ga_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enabled = enabled

        # Create directory structure if enabled
        if self.enabled:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            (self.results_dir / "chromosomes").mkdir(exist_ok=True)
            (self.results_dir / "plots").mkdir(exist_ok=True)
            (self.results_dir / "data").mkdir(exist_ok=True)

    def save_best_chromosome(
        self,
        chromosome: Any,
        fitness: float,
        generation: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Path]]:
        """Save best chromosome in both JSON and pickle formats.

        Args:
            chromosome: Best chromosome to save
            fitness: Fitness score of best chromosome
            generation: Generation when best was found
            metadata: Additional metadata (e.g., MS, OR, FH components)

        Returns:
            Dictionary with 'json' and 'pickle' file paths, or None if disabled

        Example:
            >>> manager.save_best_chromosome(
            ...     chromosome=best_chrom,
            ...     fitness=0.85,
            ...     generation=25,
            ...     metadata={'mission_success': 0.87, 'or': 0.62}
            ... )
        """
        if not self.enabled:
            return None

        # Prepare chromosome data
        chromosome_data = {
            'fitness': fitness,
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'run_name': self.run_name,
            'chromosome': chromosome.to_dict(),
            'metadata': metadata or {}
        }

        # Save as JSON (human-readable)
        json_path = self.results_dir / "chromosomes" / "best_chromosome.json"
        with open(json_path, 'w') as f:
            json.dump(chromosome_data, f, indent=2)

        # Save as pickle (for loading back)
        pickle_path = self.results_dir / "chromosomes" / "best_chromosome.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(chromosome, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {'json': json_path, 'pickle': pickle_path}

    def save_population(
        self,
        population: Any,
        fitness_scores: List[float],
        generation: int
    ) -> Optional[Path]:
        """Save entire final population.

        Args:
            population: Population object with all chromosomes
            fitness_scores: Fitness scores for all chromosomes
            generation: Final generation number

        Returns:
            Path to saved population file, or None if disabled

        Example:
            >>> manager.save_population(
            ...     population=final_pop,
            ...     fitness_scores=[0.8, 0.75, ...],
            ...     generation=50
            ... )
        """
        if not self.enabled:
            return None

        population_data = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'run_name': self.run_name,
            'population': population,
            'fitness_scores': fitness_scores
        }

        filepath = self.results_dir / "chromosomes" / "final_population.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(population_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath

    def save_fitness_history(
        self,
        history: List[Tuple[int, float, float, float]]
    ) -> Optional[Path]:
        """Export fitness history to CSV.

        Args:
            history: List of (generation, best_fitness, mean_fitness, diversity) tuples

        Returns:
            Path to saved CSV file, or None if disabled

        Example:
            >>> history = tracker.get_history()
            >>> manager.save_fitness_history(history)
        """
        if not self.enabled:
            return None

        filepath = self.results_dir / "data" / "fitness_history.csv"

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['generation', 'best_fitness', 'mean_fitness', 'diversity'])

            # Data
            for gen, best, mean, div in history:
                writer.writerow([gen, f"{best:.6f}", f"{mean:.6f}", f"{div:.6f}"])

        return filepath

    def generate_convergence_plot(
        self,
        history: List[Tuple[int, float, float, float]],
        title: Optional[str] = None,
        save_name: str = "convergence_plot.png"
    ) -> Optional[Path]:
        """Generate and save fitness convergence plot.

        Creates a plot showing best and mean fitness over generations.

        Args:
            history: List of (generation, best_fitness, mean_fitness, diversity) tuples
            title: Plot title (default: auto-generated)
            save_name: Filename for saved plot

        Returns:
            Path to saved plot, or None if disabled

        Example:
            >>> manager.generate_convergence_plot(tracker.get_history())
        """
        if not self.enabled or not history:
            return None

        # Extract data
        generations = [h[0] for h in history]
        best_fitness = [h[1] for h in history]
        mean_fitness = [h[2] for h in history]

        # Create plot
        plt.figure(figsize=(10, 6))

        plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
        plt.plot(generations, mean_fitness, 'g--', linewidth=1.5, label='Mean Fitness', alpha=0.7)

        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title(title or f'GA Convergence - {self.run_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Highlight best
        best_gen = max(enumerate(best_fitness), key=lambda x: x[1])[0]
        plt.axvline(x=generations[best_gen], color='r', linestyle=':', alpha=0.5, label=f'Best at Gen {generations[best_gen]}')

        plt.tight_layout()

        # Save
        filepath = self.results_dir / "plots" / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def generate_diversity_plot(
        self,
        history: List[Tuple[int, float, float, float]],
        title: Optional[str] = None,
        save_name: str = "diversity_plot.png"
    ) -> Optional[Path]:
        """Generate and save population diversity plot.

        Creates a plot showing population diversity over generations.

        Args:
            history: List of (generation, best_fitness, mean_fitness, diversity) tuples
            title: Plot title (default: auto-generated)
            save_name: Filename for saved plot

        Returns:
            Path to saved plot, or None if disabled

        Example:
            >>> manager.generate_diversity_plot(tracker.get_history())
        """
        if not self.enabled or not history:
            return None

        # Extract data
        generations = [h[0] for h in history]
        diversity = [h[3] for h in history]

        # Create plot
        plt.figure(figsize=(10, 6))

        plt.plot(generations, diversity, 'purple', linewidth=2, marker='s', markersize=4)
        plt.fill_between(generations, diversity, alpha=0.3, color='purple')

        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Population Diversity', fontsize=12)
        plt.title(title or f'Population Diversity - {self.run_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        plt.tight_layout()

        # Save
        filepath = self.results_dir / "plots" / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def generate_combined_plot(
        self,
        history: List[Tuple[int, float, float, float]],
        title: Optional[str] = None,
        save_name: str = "combined_plot.png"
    ) -> Optional[Path]:
        """Generate combined fitness and diversity plot.

        Creates a subplot with fitness convergence and diversity on separate axes.

        Args:
            history: List of (generation, best_fitness, mean_fitness, diversity) tuples
            title: Overall plot title (default: auto-generated)
            save_name: Filename for saved plot

        Returns:
            Path to saved plot, or None if disabled

        Example:
            >>> manager.generate_combined_plot(tracker.get_history())
        """
        if not self.enabled or not history:
            return None

        # Extract data
        generations = [h[0] for h in history]
        best_fitness = [h[1] for h in history]
        mean_fitness = [h[2] for h in history]
        diversity = [h[3] for h in history]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Fitness plot
        ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
        ax1.plot(generations, mean_fitness, 'g--', linewidth=1.5, label='Mean Fitness', alpha=0.7)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Fitness', fontsize=11)
        ax1.set_title('Fitness Convergence', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Diversity plot
        ax2.plot(generations, diversity, 'purple', linewidth=2, marker='s', markersize=4)
        ax2.fill_between(generations, diversity, alpha=0.3, color='purple')
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Population Diversity', fontsize=11)
        ax2.set_title('Population Diversity', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Overall title
        fig.suptitle(title or f'GA Optimization Summary - {self.run_name}',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save
        filepath = self.results_dir / "plots" / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def generate_summary_report(
        self,
        best_fitness: float,
        best_generation: int,
        total_generations: int,
        best_chromosome_metadata: Optional[Dict[str, Any]] = None,
        convergence_summary: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Generate comprehensive summary report in JSON format.

        Args:
            best_fitness: Best fitness achieved
            best_generation: Generation where best was found
            total_generations: Total generations run
            best_chromosome_metadata: Metadata about best chromosome (MS, OR, FH, etc.)
            convergence_summary: Summary from ConvergenceTracker
            config: GA configuration used

        Returns:
            Path to saved summary JSON, or None if disabled

        Example:
            >>> manager.generate_summary_report(
            ...     best_fitness=0.85,
            ...     best_generation=25,
            ...     total_generations=50,
            ...     best_chromosome_metadata={'mission_success': 0.87},
            ...     convergence_summary=tracker.get_improvement_summary()
            ... )
        """
        if not self.enabled:
            return None

        summary = {
            'run_name': self.run_name,
            'timestamp': datetime.now().isoformat(),
            'optimization_summary': {
                'best_fitness': best_fitness,
                'best_generation': best_generation,
                'total_generations': total_generations,
                'improvement_from_start': convergence_summary.get('improvement_from_start', 0.0) if convergence_summary else 0.0
            },
            'best_chromosome': best_chromosome_metadata or {},
            'convergence': convergence_summary or {},
            'configuration': config or {}
        }

        filepath = self.results_dir / "summary_report.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        return filepath

    def save_all_results(
        self,
        best_chromosome: Any,
        best_fitness: float,
        best_generation: int,
        population: Any,
        fitness_scores: List[float],
        final_generation: int,
        convergence_history: List[Tuple[int, float, float, float]],
        convergence_summary: Dict[str, Any],
        best_chromosome_metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        generate_plots: bool = True
    ) -> Dict[str, Optional[Path]]:
        """Save all results in one call (convenience method).

        Args:
            best_chromosome: Best chromosome found
            best_fitness: Best fitness score
            best_generation: Generation where best was found
            population: Final population
            fitness_scores: Final fitness scores
            final_generation: Final generation number
            convergence_history: Full convergence history
            convergence_summary: Summary from ConvergenceTracker
            best_chromosome_metadata: Metadata for best chromosome
            config: GA configuration
            generate_plots: Whether to generate plots (default: True)

        Returns:
            Dictionary mapping result types to file paths

        Example:
            >>> paths = manager.save_all_results(
            ...     best_chromosome=best,
            ...     best_fitness=0.85,
            ...     best_generation=25,
            ...     population=pop,
            ...     fitness_scores=scores,
            ...     final_generation=50,
            ...     convergence_history=history,
            ...     convergence_summary=summary
            ... )
        """
        if not self.enabled:
            return {}

        results = {}

        # Save best chromosome
        chrom_paths = self.save_best_chromosome(
            chromosome=best_chromosome,
            fitness=best_fitness,
            generation=best_generation,
            metadata=best_chromosome_metadata
        )
        if chrom_paths:
            results.update(chrom_paths)

        # Save final population
        results['population'] = self.save_population(
            population=population,
            fitness_scores=fitness_scores,
            generation=final_generation
        )

        # Save fitness history
        results['fitness_history'] = self.save_fitness_history(convergence_history)

        # Generate plots
        if generate_plots:
            results['convergence_plot'] = self.generate_convergence_plot(convergence_history)
            results['diversity_plot'] = self.generate_diversity_plot(convergence_history)
            results['combined_plot'] = self.generate_combined_plot(convergence_history)

        # Save summary report
        results['summary_report'] = self.generate_summary_report(
            best_fitness=best_fitness,
            best_generation=best_generation,
            total_generations=final_generation + 1,
            best_chromosome_metadata=best_chromosome_metadata,
            convergence_summary=convergence_summary,
            config=config
        )

        return results

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "enabled" if self.enabled else "disabled"
        return (
            f"ResultsManager(results_dir='{self.results_dir}', "
            f"run_name='{self.run_name}', {status})"
        )
