#!/usr/bin/env python3
"""
Unit tests for ResultsManager.

Tests results saving, plot generation, and report creation for GA optimization.
"""

import pytest
import sys
import tempfile
import json
import pickle
import csv
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.results_manager import ResultsManager
from optimization.population import Population
from policy.chromosome import Chromosome


class TestResultsManagerInitialization:
    """Test ResultsManager initialization."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir)

            assert manager.results_dir == Path(tmpdir)
            assert manager.enabled is True
            assert manager.run_name is not None

    def test_initialization_custom_run_name(self):
        """Test initialization with custom run name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir, run_name="test_run")

            assert manager.run_name == "test_run"

    def test_initialization_disabled(self):
        """Test initialization with results disabled."""
        manager = ResultsManager(enabled=False)

        assert manager.enabled is False

    def test_directory_structure_created(self):
        """Test that directory structure is created when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir, enabled=True)

            assert (manager.results_dir / "chromosomes").exists()
            assert (manager.results_dir / "plots").exists()
            assert (manager.results_dir / "data").exists()

    def test_repr(self):
        """Test __repr__ output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir, run_name="test")

            repr_str = repr(manager)

            assert "ResultsManager" in repr_str
            assert "test" in repr_str
            assert "enabled" in repr_str


class TestSaveBestChromosome:
    """Test saving best chromosome."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir, run_name="test")
            yield manager

    @pytest.fixture
    def sample_chromosome(self):
        """Create sample chromosome."""
        rng = np.random.default_rng(42)
        return Chromosome.random(tree_depth=3, n_features=5, rng=rng)

    def test_save_best_chromosome_basic(self, temp_manager, sample_chromosome):
        """Test saving best chromosome creates both JSON and pickle."""
        paths = temp_manager.save_best_chromosome(
            chromosome=sample_chromosome,
            fitness=0.85,
            generation=25
        )

        # Check both files created
        assert paths is not None
        assert 'json' in paths
        assert 'pickle' in paths
        assert paths['json'].exists()
        assert paths['pickle'].exists()

    def test_save_best_chromosome_json_structure(self, temp_manager, sample_chromosome):
        """Test that JSON contains correct structure."""
        paths = temp_manager.save_best_chromosome(
            chromosome=sample_chromosome,
            fitness=0.85,
            generation=25,
            metadata={'mission_success': 0.87, 'or': 0.62}
        )

        # Load and verify JSON
        with open(paths['json'], 'r') as f:
            data = json.load(f)

        assert data['fitness'] == 0.85
        assert data['generation'] == 25
        assert data['run_name'] == "test"
        assert 'chromosome' in data
        assert data['metadata']['mission_success'] == 0.87

    def test_save_best_chromosome_pickle_loadable(self, temp_manager, sample_chromosome):
        """Test that pickle can be loaded back."""
        paths = temp_manager.save_best_chromosome(
            chromosome=sample_chromosome,
            fitness=0.85,
            generation=25
        )

        # Load pickle
        with open(paths['pickle'], 'rb') as f:
            loaded_chrom = pickle.load(f)

        # Verify it's a valid Chromosome
        assert isinstance(loaded_chrom, Chromosome)
        assert loaded_chrom.tree_depth == sample_chromosome.tree_depth
        assert loaded_chrom.n_features == sample_chromosome.n_features

    def test_save_best_chromosome_disabled(self, sample_chromosome):
        """Test that save returns None when disabled."""
        manager = ResultsManager(enabled=False)

        paths = manager.save_best_chromosome(
            chromosome=sample_chromosome,
            fitness=0.85,
            generation=25
        )

        assert paths is None


class TestSavePopulation:
    """Test saving population."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir)
            yield manager

    @pytest.fixture
    def sample_population(self):
        """Create sample population."""
        rng = np.random.default_rng(42)
        pop = Population(size=5, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()
        return pop

    def test_save_population_basic(self, temp_manager, sample_population):
        """Test saving population."""
        filepath = temp_manager.save_population(
            population=sample_population,
            fitness_scores=[0.8, 0.7, 0.6, 0.5, 0.4],
            generation=50
        )

        assert filepath is not None
        assert filepath.exists()
        assert filepath.name == "final_population.pkl"

    def test_save_population_loadable(self, temp_manager, sample_population):
        """Test that saved population can be loaded."""
        filepath = temp_manager.save_population(
            population=sample_population,
            fitness_scores=[0.8, 0.7, 0.6, 0.5, 0.4],
            generation=50
        )

        # Load and verify
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        assert data['generation'] == 50
        assert len(data['fitness_scores']) == 5
        assert isinstance(data['population'], Population)

    def test_save_population_disabled(self, sample_population):
        """Test that save returns None when disabled."""
        manager = ResultsManager(enabled=False)

        filepath = manager.save_population(
            population=sample_population,
            fitness_scores=[0.5],
            generation=10
        )

        assert filepath is None


class TestSaveFitnessHistory:
    """Test saving fitness history CSV."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir)
            yield manager

    def test_save_fitness_history_basic(self, temp_manager):
        """Test saving fitness history."""
        history = [
            (0, 0.5, 0.4, 0.8),
            (1, 0.6, 0.5, 0.7),
            (2, 0.7, 0.6, 0.6)
        ]

        filepath = temp_manager.save_fitness_history(history)

        assert filepath is not None
        assert filepath.exists()
        assert filepath.name == "fitness_history.csv"

    def test_save_fitness_history_csv_structure(self, temp_manager):
        """Test CSV structure and content."""
        history = [
            (0, 0.5, 0.4, 0.8),
            (1, 0.6, 0.5, 0.7)
        ]

        filepath = temp_manager.save_fitness_history(history)

        # Read and verify CSV
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header
        assert rows[0] == ['generation', 'best_fitness', 'mean_fitness', 'diversity']

        # Check data
        assert rows[1][0] == '0'
        assert float(rows[1][1]) == pytest.approx(0.5)
        assert float(rows[1][2]) == pytest.approx(0.4)
        assert float(rows[1][3]) == pytest.approx(0.8)

    def test_save_fitness_history_empty(self, temp_manager):
        """Test saving empty history."""
        filepath = temp_manager.save_fitness_history([])

        assert filepath is not None
        assert filepath.exists()

        # Should have header only
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 1  # Header only

    def test_save_fitness_history_disabled(self):
        """Test that save returns None when disabled."""
        manager = ResultsManager(enabled=False)

        filepath = manager.save_fitness_history([(0, 0.5, 0.4, 0.8)])

        assert filepath is None


class TestGenerateConvergencePlot:
    """Test convergence plot generation."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir, run_name="test_run")
            yield manager

    def test_generate_convergence_plot_basic(self, temp_manager):
        """Test generating convergence plot."""
        history = [
            (0, 0.5, 0.4, 0.8),
            (1, 0.6, 0.5, 0.7),
            (2, 0.7, 0.6, 0.6),
            (3, 0.75, 0.65, 0.5)
        ]

        filepath = temp_manager.generate_convergence_plot(history)

        assert filepath is not None
        assert filepath.exists()
        assert filepath.suffix == '.png'
        assert filepath.name == "convergence_plot.png"

    def test_generate_convergence_plot_custom_name(self, temp_manager):
        """Test generating plot with custom filename."""
        history = [(0, 0.5, 0.4, 0.8), (1, 0.6, 0.5, 0.7)]

        filepath = temp_manager.generate_convergence_plot(
            history,
            save_name="custom_plot.png"
        )

        assert filepath.name == "custom_plot.png"

    def test_generate_convergence_plot_empty_history(self, temp_manager):
        """Test that empty history returns None."""
        filepath = temp_manager.generate_convergence_plot([])

        assert filepath is None

    def test_generate_convergence_plot_disabled(self):
        """Test that plot generation returns None when disabled."""
        manager = ResultsManager(enabled=False)

        filepath = manager.generate_convergence_plot([(0, 0.5, 0.4, 0.8)])

        assert filepath is None


class TestGenerateDiversityPlot:
    """Test diversity plot generation."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir)
            yield manager

    def test_generate_diversity_plot_basic(self, temp_manager):
        """Test generating diversity plot."""
        history = [
            (0, 0.5, 0.4, 0.8),
            (1, 0.6, 0.5, 0.7),
            (2, 0.7, 0.6, 0.6)
        ]

        filepath = temp_manager.generate_diversity_plot(history)

        assert filepath is not None
        assert filepath.exists()
        assert filepath.name == "diversity_plot.png"

    def test_generate_diversity_plot_empty(self, temp_manager):
        """Test that empty history returns None."""
        filepath = temp_manager.generate_diversity_plot([])

        assert filepath is None


class TestGenerateCombinedPlot:
    """Test combined plot generation."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir)
            yield manager

    def test_generate_combined_plot_basic(self, temp_manager):
        """Test generating combined plot."""
        history = [
            (0, 0.5, 0.4, 0.8),
            (1, 0.6, 0.5, 0.7),
            (2, 0.7, 0.6, 0.6)
        ]

        filepath = temp_manager.generate_combined_plot(history)

        assert filepath is not None
        assert filepath.exists()
        assert filepath.name == "combined_plot.png"

    def test_generate_combined_plot_empty(self, temp_manager):
        """Test that empty history returns None."""
        filepath = temp_manager.generate_combined_plot([])

        assert filepath is None


class TestGenerateSummaryReport:
    """Test summary report generation."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir, run_name="test_run")
            yield manager

    def test_generate_summary_report_basic(self, temp_manager):
        """Test generating summary report."""
        filepath = temp_manager.generate_summary_report(
            best_fitness=0.85,
            best_generation=25,
            total_generations=50
        )

        assert filepath is not None
        assert filepath.exists()
        assert filepath.name == "summary_report.json"

    def test_generate_summary_report_structure(self, temp_manager):
        """Test summary report structure."""
        filepath = temp_manager.generate_summary_report(
            best_fitness=0.85,
            best_generation=25,
            total_generations=50,
            best_chromosome_metadata={'mission_success': 0.87},
            convergence_summary={'improvement_from_start': 0.35}
        )

        # Load and verify
        with open(filepath, 'r') as f:
            summary = json.load(f)

        assert summary['run_name'] == "test_run"
        assert summary['optimization_summary']['best_fitness'] == 0.85
        assert summary['optimization_summary']['best_generation'] == 25
        assert summary['optimization_summary']['total_generations'] == 50
        assert summary['best_chromosome']['mission_success'] == 0.87
        assert 'timestamp' in summary

    def test_generate_summary_report_with_config(self, temp_manager):
        """Test summary includes configuration."""
        config = {'population_size': 50, 'crossover_rate': 0.70}

        filepath = temp_manager.generate_summary_report(
            best_fitness=0.85,
            best_generation=25,
            total_generations=50,
            config=config
        )

        with open(filepath, 'r') as f:
            summary = json.load(f)

        assert summary['configuration'] == config

    def test_generate_summary_report_disabled(self):
        """Test that report generation returns None when disabled."""
        manager = ResultsManager(enabled=False)

        filepath = manager.generate_summary_report(
            best_fitness=0.85,
            best_generation=25,
            total_generations=50
        )

        assert filepath is None


class TestSaveAllResults:
    """Test save_all_results convenience method."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir, run_name="test_all")
            yield manager

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        rng = np.random.default_rng(42)

        # Create population
        pop = Population(size=3, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Create best chromosome
        best_chrom = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        # Create history
        history = [
            (0, 0.5, 0.4, 0.8),
            (1, 0.6, 0.5, 0.7),
            (2, 0.7, 0.6, 0.6)
        ]

        return {
            'population': pop,
            'best_chromosome': best_chrom,
            'history': history
        }

    def test_save_all_results_basic(self, temp_manager, sample_data):
        """Test saving all results at once."""
        paths = temp_manager.save_all_results(
            best_chromosome=sample_data['best_chromosome'],
            best_fitness=0.85,
            best_generation=2,
            population=sample_data['population'],
            fitness_scores=[0.7, 0.6, 0.5],
            final_generation=2,
            convergence_history=sample_data['history'],
            convergence_summary={'improvement_from_start': 0.2}
        )

        # Check all expected outputs
        assert 'json' in paths
        assert 'pickle' in paths
        assert 'population' in paths
        assert 'fitness_history' in paths
        assert 'convergence_plot' in paths
        assert 'diversity_plot' in paths
        assert 'combined_plot' in paths
        assert 'summary_report' in paths

        # Verify files exist
        assert paths['json'].exists()
        assert paths['pickle'].exists()
        assert paths['population'].exists()
        assert paths['fitness_history'].exists()

    def test_save_all_results_no_plots(self, temp_manager, sample_data):
        """Test saving results without plots."""
        paths = temp_manager.save_all_results(
            best_chromosome=sample_data['best_chromosome'],
            best_fitness=0.85,
            best_generation=2,
            population=sample_data['population'],
            fitness_scores=[0.7],
            final_generation=2,
            convergence_history=sample_data['history'],
            convergence_summary={},
            generate_plots=False
        )

        # Should not have plot entries
        assert 'convergence_plot' not in paths or paths['convergence_plot'] is None
        assert 'diversity_plot' not in paths or paths['diversity_plot'] is None

    def test_save_all_results_disabled(self, sample_data):
        """Test that save_all_results returns empty dict when disabled."""
        manager = ResultsManager(enabled=False)

        paths = manager.save_all_results(
            best_chromosome=sample_data['best_chromosome'],
            best_fitness=0.85,
            best_generation=2,
            population=sample_data['population'],
            fitness_scores=[0.7],
            final_generation=2,
            convergence_history=sample_data['history'],
            convergence_summary={}
        )

        assert paths == {}


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary results manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(results_dir=tmpdir)
            yield manager

    def test_large_history(self, temp_manager):
        """Test handling large convergence history."""
        # Generate large history
        history = [(i, 0.5 + i*0.01, 0.4 + i*0.005, 0.8 - i*0.01) for i in range(100)]

        filepath = temp_manager.save_fitness_history(history)

        assert filepath is not None
        assert filepath.exists()

        # Verify all rows saved
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 101  # 100 data + 1 header

    def test_single_generation_history(self, temp_manager):
        """Test with single generation."""
        history = [(0, 0.5, 0.4, 0.8)]

        filepath = temp_manager.generate_convergence_plot(history)

        assert filepath is not None
        assert filepath.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
