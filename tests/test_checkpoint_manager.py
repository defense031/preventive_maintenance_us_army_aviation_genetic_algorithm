#!/usr/bin/env python3
"""
Unit tests for CheckpointManager.

Tests checkpoint saving, loading, cleanup, and GA state preservation including
RNG state for reproducibility.
"""

import pytest
import sys
import tempfile
import shutil
import time
import pickle
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.checkpoint_manager import CheckpointManager
from optimization.population import Population
from policy.chromosome import Chromosome


class TestCheckpointManagerInitialization:
    """Test CheckpointManager initialization and validation."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        manager = CheckpointManager()

        assert manager.save_dir == Path("results/ga_checkpoints")
        assert manager.filename_pattern == "checkpoint_gen{generation:03d}.pkl"
        assert manager.keep_last_n == 3
        assert manager.enabled is True

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                save_dir=tmpdir,
                filename_pattern="ckpt_{generation:04d}.pkl",
                keep_last_n=5,
                enabled=True
            )

            assert manager.save_dir == Path(tmpdir)
            assert manager.filename_pattern == "ckpt_{generation:04d}.pkl"
            assert manager.keep_last_n == 5
            assert manager.enabled is True

    def test_initialization_disabled(self):
        """Test initialization with checkpointing disabled."""
        manager = CheckpointManager(enabled=False)

        assert manager.enabled is False

    def test_invalid_keep_last_n(self):
        """Test that negative keep_last_n raises error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CheckpointManager(keep_last_n=-1)

    def test_invalid_filename_pattern(self):
        """Test that pattern without {generation} raises error."""
        with pytest.raises(ValueError, match="must contain.*generation"):
            CheckpointManager(filename_pattern="checkpoint.pkl")

    def test_directory_created_when_enabled(self):
        """Test that save directory is created if enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "checkpoints" / "nested"
            manager = CheckpointManager(save_dir=str(save_dir), enabled=True)

            assert save_dir.exists()

    def test_repr(self):
        """Test __repr__ output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last_n=5)

            repr_str = repr(manager)

            assert "CheckpointManager" in repr_str
            assert "keep_last_n=5" in repr_str
            assert "enabled" in repr_str


class TestSaveCheckpoint:
    """Test checkpoint saving functionality."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last_n=3)
            yield manager

    @pytest.fixture
    def sample_population(self):
        """Create sample population for testing."""
        rng = np.random.default_rng(42)
        pop = Population(size=5, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()
        return pop

    def test_save_checkpoint_basic(self, temp_manager, sample_population):
        """Test basic checkpoint saving."""
        rng = np.random.default_rng(42)
        fitness_scores = [0.8, 0.7, 0.6, 0.5, 0.4]
        convergence_history = [(0, 0.8, 0.6, 0.5)]

        filepath = temp_manager.save_checkpoint(
            generation=5,
            population=sample_population,
            fitness_scores=fitness_scores,
            rng_state=rng.bit_generator.state,
            convergence_history=convergence_history
        )

        # Check file was created
        assert filepath is not None
        assert filepath.exists()
        assert filepath.name == "checkpoint_gen005.pkl"

    def test_save_checkpoint_with_metadata(self, temp_manager, sample_population):
        """Test saving checkpoint with custom metadata."""
        rng = np.random.default_rng(42)
        metadata = {'config_name': 'test', 'note': 'best so far'}

        filepath = temp_manager.save_checkpoint(
            generation=10,
            population=sample_population,
            fitness_scores=[0.9],
            rng_state=rng.bit_generator.state,
            convergence_history=[],
            metadata=metadata
        )

        # Load and verify metadata
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        assert checkpoint['metadata'] == metadata

    def test_save_checkpoint_with_best_chromosome(self, temp_manager, sample_population):
        """Test saving checkpoint with best chromosome."""
        rng = np.random.default_rng(42)
        best_chrom = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        filepath = temp_manager.save_checkpoint(
            generation=3,
            population=sample_population,
            fitness_scores=[0.8],
            rng_state=rng.bit_generator.state,
            convergence_history=[],
            best_chromosome=best_chrom
        )

        # Load and verify best chromosome
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        assert checkpoint['best_chromosome'] is not None
        assert isinstance(checkpoint['best_chromosome'], Chromosome)

    def test_save_checkpoint_disabled(self, sample_population):
        """Test that save returns None when disabled."""
        manager = CheckpointManager(enabled=False)
        rng = np.random.default_rng(42)

        filepath = manager.save_checkpoint(
            generation=1,
            population=sample_population,
            fitness_scores=[0.5],
            rng_state=rng.bit_generator.state,
            convergence_history=[]
        )

        assert filepath is None

    def test_save_checkpoint_structure(self, temp_manager, sample_population):
        """Test that checkpoint contains all required fields."""
        rng = np.random.default_rng(42)

        filepath = temp_manager.save_checkpoint(
            generation=7,
            population=sample_population,
            fitness_scores=[0.8, 0.7, 0.6],
            rng_state=rng.bit_generator.state,
            convergence_history=[(0, 0.5, 0.4, 0.3)]
        )

        # Load and verify structure
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        # Check all required fields
        assert checkpoint['generation'] == 7
        assert checkpoint['population'] is not None
        assert checkpoint['fitness_scores'] == [0.8, 0.7, 0.6]
        assert checkpoint['rng_state'] is not None
        assert checkpoint['convergence_history'] == [(0, 0.5, 0.4, 0.3)]
        assert 'timestamp' in checkpoint


class TestLoadCheckpoint:
    """Test checkpoint loading functionality."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)
            yield manager

    @pytest.fixture
    def sample_population(self):
        """Create sample population for testing."""
        rng = np.random.default_rng(42)
        pop = Population(size=3, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()
        return pop

    def test_load_checkpoint_basic(self, temp_manager, sample_population):
        """Test loading saved checkpoint."""
        rng = np.random.default_rng(42)
        fitness_scores = [0.8, 0.7, 0.6]
        convergence_history = [(0, 0.8, 0.7, 0.5)]

        # Save checkpoint
        filepath = temp_manager.save_checkpoint(
            generation=5,
            population=sample_population,
            fitness_scores=fitness_scores,
            rng_state=rng.bit_generator.state,
            convergence_history=convergence_history
        )

        # Load checkpoint
        state = temp_manager.load_checkpoint(str(filepath))

        assert state['generation'] == 5
        assert state['fitness_scores'] == fitness_scores
        assert state['convergence_history'] == convergence_history

    def test_load_nonexistent_checkpoint(self, temp_manager):
        """Test loading nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError, match="not found"):
            temp_manager.load_checkpoint("/nonexistent/checkpoint.pkl")

    def test_load_corrupted_checkpoint(self, temp_manager):
        """Test loading corrupted checkpoint raises error."""
        # Create corrupted file
        corrupted_path = temp_manager.save_dir / "corrupted.pkl"
        with open(corrupted_path, 'w') as f:
            f.write("not a pickle file")

        with pytest.raises(IOError, match="Failed to load"):
            temp_manager.load_checkpoint(str(corrupted_path))

    def test_load_incomplete_checkpoint(self, temp_manager):
        """Test loading checkpoint with missing fields raises error."""
        # Create incomplete checkpoint
        incomplete = {'generation': 5}  # Missing required fields
        incomplete_path = temp_manager.save_dir / "incomplete.pkl"

        with open(incomplete_path, 'wb') as f:
            pickle.dump(incomplete, f)

        with pytest.raises(IOError, match="Corrupted checkpoint.*missing keys"):
            temp_manager.load_checkpoint(str(incomplete_path))


class TestSaveLoadRoundtrip:
    """Test save/load roundtrip preserves state correctly."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)
            yield manager

    def test_roundtrip_preserves_population(self, temp_manager):
        """Test that population is preserved through save/load."""
        rng = np.random.default_rng(42)
        pop = Population(size=3, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Save
        filepath = temp_manager.save_checkpoint(
            generation=10,
            population=pop,
            fitness_scores=[0.8, 0.7, 0.6],
            rng_state=rng.bit_generator.state,
            convergence_history=[]
        )

        # Load
        state = temp_manager.load_checkpoint(str(filepath))
        loaded_pop = state['population']

        # Verify population preserved
        assert loaded_pop.size == pop.size
        assert loaded_pop.tree_depth == pop.tree_depth
        assert loaded_pop.n_features == pop.n_features
        assert len(loaded_pop.chromosomes) == len(pop.chromosomes)

    def test_roundtrip_preserves_rng_state(self, temp_manager):
        """Test that RNG state is preserved for reproducibility."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        # Generate some random numbers
        _ = rng1.random(10)

        # Save RNG state
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        filepath = temp_manager.save_checkpoint(
            generation=1,
            population=pop,
            fitness_scores=[0.5, 0.4],
            rng_state=rng1.bit_generator.state,
            convergence_history=[]
        )

        # Load RNG state
        state = temp_manager.load_checkpoint(str(filepath))
        rng2.bit_generator.state = state['rng_state']

        # Generate numbers from both RNGs
        nums1 = rng1.random(5)
        nums2 = rng2.random(5)

        # Should be identical
        np.testing.assert_array_equal(nums1, nums2)

    def test_roundtrip_preserves_convergence_history(self, temp_manager):
        """Test that convergence history is preserved."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        history = [
            (0, 0.5, 0.4, 0.8),
            (1, 0.6, 0.5, 0.7),
            (2, 0.7, 0.6, 0.6)
        ]

        filepath = temp_manager.save_checkpoint(
            generation=2,
            population=pop,
            fitness_scores=[0.7, 0.6],
            rng_state=rng.bit_generator.state,
            convergence_history=history
        )

        state = temp_manager.load_checkpoint(str(filepath))

        assert state['convergence_history'] == history


class TestLatestCheckpoint:
    """Test finding latest checkpoint."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)
            yield manager

    def test_get_latest_empty_directory(self, temp_manager):
        """Test get_latest returns None when no checkpoints exist."""
        latest = temp_manager.get_latest_checkpoint()
        assert latest is None

    def test_get_latest_single_checkpoint(self, temp_manager):
        """Test get_latest with single checkpoint."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        filepath = temp_manager.save_checkpoint(
            generation=5,
            population=pop,
            fitness_scores=[0.5, 0.4],
            rng_state=rng.bit_generator.state,
            convergence_history=[]
        )

        latest = temp_manager.get_latest_checkpoint()
        assert latest == filepath

    def test_get_latest_multiple_checkpoints(self, temp_manager):
        """Test get_latest returns most recent checkpoint."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Save multiple checkpoints with small delays
        for gen in [3, 7, 10]:
            temp_manager.save_checkpoint(
                generation=gen,
                population=pop,
                fitness_scores=[0.5],
                rng_state=rng.bit_generator.state,
                convergence_history=[]
            )
            time.sleep(0.01)  # Ensure different modification times

        latest = temp_manager.get_latest_checkpoint()

        # Should be generation 10 (most recent)
        assert latest.name == "checkpoint_gen010.pkl"


class TestListCheckpoints:
    """Test listing checkpoints."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)
            yield manager

    def test_list_empty_directory(self, temp_manager):
        """Test list returns empty when no checkpoints exist."""
        checkpoints = temp_manager.list_checkpoints()
        assert checkpoints == []

    def test_list_multiple_checkpoints(self, temp_manager):
        """Test listing multiple checkpoints."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Save multiple checkpoints
        for gen in [5, 10, 15]:
            temp_manager.save_checkpoint(
                generation=gen,
                population=pop,
                fitness_scores=[0.5],
                rng_state=rng.bit_generator.state,
                convergence_history=[]
            )
            time.sleep(0.01)

        checkpoints = temp_manager.list_checkpoints()

        # Should have 3 checkpoints
        assert len(checkpoints) == 3

        # Should be sorted newest first
        assert checkpoints[0].name == "checkpoint_gen015.pkl"
        assert checkpoints[-1].name == "checkpoint_gen005.pkl"


class TestCleanupCheckpoints:
    """Test checkpoint cleanup functionality."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last_n=3)
            yield manager

    def test_cleanup_with_keep_all(self):
        """Test that cleanup does nothing when keep_last_n=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last_n=0)
            rng = np.random.default_rng(42)
            pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
            pop.initialize_random()

            # Save 5 checkpoints
            for gen in range(5):
                manager.save_checkpoint(
                    generation=gen,
                    population=pop,
                    fitness_scores=[0.5],
                    rng_state=rng.bit_generator.state,
                    convergence_history=[]
                )

            deleted = manager.cleanup_old_checkpoints()

            assert deleted == 0
            assert len(manager.list_checkpoints()) == 5

    def test_cleanup_keeps_recent(self, temp_manager):
        """Test that cleanup keeps only most recent N checkpoints."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Save 6 checkpoints
        for gen in range(6):
            temp_manager.save_checkpoint(
                generation=gen,
                population=pop,
                fitness_scores=[0.5],
                rng_state=rng.bit_generator.state,
                convergence_history=[]
            )
            time.sleep(0.01)

        # Cleanup should have been called automatically (keep_last_n=3)
        checkpoints = temp_manager.list_checkpoints()

        # Should keep only last 3
        assert len(checkpoints) == 3

        # Check that newest 3 remain
        names = [cp.name for cp in checkpoints]
        assert "checkpoint_gen005.pkl" in names
        assert "checkpoint_gen004.pkl" in names
        assert "checkpoint_gen003.pkl" in names

    def test_cleanup_no_deletion_when_below_limit(self, temp_manager):
        """Test that cleanup doesn't delete when below limit."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Save 2 checkpoints (below keep_last_n=3)
        for gen in range(2):
            temp_manager.save_checkpoint(
                generation=gen,
                population=pop,
                fitness_scores=[0.5],
                rng_state=rng.bit_generator.state,
                convergence_history=[]
            )

        deleted = temp_manager.cleanup_old_checkpoints()

        assert deleted == 0
        assert len(temp_manager.list_checkpoints()) == 2


class TestDeleteAllCheckpoints:
    """Test deleting all checkpoints."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)
            yield manager

    def test_delete_all_empty(self, temp_manager):
        """Test delete_all with no checkpoints."""
        deleted = temp_manager.delete_all_checkpoints()
        assert deleted == 0

    def test_delete_all_multiple(self, temp_manager):
        """Test deleting all checkpoints."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        # Save 5 checkpoints
        for gen in range(5):
            temp_manager.save_checkpoint(
                generation=gen,
                population=pop,
                fitness_scores=[0.5],
                rng_state=rng.bit_generator.state,
                convergence_history=[]
            )

        deleted = temp_manager.delete_all_checkpoints()

        assert deleted == 5
        assert len(temp_manager.list_checkpoints()) == 0


class TestCheckpointInfo:
    """Test getting checkpoint metadata without full load."""

    @pytest.fixture
    def temp_manager(self):
        """Create temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)
            yield manager

    def test_get_checkpoint_info_basic(self, temp_manager):
        """Test getting checkpoint info."""
        rng = np.random.default_rng(42)
        pop = Population(size=5, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()

        filepath = temp_manager.save_checkpoint(
            generation=10,
            population=pop,
            fitness_scores=[0.8, 0.7, 0.6, 0.5, 0.4],
            rng_state=rng.bit_generator.state,
            convergence_history=[],
            metadata={'note': 'test'}
        )

        info = temp_manager.get_checkpoint_info(str(filepath))

        assert info['generation'] == 10
        assert info['population_size'] == 5
        assert info['has_best_chromosome'] is False
        assert info['metadata'] == {'note': 'test'}
        assert 'file_size_mb' in info
        assert 'timestamp' in info

    def test_get_checkpoint_info_with_best(self, temp_manager):
        """Test info includes best chromosome flag."""
        rng = np.random.default_rng(42)
        pop = Population(size=2, tree_depth=3, n_features=5, seed=42)
        pop.initialize_random()
        best = Chromosome.random(tree_depth=3, n_features=5, rng=rng)

        filepath = temp_manager.save_checkpoint(
            generation=5,
            population=pop,
            fitness_scores=[0.8, 0.7],
            rng_state=rng.bit_generator.state,
            convergence_history=[],
            best_chromosome=best
        )

        info = temp_manager.get_checkpoint_info(str(filepath))

        assert info['has_best_chromosome'] is True

    def test_get_checkpoint_info_nonexistent(self, temp_manager):
        """Test getting info for nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            temp_manager.get_checkpoint_info("/nonexistent.pkl")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
