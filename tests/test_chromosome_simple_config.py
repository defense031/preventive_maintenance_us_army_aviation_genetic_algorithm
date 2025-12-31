#!/usr/bin/env python3
"""
Tests for Simple Config Chromosome (15 genes).

Validates the corrected gene count implementation:
- Simple: 15 genes (7 variable indices + 7 thresholds + 1 tiebreak)

Key properties:
- depth=3 → n_splits=7, n_leaves=8
- Direct bucket mapping: leaf 0→1, leaf 1→2, ..., leaf 7→8
- No bucket_assignments genes (removed)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.chromosome import Chromosome


class TestSimpleConfigGeneCount:
    """Verify correct gene counts for simple config."""

    def test_simple_config_has_15_genes(self):
        """Simple config with depth=3 should have exactly 15 genes."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=rng
        )

        assert chrome.n_genes == 15, f"Expected 15 genes, got {chrome.n_genes}"
        assert chrome.n_splits == 7, f"Expected 7 splits, got {chrome.n_splits}"
        assert chrome.n_leaves == 8, f"Expected 8 leaves, got {chrome.n_leaves}"

    def test_simple_config_gene_breakdown(self):
        """Verify gene breakdown: 7 indices + 7 thresholds + 1 tiebreak."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=rng
        )

        assert len(chrome.feature_indices) == 7, f"Expected 7 feature indices, got {len(chrome.feature_indices)}"
        assert len(chrome.thresholds) == 7, f"Expected 7 thresholds, got {len(chrome.thresholds)}"
        assert isinstance(chrome.tiebreak_feature, int), "tiebreak_feature should be int"

        # No bucket_assignments in simple config
        assert not hasattr(chrome, 'bucket_assignments') or chrome.__dict__.get('bucket_assignments') is None


class TestSimpleConfigValidation:
    """Test validation for simple config."""

    def test_valid_chromosome_passes_validation(self):
        """Manually constructed valid chromosome passes validation."""
        chrome = Chromosome(
            feature_indices=[0, 1, 2, 3, 4, 0, 1],  # 7 indices
            thresholds=[100.0, 200.0, 50.0, 150.0, 300.0, 75.0, 125.0],  # 7 thresholds
            tiebreak_feature=2,
            config_type='simple',
            tree_depth=3,
            n_features=5
        )

        # Should not raise
        chrome.validate()
        assert chrome.n_genes == 15

    def test_wrong_feature_indices_length_fails(self):
        """Wrong number of feature indices should fail validation."""
        with pytest.raises(ValueError, match="feature_indices length mismatch"):
            Chromosome(
                feature_indices=[0, 1, 2],  # Wrong: only 3 instead of 7
                thresholds=[100.0, 200.0, 50.0, 150.0, 300.0, 75.0, 125.0],
                tiebreak_feature=2,
                config_type='simple',
                tree_depth=3,
                n_features=5
            )

    def test_wrong_thresholds_length_fails(self):
        """Wrong number of thresholds should fail validation."""
        with pytest.raises(ValueError, match="thresholds length mismatch"):
            Chromosome(
                feature_indices=[0, 1, 2, 3, 4, 0, 1],
                thresholds=[100.0, 200.0, 50.0],  # Wrong: only 3 instead of 7
                tiebreak_feature=2,
                config_type='simple',
                tree_depth=3,
                n_features=5
            )

    def test_invalid_feature_index_fails(self):
        """Feature index out of range should fail validation."""
        with pytest.raises(ValueError, match="Invalid feature index"):
            Chromosome(
                feature_indices=[0, 1, 2, 3, 9, 0, 1],  # 9 is out of range (0-4)
                thresholds=[100.0, 200.0, 50.0, 150.0, 300.0, 75.0, 125.0],
                tiebreak_feature=2,
                config_type='simple',
                tree_depth=3,
                n_features=5
            )

    def test_threshold_out_of_bounds_fails(self):
        """Threshold outside feature bounds should fail validation."""
        with pytest.raises(ValueError, match="out of range"):
            Chromosome(
                feature_indices=[0, 1, 2, 3, 4, 0, 1],
                thresholds=[100.0, 200.0, 50.0, 150.0, 300.0, 75.0, 999.0],  # 999 out of bounds
                tiebreak_feature=2,
                config_type='simple',
                tree_depth=3,
                n_features=5
            )


class TestSimpleConfigBucketMapping:
    """Test direct leaf-to-bucket mapping."""

    def test_leaf_to_bucket_mapping(self):
        """Direct mapping: leaf 0→bucket 1, ..., leaf 7→bucket 8."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=rng
        )

        for leaf_idx in range(8):
            bucket = chrome.get_bucket_from_leaf(leaf_idx)
            expected_bucket = leaf_idx + 1
            assert bucket == expected_bucket, f"Leaf {leaf_idx} should map to bucket {expected_bucket}, got {bucket}"

    def test_invalid_leaf_raises_error(self):
        """Invalid leaf index should raise ValueError."""
        chrome = Chromosome.random(tree_depth=3, n_features=5, config_type='simple')

        with pytest.raises(ValueError, match="Invalid leaf_idx"):
            chrome.get_bucket_from_leaf(-1)

        with pytest.raises(ValueError, match="Invalid leaf_idx"):
            chrome.get_bucket_from_leaf(8)


class TestSimpleConfigNumpyConversion:
    """Test numpy array conversion for simple config."""

    def test_to_numpy_produces_correct_length(self):
        """to_numpy should produce array of length 15."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=rng
        )

        genes = chrome.to_numpy()

        assert len(genes) == 15, f"Expected 15 genes in numpy array, got {len(genes)}"

    def test_to_numpy_gene_layout(self):
        """Verify gene layout: [7 indices, 7 thresholds, 1 tiebreak]."""
        chrome = Chromosome(
            feature_indices=[0, 1, 2, 3, 4, 0, 1],
            thresholds=[100.0, 200.0, 50.0, 150.0, 300.0, 75.0, 125.0],
            tiebreak_feature=2,
            config_type='simple',
            tree_depth=3,
            n_features=5
        )

        genes = chrome.to_numpy()

        # First 7: feature indices
        assert list(genes[0:7]) == [0, 1, 2, 3, 4, 0, 1]

        # Next 7: thresholds
        np.testing.assert_allclose(genes[7:14], [100.0, 200.0, 50.0, 150.0, 300.0, 75.0, 125.0], rtol=1e-5)

        # Last 1: tiebreak
        assert genes[14] == 2

    def test_from_numpy_reconstructs_chromosome(self):
        """from_numpy should reconstruct equivalent chromosome."""
        original = Chromosome(
            feature_indices=[0, 1, 2, 3, 4, 0, 1],
            thresholds=[100.0, 200.0, 50.0, 150.0, 300.0, 75.0, 125.0],
            tiebreak_feature=2,
            config_type='simple',
            tree_depth=3,
            n_features=5
        )

        genes = original.to_numpy()
        reconstructed = Chromosome.from_numpy(genes, tree_depth=3, n_features=5, config_type='simple')

        assert reconstructed.feature_indices == original.feature_indices
        np.testing.assert_allclose(reconstructed.thresholds, original.thresholds, rtol=1e-5)
        assert reconstructed.tiebreak_feature == original.tiebreak_feature
        assert reconstructed.n_genes == 15

    def test_roundtrip_random_chromosome(self):
        """Random chromosome should survive numpy roundtrip."""
        rng = np.random.default_rng(42)

        original = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=rng
        )

        genes = original.to_numpy()
        reconstructed = Chromosome.from_numpy(genes, tree_depth=3, n_features=5, config_type='simple')

        assert reconstructed.feature_indices == original.feature_indices
        np.testing.assert_allclose(reconstructed.thresholds, original.thresholds, rtol=1e-5)
        assert reconstructed.tiebreak_feature == original.tiebreak_feature


class TestSimpleConfigMutation:
    """Test mutation for simple config."""

    def test_mutation_produces_valid_offspring(self):
        """Mutated chromosome should pass validation."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=rng
        )

        mutated = chrome.mutate(mutation_rate=0.5, mutation_sigma=0.1, rng=rng)

        # Should be valid
        mutated.validate()
        assert mutated.n_genes == 15

    def test_mutation_rate_zero_returns_clone(self):
        """mutation_rate=0 should return identical chromosome."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=rng
        )

        mutated = chrome.mutate(mutation_rate=0.0, rng=rng)

        assert mutated.feature_indices == chrome.feature_indices
        assert mutated.thresholds == chrome.thresholds
        assert mutated.tiebreak_feature == chrome.tiebreak_feature


class TestSimpleConfigCrossover:
    """Test crossover for simple config."""

    def test_crossover_produces_valid_offspring(self):
        """Crossover should produce valid offspring."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)

        child1, child2 = parent1.crossover(parent2, crossover_rate=0.7, rng=rng)

        child1.validate()
        child2.validate()

        assert child1.n_genes == 15
        assert child2.n_genes == 15

    def test_crossover_maintains_structure(self):
        """Offspring should have same structure as parents."""
        rng = np.random.default_rng(42)

        parent1 = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)
        parent2 = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)

        child1, child2 = parent1.crossover(parent2, crossover_rate=0.7, rng=rng)

        assert child1.tree_depth == 3
        assert child1.n_features == 5
        assert child1.config_type == 'simple'
        assert len(child1.feature_indices) == 7
        assert len(child1.thresholds) == 7


class TestSimpleConfigSerialization:
    """Test JSON serialization for simple config."""

    def test_to_dict_includes_correct_fields(self):
        """to_dict should include all necessary fields."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)
        data = chrome.to_dict()

        assert 'feature_indices' in data
        assert 'thresholds' in data
        assert 'tiebreak_feature' in data
        assert 'config_type' in data
        assert 'n_genes' in data

        assert len(data['feature_indices']) == 7
        assert len(data['thresholds']) == 7
        assert data['config_type'] == 'simple'
        assert data['n_genes'] == 15

    def test_roundtrip_dict_serialization(self):
        """Chromosome should survive dict roundtrip."""
        rng = np.random.default_rng(42)

        original = Chromosome.random(tree_depth=3, n_features=5, config_type='simple', rng=rng)
        data = original.to_dict()
        reconstructed = Chromosome.from_dict(data)

        assert reconstructed.feature_indices == original.feature_indices
        assert reconstructed.thresholds == original.thresholds
        assert reconstructed.tiebreak_feature == original.tiebreak_feature
        assert reconstructed.n_genes == 15


class TestSimpleConfigDeterminism:
    """Test deterministic behavior with fixed seed."""

    def test_random_with_seed_is_deterministic(self):
        """Random chromosomes with same seed should be identical."""
        chrome1 = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=np.random.default_rng(42)
        )

        chrome2 = Chromosome.random(
            tree_depth=3,
            n_features=5,
            config_type='simple',
            rng=np.random.default_rng(42)
        )

        assert chrome1.feature_indices == chrome2.feature_indices
        assert chrome1.thresholds == chrome2.thresholds
        assert chrome1.tiebreak_feature == chrome2.tiebreak_feature


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
