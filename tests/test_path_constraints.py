"""
Tests for Path-Aware Threshold Constraints in Decision Tree Chromosomes

These tests verify that:
1. Effective bounds are computed correctly based on ancestor splits
2. Random generation produces path-consistent chromosomes
3. Mutation preserves path constraints
4. Crossover repairs path constraint violations
5. Validation catches path constraint violations
6. All leaves remain reachable (no impossible paths)
"""

import pytest
import numpy as np
from policy.chromosome import Chromosome


class TestEffectiveBoundsComputation:
    """Test _compute_effective_bounds() helper method."""

    def test_root_node_uses_base_bounds(self):
        """Root node (idx 0) should use full base bounds."""
        base_bounds = (0.0, 500.0)
        feature_indices = [1, 2, 1, 1, 0, 1, 0]
        thresholds = [200.0, 100.0, 150.0, 50.0, 80.0, 75.0, 40.0]

        eff_min, eff_max = Chromosome._compute_effective_bounds(
            node_idx=0,
            feature_idx=1,
            feature_indices=feature_indices,
            thresholds=thresholds,
            base_bounds=base_bounds
        )

        assert eff_min == 0.0
        assert eff_max == 500.0

    def test_left_child_narrows_upper_bound(self):
        """Left child of root (went left: feature < threshold) should narrow upper bound."""
        base_bounds = (-2.0, 500.0)
        feature_indices = [1, 1, 1, 1, 0, 1, 0]  # All use feature 1
        thresholds = [300.0, 100.0, 400.0, 50.0, 80.0, 75.0, 40.0]

        # Node 1 is left child of node 0 (feature 1 < 300.0)
        eff_min, eff_max = Chromosome._compute_effective_bounds(
            node_idx=1,
            feature_idx=1,
            feature_indices=feature_indices,
            thresholds=thresholds,
            base_bounds=base_bounds
        )

        # Should have upper bound narrowed by parent's threshold
        assert eff_min == -2.0
        assert eff_max == 300.0

    def test_right_child_narrows_lower_bound(self):
        """Right child of root (went right: feature >= threshold) should narrow lower bound."""
        base_bounds = (-2.0, 500.0)
        feature_indices = [1, 1, 1, 1, 0, 1, 0]  # All use feature 1
        thresholds = [300.0, 100.0, 400.0, 50.0, 80.0, 75.0, 40.0]

        # Node 2 is right child of node 0 (feature 1 >= 300.0)
        eff_min, eff_max = Chromosome._compute_effective_bounds(
            node_idx=2,
            feature_idx=1,
            feature_indices=feature_indices,
            thresholds=thresholds,
            base_bounds=base_bounds
        )

        # Should have lower bound narrowed by parent's threshold
        assert eff_min == 300.0
        assert eff_max == 500.0

    def test_multiple_ancestors_compound(self):
        """Multiple ancestors using same feature should compound constraints."""
        base_bounds = (-2.0, 500.0)
        feature_indices = [1, 1, 1, 1, 0, 1, 0]  # Nodes 0, 1, 3 all use feature 1
        thresholds = [300.0, 200.0, 400.0, 100.0, 80.0, 75.0, 40.0]

        # Node 3 path: root(0) -> node 1 (left, <300) -> node 3 (left, <200)
        eff_min, eff_max = Chromosome._compute_effective_bounds(
            node_idx=3,
            feature_idx=1,
            feature_indices=feature_indices,
            thresholds=thresholds,
            base_bounds=base_bounds
        )

        # Compounded: feature 1 < 300 AND feature 1 < 200 => feature 1 < 200
        assert eff_min == -2.0
        assert eff_max == 200.0

    def test_different_feature_no_narrowing(self):
        """Using different feature than ancestors should not narrow bounds."""
        base_bounds = (0.0, 258.0)
        feature_indices = [1, 2, 1, 0, 0, 1, 0]  # Node 3 uses feature 0
        thresholds = [300.0, 100.0, 400.0, 50.0, 80.0, 75.0, 40.0]

        # Node 3 uses feature 0, ancestors use features 1 and 2
        eff_min, eff_max = Chromosome._compute_effective_bounds(
            node_idx=3,
            feature_idx=0,
            feature_indices=feature_indices,
            thresholds=thresholds,
            base_bounds=base_bounds
        )

        # No narrowing since ancestors use different features
        assert eff_min == 0.0
        assert eff_max == 258.0

    def test_partial_thresholds_during_generation(self):
        """Should handle partial threshold list during sequential generation."""
        base_bounds = (-2.0, 500.0)
        feature_indices = [1, 1, 1, 1, 0, 1, 0]
        thresholds = [300.0, 200.0]  # Only first 2 thresholds set

        # Node 3 - should only see ancestors 0 and 1
        eff_min, eff_max = Chromosome._compute_effective_bounds(
            node_idx=3,
            feature_idx=1,
            feature_indices=feature_indices,
            thresholds=thresholds,
            base_bounds=base_bounds
        )

        # Node 3 path: root(0) -> node 1 (left)
        # Both use feature 1 with thresholds 300 and 200
        assert eff_min == -2.0
        assert eff_max == 200.0


class TestRandomGeneratesValidPaths:
    """Test that random chromosome generation produces path-consistent trees."""

    @pytest.fixture
    def feature_bounds(self):
        """Standard feature bounds."""
        return [(0, 258), (-2, 500), (-2, 250), (0, 500), (0, 500)]

    def test_simple_config_random_valid(self, feature_bounds):
        """Random simple config should pass validation."""
        rng = np.random.default_rng(42)

        for _ in range(20):  # Test multiple random chromosomes
            chrome = Chromosome.random(
                tree_depth=3,
                n_features=5,
                feature_bounds=feature_bounds,
                config_type='simple',
                rng=rng
            )

            # Should not raise
            chrome.validate()

    def test_no_impossible_paths_in_random(self, feature_bounds):
        """Random chromosomes should not have impossible paths."""
        rng = np.random.default_rng(123)

        for _ in range(20):
            chrome = Chromosome.random(
                tree_depth=3,
                n_features=5,
                feature_bounds=feature_bounds,
                config_type='simple',
                rng=rng
            )

            # Verify all nodes have valid effective bounds
            for node_idx in range(len(chrome.feature_indices)):
                feat_idx = chrome.feature_indices[node_idx]
                base_bounds = feature_bounds[feat_idx]

                eff_min, eff_max = Chromosome._compute_effective_bounds(
                    node_idx, feat_idx, chrome.feature_indices,
                    chrome.thresholds, base_bounds
                )

                # Effective bounds should be non-empty
                assert eff_min < eff_max, f"Node {node_idx} has collapsed bounds [{eff_min}, {eff_max}]"


class TestMutationPreservesConstraints:
    """Test that mutation preserves path constraints."""

    @pytest.fixture
    def feature_bounds(self):
        return [(0, 258), (-2, 500), (-2, 250), (0, 500), (0, 500)]

    def test_mutated_chromosome_valid(self, feature_bounds):
        """Mutated chromosomes should pass validation."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            # Create random parent
            parent = Chromosome.random(
                tree_depth=3,
                n_features=5,
                feature_bounds=feature_bounds,
                config_type='simple',
                rng=rng
            )

            # Mutate with high rate to stress test
            mutated = parent.mutate(
                mutation_rate=0.5,
                mutation_sigma=0.3,
                feature_bounds=feature_bounds,
                rng=rng
            )

            # Should not raise
            mutated.validate()

    def test_aggressive_mutation_still_valid(self, feature_bounds):
        """Even aggressive mutation should produce valid chromosomes."""
        rng = np.random.default_rng(999)

        parent = Chromosome.random(
            tree_depth=3,
            n_features=5,
            feature_bounds=feature_bounds,
            config_type='simple',
            rng=rng
        )

        # Very aggressive mutation
        for _ in range(10):
            mutated = parent.mutate(
                mutation_rate=1.0,  # Mutate every gene
                mutation_sigma=0.5,
                feature_bounds=feature_bounds,
                rng=rng
            )
            mutated.validate()
            parent = mutated  # Chain mutations


class TestCrossoverRepairsConstraints:
    """Test that crossover repairs path constraint violations."""

    @pytest.fixture
    def feature_bounds(self):
        return [(0, 258), (-2, 500), (-2, 250), (0, 500), (0, 500)]

    def test_crossed_offspring_valid(self, feature_bounds):
        """Offspring from crossover should pass validation."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            parent1 = Chromosome.random(
                tree_depth=3,
                n_features=5,
                feature_bounds=feature_bounds,
                config_type='simple',
                rng=rng
            )
            parent2 = Chromosome.random(
                tree_depth=3,
                n_features=5,
                feature_bounds=feature_bounds,
                config_type='simple',
                rng=rng
            )

            child1, child2 = parent1.crossover(
                parent2,
                crossover_rate=1.0,  # Always crossover
                feature_bounds=feature_bounds,
                rng=rng
            )

            # Both children should be valid
            child1.validate()
            child2.validate()

    def test_deterministic_crossover_with_seed(self, feature_bounds):
        """Same seed should produce same offspring."""
        parent1 = Chromosome.random(
            tree_depth=3, n_features=5, feature_bounds=feature_bounds,
            config_type='simple', rng=np.random.default_rng(100)
        )
        parent2 = Chromosome.random(
            tree_depth=3, n_features=5, feature_bounds=feature_bounds,
            config_type='simple', rng=np.random.default_rng(200)
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        child1_a, child2_a = parent1.crossover(parent2, crossover_rate=1.0,
                                                feature_bounds=feature_bounds, rng=rng1)
        child1_b, child2_b = parent1.crossover(parent2, crossover_rate=1.0,
                                                feature_bounds=feature_bounds, rng=rng2)

        assert child1_a.feature_indices == child1_b.feature_indices
        assert child1_a.thresholds == child1_b.thresholds


class TestValidationCatchesViolations:
    """Test that validation catches manually constructed violations."""

    def test_path_violation_detected(self):
        """Manually constructed path violation should fail validation."""
        # Create a chromosome where node 2 (right child of 0) uses same feature
        # with threshold that creates impossible path
        with pytest.raises(ValueError, match="Path constraint violation"):
            Chromosome(
                feature_indices=[1, 2, 1, 1, 0, 1, 0],  # Nodes 0 and 2 both use feature 1
                thresholds=[300.0, 100.0, 100.0, 50.0, 80.0, 75.0, 40.0],
                # Node 0: feature 1 < 300 (left) or >= 300 (right)
                # Node 2: feature 1 < 100 - but we got here via right (>=300)!
                # This is impossible: feature 1 >= 300 AND feature 1 < 100
                tiebreak_feature=0,
                config_type='simple',
                tree_depth=3,
                n_features=5
            )

    def test_valid_chromosome_passes(self):
        """Properly constructed chromosome should pass validation."""
        # Create a valid chromosome
        chrome = Chromosome(
            feature_indices=[1, 2, 1, 1, 0, 1, 0],
            thresholds=[300.0, 100.0, 400.0, 50.0, 80.0, 350.0, 40.0],
            # Node 2: feature 1 < 400 - valid since we came from right (>=300)
            # 300 <= feature 1 < 400 is possible
            tiebreak_feature=0,
            config_type='simple',
            tree_depth=3,
            n_features=5
        )

        # Should not raise
        chrome.validate()


class TestAllLeavesReachable:
    """Test that all leaves remain reachable in generated trees."""

    @pytest.fixture
    def feature_bounds(self):
        return [(0, 258), (-2, 500), (-2, 250), (0, 500), (0, 500)]

    def test_random_tree_leaves_reachable(self, feature_bounds):
        """All 8 leaves should be potentially reachable in random trees."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            chrome = Chromosome.random(
                tree_depth=3,
                n_features=5,
                feature_bounds=feature_bounds,
                config_type='simple',
                rng=rng
            )

            # For each leaf, verify the path to it has non-empty bounds
            n_splits = 7
            for leaf_idx in range(8):
                # Compute path to this leaf
                node_idx = leaf_idx + n_splits  # Leaf indices start after internal nodes

                # Walk back to root checking bounds
                current = node_idx
                path_valid = True

                while current > 0:
                    parent = (current - 1) // 2
                    is_left = (current == 2 * parent + 1)

                    if parent < n_splits:  # Internal node
                        feat_idx = chrome.feature_indices[parent]
                        threshold = chrome.thresholds[parent]
                        base_bounds = feature_bounds[feat_idx]

                        eff_min, eff_max = Chromosome._compute_effective_bounds(
                            parent, feat_idx, chrome.feature_indices,
                            chrome.thresholds, base_bounds
                        )

                        # Check if threshold creates valid split
                        if is_left:
                            # Going left: need some values < threshold
                            if eff_min >= threshold:
                                path_valid = False
                                break
                        else:
                            # Going right: need some values >= threshold
                            if eff_max <= threshold:
                                path_valid = False
                                break

                    current = parent

                assert path_valid, f"Leaf {leaf_idx} is unreachable in chromosome"


class TestRepairPathConstraints:
    """Test _repair_path_constraints() method directly."""

    @pytest.fixture
    def feature_bounds(self):
        return [(0, 258), (-2, 500), (-2, 250), (0, 500), (0, 500)]

    def test_repair_fixes_violation(self, feature_bounds):
        """Repair should fix path constraint violations."""
        # Create indices and thresholds with a violation
        feature_indices = [1, 2, 1, 1, 0, 1, 0]  # Nodes 0 and 2 both use feature 1
        thresholds = [300.0, 100.0, 100.0, 50.0, 80.0, 75.0, 40.0]
        # Node 2 violation: should be >= 300 (came from right) but threshold is 100

        rng = np.random.default_rng(42)
        repaired_features, repaired_thresholds = Chromosome._repair_path_constraints(
            feature_indices, thresholds, feature_bounds, 5, rng
        )

        # Create chromosome with repaired values - should not raise
        chrome = Chromosome(
            feature_indices=repaired_features,
            thresholds=repaired_thresholds,
            tiebreak_feature=0,
            config_type='simple',
            tree_depth=3,
            n_features=5
        )
        chrome.validate()

    def test_repair_preserves_valid(self, feature_bounds):
        """Repair should not change already valid chromosomes significantly."""
        rng = np.random.default_rng(42)

        chrome = Chromosome.random(
            tree_depth=3,
            n_features=5,
            feature_bounds=feature_bounds,
            config_type='simple',
            rng=rng
        )

        repaired_features, repaired_thresholds = Chromosome._repair_path_constraints(
            chrome.feature_indices.copy(),
            chrome.thresholds.copy(),
            feature_bounds,
            5,
            rng
        )

        # Features should be unchanged (no violations to fix)
        assert repaired_features == chrome.feature_indices
        # Thresholds should be very close (just clamping to effective bounds)
        for orig, rep in zip(chrome.thresholds, repaired_thresholds):
            assert abs(orig - rep) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
