"""
Chromosome Data Structure for GA-Optimized Decision Trees

Encodes decision tree parameters as genes:
- variable_indices: Which features to split on at each decision node
- thresholds: Split threshold values (decision boundaries)
- tiebreak_feature: Feature used for within-bucket ordering (global across all contexts)
- Direct bucket mapping: 8 leaves → buckets 1-8 (leaves 0-7 map to buckets 1-8)
- early_phase_window: Proactive phase trigger (Full config only, 0-100 hours)
- Optional hierarchical context: Fleet splits for multi-context trees

This representation allows the GA to simultaneously optimize both which features
to use AND where to split them, while classifying aircraft into 8 buckets for
greedy sweep action assignment.

Architecture:
- NMC aircraft (Bucket 0): Bypass tree entirely (deterministic execution logic)
- FMC aircraft: Classified into Buckets 1-8 by 15-node decision tree
- Action assignment: Three-sweep greedy algorithm (see GREEDY_SWEEP_ADJUDICATION.md)
- Early phase (Full config only): Binding enforcement for buckets 1-2 when hours_to_major < early_phase_window

Gene Counts by Configuration (depth=3):
- Simple (flat): 15 genes (7 variable indices + 7 thresholds + 1 tiebreak)
- Medium (hierarchical): 31 genes (2 fleet + 2×14 subtree + 1 tiebreak)
- Full (deep hierarchical): 72 genes (14 fleet + 4×14 subtree + 1 tiebreak + 1 early_phase_window)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json


@dataclass
class Chromosome:
    """
    Genetic encoding of a decision tree policy for aircraft maintenance scheduling.

    For a flat tree of depth=3:
    - Number of internal decision nodes: 7 (1 + 2 + 4)
    - Number of leaf nodes: 8
    - Direct bucket mapping: leaf_idx + 1 = bucket (leaves 0-7 → buckets 1-8)
    - Total genes: 7 variable_indices + 7 thresholds + 1 tiebreak = 15 genes

    Simple Configuration (15 genes total):
        variable_indices = [0, 1, 2, 3, 4, 0, 1]  # 7 internal nodes
        thresholds = [100.0, 200.0, 150.0, ...]  # 7 thresholds (raw hour values)
        tiebreak_feature = 0  # Global tiebreaker for within-bucket ordering (0-4)

        # Direct bucket mapping (not encoded as gene):
        # leaf 0 → bucket 1, leaf 1 → bucket 2, ..., leaf 7 → bucket 8

    Tree structure (depth=3, 7 internal decision nodes):
                                Root (var[0] < thresh[0])?
                                /                              \\
                      Node1 (var[1])                    Node2 (var[2])
                      /              \\                  /              \\
              Node3 (var[3])    Node4 (var[4])    Node5 (var[5])    Node6 (var[6])
              /      \\          /      \\          /      \\          /      \\
            L0  L1          L2  L3          L4  L5          L6  L7
        (8 leaves: direct mapping to buckets 1-8, NOT encoded as genes)

    Medium Configuration (31 genes total):
        fleet_variable_indices = [0]  # 1 fleet split (2 genes: 1 var + 1 threshold)
        fleet_thresholds = [0.5]  # Normalized fleet feature threshold

        # Context A subtree (14 genes: 7 vars + 7 thresholds, NO tiebreak)
        context_A_variable_indices = [0, 1, 2, ...]  # 7 internal nodes
        context_A_thresholds = [100.0, 200.0, ...]  # 7 thresholds

        # Context B subtree (14 genes: 7 vars + 7 thresholds, NO tiebreak)
        context_B_variable_indices = [1, 2, 0, ...]  # 7 internal nodes (different structure!)
        context_B_thresholds = [150.0, 250.0, ...]  # 7 thresholds

        # Global tiebreaker (1 gene - applies across ALL contexts)
        tiebreak_feature = 2  # Feature index for within-bucket sorting

    Full Configuration (72 genes total):
        # Fleet tree: depth-3 tree with 7 internal nodes → 8 leaves → 4 contexts via [1,2,2,2,3,3,3,4]
        fleet_variable_indices = [0, 1, 2, 3, 4, 5, 6]  # 7 internal nodes
        fleet_thresholds = [0.5, 0.3, 0.7, ...]  # 7 normalized thresholds

        # 4 context subtrees (4 × 14 = 56 genes: each subtree has 7 vars + 7 thresholds, NO tiebreak)
        context_1_* ... context_4_* (each 14 genes)

        # Global tiebreaker (1 gene - applies across ALL contexts)
        tiebreak_feature = 0

        # Early phase window (1 gene - FULL CONFIG ONLY)
        early_phase_window = 75  # Hours before phase (0-100, binding for buckets 1-2 only)
    """

    feature_indices: List[int]
    """Which feature to split on at each node (0-indexed into feature vector)."""

    thresholds: List[float]
    """Threshold value for each split (feature-specific ranges)."""

    tiebreak_feature: int = 0
    """Feature index used for sorting within buckets (0-4 for per-aircraft features)."""

    config_type: str = 'simple'
    """Configuration type: 'simple' (15 genes) | 'medium' (31 genes) | 'full' (72 genes)."""

    tree_depth: int = 3
    """Depth of decision tree (default: 3 for 8 leaves, 7 internal decision nodes)."""

    n_features: int = field(default=4, repr=False)
    """Number of per-aircraft features available for splitting."""

    feature_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(0, 400), (-2, 500), (-2, 250), (-500, 500)], repr=False)
    """Bounds for each feature: [(min, max), ...]. Default: [observed_rul, hours_to_major, hours_to_minor, da_line_deviation].

    Note: observed_rul max=400 accounts for true_rul max=300 plus Gamma noise right tail.
    da_line_deviation uses symmetric bounds [-500, 500] matching theoretical range."""

    # Hierarchical context support (Medium/Full configs)
    fleet_variable_indices: Optional[List[int]] = None
    """Fleet-level variable indices for context splits (None for simple config)."""

    fleet_thresholds: Optional[List[float]] = None
    """Thresholds for fleet context splits (None for simple config)."""

    context_subtrees: Optional[List['Chromosome']] = None
    """List of subtree chromosomes for hierarchical trees (None for simple config)."""

    early_phase_window: Optional[int] = None
    """Hours before phase trigger for early maintenance (Full config only, 0-100 hours).
    Binding enforcement for buckets 1-2: if hours_to_major < early_phase_window, must do early phase maintenance.
    None for Simple/Medium configs (purely reactive phase maintenance)."""

    rul_threshold: Optional[int] = None
    """Hours before failure to trigger preventive maintenance (5-100 hours).
    When None, DecisionTreePolicy uses default 100h. When set, becomes trainable gene.
    Controlled by _prevtrain experiment flag."""

    def __post_init__(self):
        """Validate chromosome consistency after initialization."""
        self.validate()

    @property
    def n_splits(self) -> int:
        """Number of split nodes in tree."""
        return 2 ** self.tree_depth - 1

    @property
    def n_leaves(self) -> int:
        """Number of leaf nodes in tree."""
        return 2 ** self.tree_depth

    @property
    def n_genes(self) -> int:
        """Total number of genes in chromosome (for depth=3 tree).

        - Simple: 7 vars + 7 thresholds + 1 tiebreak [+ 1 rul_threshold if trainable] = 15 or 16
        - Medium: 1 fleet var + 1 fleet threshold + 2 subtrees (14 each) + 1 tiebreak [+ 1 rul_threshold] = 31 or 32
        - Full: 7 fleet vars + 7 fleet thresholds + 4 subtrees (14 each) + 1 tiebreak + 1 early_phase [+ 1 rul_threshold] = 72 or 73
        """
        rul_gene = 1 if self.rul_threshold is not None else 0
        if self.config_type == 'simple':
            # Flat tree: 7 nodes + 7 thresholds + 1 tiebreak [+ rul] = 15 or 16
            return 2 * self.n_splits + 1 + rul_gene
        elif self.config_type == 'medium':
            # 1 fleet var + 1 fleet threshold + 2×14 subtrees + 1 global tiebreak [+ rul] = 31 or 32
            return 2 + 2 * (2 * self.n_splits) + 1 + rul_gene
        elif self.config_type == 'full':
            # 7 fleet vars + 7 fleet thresholds + 4×14 subtrees + 1 global tiebreak + 1 early_phase [+ rul] = 72 or 73
            return 2 * self.n_splits + 4 * (2 * self.n_splits) + 1 + 1 + rul_gene
        else:
            raise ValueError(f"Unknown config_type: {self.config_type}")

    def get_bucket_from_leaf(self, leaf_idx: int) -> int:
        """Get bucket number from leaf index using direct mapping [1,2,3,4,5,6,7,8].

        Args:
            leaf_idx: Leaf index (0-7)

        Returns:
            bucket: Bucket number (1-8)

        Raises:
            ValueError: If leaf_idx out of range
        """
        if not (0 <= leaf_idx < 8):
            raise ValueError(f"Invalid leaf_idx: {leaf_idx} (must be 0-7)")
        return leaf_idx + 1  # Direct mapping: 0→1, 1→2, ..., 7→8

    @staticmethod
    def _compute_effective_bounds(
        node_idx: int,
        feature_idx: int,
        feature_indices: List[int],
        thresholds: List[float],
        base_bounds: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compute effective bounds for a feature at a given node.

        Walks from root to node_idx, narrowing bounds based on ancestor splits
        on the same feature. This ensures path-consistent thresholds.

        Tree uses breadth-first indexing:
        - Root = 0
        - Left child of node i = 2*i + 1
        - Right child of node i = 2*i + 2
        - Parent of node i = (i - 1) // 2

        Args:
            node_idx: Index of the current node (0 to n_splits-1)
            feature_idx: Feature being used at this node
            feature_indices: List of feature indices for all nodes (may be partial during generation)
            thresholds: List of thresholds for all nodes (may be partial during generation)
            base_bounds: Global (min, max) bounds for this feature

        Returns:
            Tuple of (effective_min, effective_max) bounds for this node
        """
        min_val, max_val = base_bounds

        if node_idx == 0:
            return min_val, max_val

        # Build path from root to this node's parent
        path = []
        current = node_idx
        while current > 0:
            parent = (current - 1) // 2
            is_left = (current == 2 * parent + 1)
            path.append((parent, is_left))
            current = parent
        path.reverse()  # Now path goes root → ... → parent of node_idx

        # Narrow bounds based on ancestors using same feature
        for parent_idx, went_left in path:
            # Only check ancestors that have been assigned (for partial trees during generation)
            if parent_idx >= len(feature_indices) or parent_idx >= len(thresholds):
                continue

            if feature_indices[parent_idx] == feature_idx:
                parent_threshold = thresholds[parent_idx]
                if went_left:
                    # Went left: feature < threshold, so upper bound is threshold
                    max_val = min(max_val, parent_threshold)
                else:
                    # Went right: feature >= threshold, so lower bound is threshold
                    min_val = max(min_val, parent_threshold)

        return min_val, max_val

    def _validate_path_constraints(self) -> None:
        """
        Validate that thresholds satisfy path constraints.

        Ensures that when the same feature is used at multiple nodes along a path,
        the thresholds are logically consistent (no impossible branches).

        Raises:
            ValueError: If path constraint violation detected
        """
        for node_idx in range(len(self.feature_indices)):
            feat_idx = self.feature_indices[node_idx]
            threshold = self.thresholds[node_idx]
            base_min, base_max = self.feature_bounds[feat_idx]

            eff_min, eff_max = self._compute_effective_bounds(
                node_idx, feat_idx, self.feature_indices, self.thresholds,
                (base_min, base_max)
            )

            # Check if threshold is within effective bounds
            # Use small epsilon for floating point comparison
            epsilon = 1e-6
            if threshold < eff_min - epsilon or threshold > eff_max + epsilon:
                raise ValueError(
                    f"Path constraint violation at node {node_idx}: "
                    f"threshold {threshold:.4f} not in effective bounds "
                    f"[{eff_min:.4f}, {eff_max:.4f}] for feature {feat_idx}"
                )

    @staticmethod
    def _repair_path_constraints(
        feature_indices: List[int],
        thresholds: List[float],
        feature_bounds: List[Tuple[float, float]],
        n_features: int,
        rng: np.random.Generator = None
    ) -> Tuple[List[int], List[float]]:
        """
        Repair thresholds (and potentially feature indices) to satisfy path constraints.

        Processes nodes in order, clamping each threshold to effective bounds.
        If bounds collapse completely, re-assigns feature index to avoid constraint violation.

        Args:
            feature_indices: List of feature indices for all nodes (will be modified if needed)
            thresholds: List of thresholds for all nodes
            feature_bounds: Global bounds for each feature
            n_features: Total number of features available
            rng: Random number generator for feature re-rolling

        Returns:
            Tuple of (repaired_feature_indices, repaired_thresholds)
        """
        if rng is None:
            rng = np.random.default_rng()

        repaired_features = feature_indices.copy()
        repaired_thresholds = []

        for node_idx in range(len(feature_indices)):
            feat_idx = repaired_features[node_idx]
            base_min, base_max = feature_bounds[feat_idx]
            original_threshold = thresholds[node_idx]

            # Try to find valid bounds, re-rolling feature if bounds collapse
            max_attempts = 10
            for attempt in range(max_attempts):
                eff_min, eff_max = Chromosome._compute_effective_bounds(
                    node_idx, feat_idx, repaired_features, repaired_thresholds,
                    (base_min, base_max)
                )

                if eff_min < eff_max:
                    # Valid bounds - clamp threshold
                    repaired_threshold = float(np.clip(original_threshold, eff_min, eff_max))
                    break
                else:
                    # Bounds collapsed - try different feature
                    available_features = [f for f in range(n_features) if f != feat_idx]
                    if available_features:
                        feat_idx = int(rng.choice(available_features))
                        repaired_features[node_idx] = feat_idx
                        base_min, base_max = feature_bounds[feat_idx]
                        # Re-sample threshold for new feature within its effective bounds
                        original_threshold = rng.uniform(base_min, base_max)
                    else:
                        # No alternatives - use midpoint as fallback
                        repaired_threshold = (eff_min + eff_max) / 2.0
                        break
            else:
                # Exhausted attempts - use midpoint
                repaired_threshold = (eff_min + eff_max) / 2.0

            repaired_thresholds.append(repaired_threshold)

        return repaired_features, repaired_thresholds

    def validate(self) -> None:
        """
        Validate chromosome structure and constraints.

        Raises:
            ValueError: If chromosome is malformed
        """
        if self.config_type == 'simple':
            # Validate simple config (31 genes)
            expected_splits = self.n_splits  # 15

            if len(self.feature_indices) != expected_splits:
                raise ValueError(
                    f"feature_indices length mismatch: expected {expected_splits}, "
                    f"got {len(self.feature_indices)} for depth {self.tree_depth}"
                )

            if len(self.thresholds) != expected_splits:
                raise ValueError(
                    f"thresholds length mismatch: expected {expected_splits}, "
                    f"got {len(self.thresholds)} for depth {self.tree_depth}"
                )

            # Check feature indices are valid (0-4 for per-aircraft)
            for i, feat_idx in enumerate(self.feature_indices):
                if not (0 <= feat_idx < self.n_features):
                    raise ValueError(
                        f"Invalid feature index at split {i}: {feat_idx} "
                        f"(must be in [0, {self.n_features}))"
                    )

            # Check thresholds are within feature-specific bounds
            for i, threshold in enumerate(self.thresholds):
                feat_idx = self.feature_indices[i]
                min_val, max_val = self.feature_bounds[feat_idx]
                if not (min_val <= threshold <= max_val):
                    raise ValueError(
                        f"Threshold at split {i} (feature {feat_idx}) out of range: {threshold} "
                        f"(must be in [{min_val}, {max_val}])"
                    )

            # Check path constraints (no impossible branches from same feature on path)
            self._validate_path_constraints()

            # Check tiebreak feature is valid
            if not (0 <= self.tiebreak_feature < self.n_features):
                raise ValueError(
                    f"Invalid tiebreak_feature: {self.tiebreak_feature} "
                    f"(must be in [0, {self.n_features}))"
                )

            # Validate rul_threshold if present (any config type)
            if self.rul_threshold is not None:
                if not isinstance(self.rul_threshold, int):
                    raise ValueError(f"rul_threshold must be int, got {type(self.rul_threshold)}")
                if not (5 <= self.rul_threshold <= 100):
                    raise ValueError(
                        f"rul_threshold must be 5-100 hours, got {self.rul_threshold}"
                    )

        elif self.config_type in ['medium', 'full']:
            # Validate hierarchical config
            if self.fleet_variable_indices is None or self.fleet_thresholds is None:
                raise ValueError(
                    f"{self.config_type} config requires fleet_variable_indices and fleet_thresholds"
                )

            if self.context_subtrees is None:
                raise ValueError(f"{self.config_type} config requires context_subtrees")

            # Validate fleet splits
            expected_fleet_nodes = 1 if self.config_type == 'medium' else 7  # Medium: 1 node, Full: depth-3 tree with 7 nodes
            expected_contexts = 2 if self.config_type == 'medium' else 4

            if len(self.fleet_variable_indices) != expected_fleet_nodes:
                raise ValueError(
                    f"fleet_variable_indices length mismatch for {self.config_type}: "
                    f"expected {expected_fleet_nodes}, got {len(self.fleet_variable_indices)}"
                )

            if len(self.fleet_thresholds) != expected_fleet_nodes:
                raise ValueError(
                    f"fleet_thresholds length mismatch for {self.config_type}: "
                    f"expected {expected_fleet_nodes}, got {len(self.fleet_thresholds)}"
                )

            if len(self.context_subtrees) != expected_contexts:
                raise ValueError(
                    f"context_subtrees length mismatch for {self.config_type}: "
                    f"expected {expected_contexts}, got {len(self.context_subtrees)}"
                )

            # Validate fleet feature indices (0-6 for 7 fleet features)
            for i, feat_idx in enumerate(self.fleet_variable_indices):
                if not (0 <= feat_idx < 7):  # 7 fleet features
                    raise ValueError(
                        f"Invalid fleet feature index at node {i}: {feat_idx} "
                        f"(must be in [0, 7))"
                    )

            # Validate fleet thresholds (use feature-specific bounds)
            # Fleet features have their own bounds (e.g., avg_rul: 0-258, pct_nmc: 0-1)
            # For now, accept any reasonable float value - stricter validation in feature config
            for i, threshold in enumerate(self.fleet_thresholds):
                if not isinstance(threshold, (int, float)):
                    raise ValueError(
                        f"Fleet threshold at node {i} must be numeric, got {type(threshold)}"
                    )

            # Validate each subtree recursively
            for i, subtree in enumerate(self.context_subtrees):
                if subtree.config_type != 'simple':
                    raise ValueError(
                        f"Context subtree {i} must have config_type='simple', "
                        f"got '{subtree.config_type}'"
                    )
                subtree.validate()

            # Validate early_phase_window for Full config only
            if self.config_type == 'full':
                if self.early_phase_window is None:
                    raise ValueError("Full config requires early_phase_window (cannot be None)")
                if not isinstance(self.early_phase_window, int):
                    raise ValueError(f"early_phase_window must be int, got {type(self.early_phase_window)}")
                if not (0 <= self.early_phase_window <= 100):
                    raise ValueError(
                        f"early_phase_window must be 0-100 hours, got {self.early_phase_window}"
                    )
            elif self.early_phase_window is not None:
                raise ValueError(
                    f"{self.config_type} config should not have early_phase_window "
                    f"(only Full config), but got {self.early_phase_window}"
                )

            # Validate rul_threshold if present (any config type)
            if self.rul_threshold is not None:
                if not isinstance(self.rul_threshold, int):
                    raise ValueError(f"rul_threshold must be int, got {type(self.rul_threshold)}")
                if not (5 <= self.rul_threshold <= 100):
                    raise ValueError(
                        f"rul_threshold must be 5-100 hours, got {self.rul_threshold}"
                    )

        else:
            raise ValueError(f"Unknown config_type: {self.config_type}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize chromosome to dictionary for JSON export.

        Returns:
            Dictionary representation
        """
        result = {
            'feature_indices': self.feature_indices,
            'thresholds': self.thresholds,
            'tiebreak_feature': self.tiebreak_feature,
            'config_type': self.config_type,
            'tree_depth': self.tree_depth,
            'n_features': self.n_features,
            'n_genes': self.n_genes
        }

        # Add early_phase_window only for Full config
        if self.config_type == 'full' and self.early_phase_window is not None:
            result['early_phase_window'] = self.early_phase_window

        # Add rul_threshold if present (any config)
        if self.rul_threshold is not None:
            result['rul_threshold'] = self.rul_threshold

        # Add hierarchical fields if present (Medium/Full)
        if self.fleet_variable_indices is not None:
            result['fleet_variable_indices'] = self.fleet_variable_indices
        if self.fleet_thresholds is not None:
            result['fleet_thresholds'] = self.fleet_thresholds
        if self.context_subtrees is not None:
            # Recursively serialize subtrees
            result['context_subtrees'] = [subtree.to_dict() for subtree in self.context_subtrees]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chromosome':
        """
        Deserialize chromosome from dictionary.

        Args:
            data: Dictionary with chromosome fields

        Returns:
            Chromosome instance
        """
        config_type = data.get('config_type', 'simple')

        # Recursively deserialize context subtrees if present
        context_subtrees = None
        if 'context_subtrees' in data and data['context_subtrees'] is not None:
            context_subtrees = [cls.from_dict(subtree_data) for subtree_data in data['context_subtrees']]

        return cls(
            feature_indices=data['feature_indices'],
            thresholds=data['thresholds'],
            tiebreak_feature=data.get('tiebreak_feature', 0),
            config_type=config_type,
            tree_depth=data.get('tree_depth', 3),
            n_features=data.get('n_features', 5),
            fleet_variable_indices=data.get('fleet_variable_indices'),
            fleet_thresholds=data.get('fleet_thresholds'),
            context_subtrees=context_subtrees,
            early_phase_window=data.get('early_phase_window'),  # None for Simple/Medium, int for Full
            rul_threshold=data.get('rul_threshold')  # None if not trainable, int if trainable
        )

    def to_json(self, filepath: str) -> None:
        """
        Save chromosome to JSON file.

        Args:
            filepath: Path to save JSON
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'Chromosome':
        """
        Load chromosome from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Chromosome instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_numpy(self) -> np.ndarray:
        """
        Convert chromosome to flat numpy array for GA operations.

        NOTE: Currently only supports Simple config. Hierarchical configs (Medium/Full)
        should use object-based GA operations instead.

        Returns:
            1D array of shape (n_genes,)

        Raises:
            NotImplementedError: If called on hierarchical config
        """
        if self.config_type != 'simple':
            raise NotImplementedError(
                f"to_numpy() only supports Simple config, got {self.config_type}. "
                "Use object-based GA operations for hierarchical chromosomes."
            )

        # Simple config: 7 indices + 7 thresholds + 1 tiebreak = 15 genes (depth=3)
        return np.concatenate([
            np.array(self.feature_indices, dtype=np.int32),
            np.array(self.thresholds, dtype=np.float32),
            np.array([self.tiebreak_feature], dtype=np.int32)
        ])

    @classmethod
    def from_numpy(
        cls,
        genes: np.ndarray,
        tree_depth: int,
        n_features: int,
        config_type: str = 'simple'
    ) -> 'Chromosome':
        """
        Reconstruct chromosome from flat numpy array.

        NOTE: Currently only supports Simple config. Hierarchical configs should use
        object-based construction instead.

        Args:
            genes: 1D array of genes (expected: 15 genes for simple config, depth=3)
            tree_depth: Depth of tree (default: 3)
            n_features: Number of features available (default: 5)
            config_type: Must be 'simple' (default: 'simple')

        Returns:
            Chromosome instance

        Raises:
            NotImplementedError: If config_type is not 'simple'
        """
        if config_type != 'simple':
            raise NotImplementedError(
                f"from_numpy() only supports Simple config, got {config_type}"
            )

        n_splits = 2 ** tree_depth - 1  # 7 for depth=3

        # Extract gene segments (Simple config: 7 + 7 + 1 = 15 genes for depth=3)
        idx = 0
        feature_indices = genes[idx:idx+n_splits].astype(int).tolist()
        idx += n_splits

        thresholds = genes[idx:idx+n_splits].tolist()
        idx += n_splits

        tiebreak_feature = int(genes[idx])

        return cls(
            feature_indices=feature_indices,
            thresholds=thresholds,
            tiebreak_feature=tiebreak_feature,
            config_type='simple',
            tree_depth=tree_depth,
            n_features=n_features,
            early_phase_window=None  # Simple config has no early phase
        )

    @classmethod
    def random(
        cls,
        tree_depth: int = 3,
        n_features: int = 5,
        feature_bounds: List[Tuple[float, float]] = None,
        config_type: str = 'simple',
        n_fleet_features: int = 7,
        fleet_feature_bounds: List[Tuple[float, float]] = None,
        trainable_rul_threshold: bool = False,
        rng: np.random.Generator = None
    ) -> 'Chromosome':
        """
        Generate random chromosome for GA initialization.

        Supports three configurations (gene counts for depth=3):
        - Simple (15 or 16 genes): Flat decision tree, no fleet context, no early phase
        - Medium (31 or 32 genes): 2-context hierarchy with 1 fleet split, no early phase
        - Full (72 or 73 genes): 4-context deep hierarchy with 7 fleet nodes, early phase window

        Args:
            tree_depth: Depth of per-aircraft tree (default: 3)
            n_features: Number of per-aircraft features (default: 5)
            feature_bounds: Bounds for per-aircraft features [(min, max), ...]. If None, uses default.
            config_type: Configuration type ('simple', 'medium', 'full')
            n_fleet_features: Number of fleet-level features (default: 8)
            fleet_feature_bounds: Bounds for fleet features. If None, uses normalized [0, 1].
            trainable_rul_threshold: If True, initializes rul_threshold gene (5-100h)
            rng: Random number generator (default: new RNG)

        Returns:
            Random chromosome with valid genes
        """
        if rng is None:
            rng = np.random.default_rng()

        if feature_bounds is None:
            raise ValueError("feature_bounds must be provided - no default fallback")

        if fleet_feature_bounds is None:
            # Default fleet feature bounds (all normalized [0, 1] for now)
            fleet_feature_bounds = [(0, 1) for _ in range(n_fleet_features)]

        # --- SIMPLE CONFIG: Flat 15-gene tree (depth=3) ---
        if config_type == 'simple':
            n_splits = 2 ** tree_depth - 1  # 7 for depth=3

            # Random feature indices (initial assignment)
            feature_indices = rng.integers(0, n_features, size=n_splits).tolist()

            # Generate thresholds sequentially with path-aware bounds
            thresholds = []
            for node_idx in range(n_splits):
                feat_idx = feature_indices[node_idx]
                base_min, base_max = feature_bounds[feat_idx]

                # Try to find valid bounds, re-rolling feature if bounds collapse
                max_attempts = 10
                for attempt in range(max_attempts):
                    eff_min, eff_max = cls._compute_effective_bounds(
                        node_idx, feat_idx, feature_indices, thresholds,
                        (base_min, base_max)
                    )

                    if eff_min < eff_max:
                        # Valid bounds - sample threshold
                        threshold = float(rng.uniform(eff_min, eff_max))
                        break
                    else:
                        # Bounds collapsed - try different feature
                        available_features = [f for f in range(n_features) if f != feat_idx]
                        if available_features:
                            feat_idx = int(rng.choice(available_features))
                            feature_indices[node_idx] = feat_idx
                            base_min, base_max = feature_bounds[feat_idx]
                        else:
                            # No alternatives - use midpoint as fallback
                            threshold = (eff_min + eff_max) / 2.0
                            break
                else:
                    # Exhausted attempts - use midpoint of last effective bounds
                    threshold = (eff_min + eff_max) / 2.0

                thresholds.append(threshold)

            # Random tiebreak feature
            tiebreak_feature = int(rng.integers(0, n_features))

            # Optional trainable RUL threshold (5-100h)
            rul_threshold = int(rng.integers(5, 101)) if trainable_rul_threshold else None

            return cls(
                feature_indices=feature_indices,
                thresholds=thresholds,
                tiebreak_feature=tiebreak_feature,
                config_type='simple',
                tree_depth=tree_depth,
                n_features=n_features,
                early_phase_window=None,  # No early phase in Simple
                rul_threshold=rul_threshold
            )

        # --- MEDIUM CONFIG: 2-context hierarchy (31 genes for depth=3) ---
        elif config_type == 'medium':
            # Generate 1 fleet-level split (2 genes: 1 index + 1 threshold)
            fleet_variable_indices = [int(rng.integers(0, n_fleet_features))]

            fleet_thresholds = []
            for feat_idx in fleet_variable_indices:
                min_val, max_val = fleet_feature_bounds[feat_idx]
                threshold = float(rng.uniform(min_val, max_val))
                fleet_thresholds.append(threshold)

            # Generate 2 simple subtrees (14 genes each = 28 total for depth=3)
            context_subtrees = []
            for _ in range(2):
                subtree = cls.random(
                    tree_depth=tree_depth,
                    n_features=n_features,
                    feature_bounds=feature_bounds,
                    config_type='simple',
                    rng=rng
                )
                context_subtrees.append(subtree)

            # Global tiebreak feature
            tiebreak_feature = int(rng.integers(0, n_features))

            # Optional trainable RUL threshold (5-100h)
            rul_threshold = int(rng.integers(5, 101)) if trainable_rul_threshold else None

            return cls(
                feature_indices=[],  # No top-level per-aircraft splits
                thresholds=[],
                tiebreak_feature=tiebreak_feature,
                config_type='medium',
                tree_depth=tree_depth,
                n_features=n_features,
                fleet_variable_indices=fleet_variable_indices,
                fleet_thresholds=fleet_thresholds,
                context_subtrees=context_subtrees,
                early_phase_window=None,  # No early phase in Medium
                rul_threshold=rul_threshold
            )

        # --- FULL CONFIG: 4-context deep hierarchy (72 genes for depth=3) ---
        elif config_type == 'full':
            # Generate 7 fleet-level nodes (depth-3 tree: 14 genes = 7 indices + 7 thresholds)
            n_fleet_splits = 2 ** 3 - 1  # 7

            fleet_variable_indices = rng.integers(0, n_fleet_features, size=n_fleet_splits).tolist()

            # Generate fleet thresholds sequentially with path-aware bounds
            fleet_thresholds = []
            for node_idx in range(n_fleet_splits):
                feat_idx = fleet_variable_indices[node_idx]
                base_min, base_max = fleet_feature_bounds[feat_idx]

                # Try to find valid bounds, re-rolling feature if bounds collapse
                max_attempts = 10
                for attempt in range(max_attempts):
                    eff_min, eff_max = cls._compute_effective_bounds(
                        node_idx, feat_idx, fleet_variable_indices, fleet_thresholds,
                        (base_min, base_max)
                    )

                    if eff_min < eff_max:
                        threshold = float(rng.uniform(eff_min, eff_max))
                        break
                    else:
                        # Bounds collapsed - try different feature
                        available_features = [f for f in range(n_fleet_features) if f != feat_idx]
                        if available_features:
                            feat_idx = int(rng.choice(available_features))
                            fleet_variable_indices[node_idx] = feat_idx
                            base_min, base_max = fleet_feature_bounds[feat_idx]
                        else:
                            threshold = (eff_min + eff_max) / 2.0
                            break
                else:
                    threshold = (eff_min + eff_max) / 2.0

                fleet_thresholds.append(threshold)

            # Generate 4 simple subtrees (14 genes each = 56 total for depth=3)
            context_subtrees = []
            for _ in range(4):
                subtree = cls.random(
                    tree_depth=tree_depth,
                    n_features=n_features,
                    feature_bounds=feature_bounds,
                    config_type='simple',
                    rng=rng
                )
                context_subtrees.append(subtree)

            # Global tiebreak feature
            tiebreak_feature = int(rng.integers(0, n_features))

            # Early phase window (0-100 hours, Full config only)
            early_phase_window = int(rng.integers(0, 101))

            # Optional trainable RUL threshold (5-100h)
            rul_threshold = int(rng.integers(5, 101)) if trainable_rul_threshold else None

            return cls(
                feature_indices=[],  # No top-level per-aircraft splits
                thresholds=[],
                tiebreak_feature=tiebreak_feature,
                config_type='full',
                tree_depth=tree_depth,
                n_features=n_features,
                fleet_variable_indices=fleet_variable_indices,
                fleet_thresholds=fleet_thresholds,
                context_subtrees=context_subtrees,
                early_phase_window=early_phase_window,
                rul_threshold=rul_threshold
            )

        else:
            raise ValueError(f"Unknown config_type: {config_type}. Must be 'simple', 'medium', or 'full'.")

    def mutate(
        self,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.1,
        feature_bounds: List[Tuple[float, float]] = None,
        fleet_feature_bounds: List[Tuple[float, float]] = None,
        rng: np.random.Generator = None
    ) -> 'Chromosome':
        """
        Create mutated copy of chromosome.

        Supports hierarchical mutation for Medium/Full configs by recursively
        mutating subtrees while mutating fleet-level genes independently.

        Args:
            mutation_rate: Probability of mutating each gene
            mutation_sigma: Standard deviation for Gaussian noise on thresholds
            feature_bounds: Bounds for per-aircraft features (default: standard bounds)
            fleet_feature_bounds: Bounds for fleet features (default: normalized [0, 1])
            rng: Random number generator

        Returns:
            New mutated chromosome
        """
        if rng is None:
            rng = np.random.default_rng()

        if feature_bounds is None:
            # Use self's feature_bounds to stay consistent with validation
            feature_bounds = self.feature_bounds

        if fleet_feature_bounds is None:
            # Default fleet feature bounds (all normalized [0, 1] for now)
            fleet_feature_bounds = [(0, 1) for _ in range(7)]  # 7 fleet features

        # --- SIMPLE CONFIG: Mutate flat tree ---
        if self.config_type == 'simple':
            # Copy genes
            new_feature_indices = self.feature_indices.copy()
            new_thresholds = self.thresholds.copy()

            # Mutate feature indices (random swap)
            for i in range(len(new_feature_indices)):
                if rng.random() < mutation_rate:
                    new_feature_indices[i] = int(rng.integers(0, self.n_features))

            # Mutate thresholds (Gaussian noise with feature-specific bounds)
            # ALWAYS clip to new feature's bounds, even without mutation
            for i in range(len(new_thresholds)):
                feat_idx = new_feature_indices[i]
                min_val, max_val = feature_bounds[feat_idx]

                if rng.random() < mutation_rate:
                    # Scale noise by feature range
                    feat_range = max_val - min_val
                    noise = rng.normal(0, mutation_sigma * feat_range)
                    new_thresholds[i] = float(np.clip(new_thresholds[i] + noise, min_val, max_val))
                else:
                    # Still clip to ensure threshold is valid for (possibly changed) feature
                    new_thresholds[i] = float(np.clip(new_thresholds[i], min_val, max_val))

            # Mutate tiebreak_feature
            new_tiebreak = self.tiebreak_feature
            if rng.random() < mutation_rate:
                new_tiebreak = int(rng.integers(0, self.n_features))

            # Repair any path constraint violations after mutation
            new_feature_indices, new_thresholds = self._repair_path_constraints(
                new_feature_indices, new_thresholds, feature_bounds, self.n_features, rng
            )

            # Mutate rul_threshold (±10 hours, if present)
            new_rul_threshold = self.rul_threshold
            if self.rul_threshold is not None and rng.random() < mutation_rate:
                noise = int(rng.integers(-10, 11))
                new_rul_threshold = int(np.clip(new_rul_threshold + noise, 5, 100))

            return Chromosome(
                feature_indices=new_feature_indices,
                thresholds=new_thresholds,
                tiebreak_feature=new_tiebreak,
                config_type='simple',
                tree_depth=self.tree_depth,
                n_features=self.n_features,
                early_phase_window=None,  # No early phase in Simple
                rul_threshold=new_rul_threshold
            )

        # --- MEDIUM CONFIG: Mutate fleet split + subtrees ---
        elif self.config_type == 'medium':
            # Mutate fleet variable indices
            n_fleet_features = len(fleet_feature_bounds)
            new_fleet_indices = self.fleet_variable_indices.copy()
            for i in range(len(new_fleet_indices)):
                if rng.random() < mutation_rate:
                    new_fleet_indices[i] = int(rng.integers(0, n_fleet_features))

            # Mutate fleet thresholds
            new_fleet_thresholds = self.fleet_thresholds.copy()
            for i in range(len(new_fleet_thresholds)):
                if rng.random() < mutation_rate:
                    feat_idx = new_fleet_indices[i]
                    min_val, max_val = fleet_feature_bounds[feat_idx]
                    feat_range = max_val - min_val
                    noise = rng.normal(0, mutation_sigma * feat_range)
                    new_fleet_thresholds[i] = float(np.clip(new_fleet_thresholds[i] + noise, min_val, max_val))

            # Recursively mutate subtrees
            new_subtrees = []
            for subtree in self.context_subtrees:
                mutated_subtree = subtree.mutate(
                    mutation_rate=mutation_rate,
                    mutation_sigma=mutation_sigma,
                    feature_bounds=feature_bounds,
                    fleet_feature_bounds=fleet_feature_bounds,
                    rng=rng
                )
                new_subtrees.append(mutated_subtree)

            # Mutate tiebreak_feature
            new_tiebreak = self.tiebreak_feature
            if rng.random() < mutation_rate:
                new_tiebreak = int(rng.integers(0, self.n_features))

            # Mutate rul_threshold (±10 hours, if present)
            new_rul_threshold = self.rul_threshold
            if self.rul_threshold is not None and rng.random() < mutation_rate:
                noise = int(rng.integers(-10, 11))
                new_rul_threshold = int(np.clip(new_rul_threshold + noise, 5, 100))

            return Chromosome(
                feature_indices=[],
                thresholds=[],
                tiebreak_feature=new_tiebreak,
                config_type='medium',
                tree_depth=self.tree_depth,
                n_features=self.n_features,
                fleet_variable_indices=new_fleet_indices,
                fleet_thresholds=new_fleet_thresholds,
                context_subtrees=new_subtrees,
                early_phase_window=None,  # No early phase in Medium
                rul_threshold=new_rul_threshold
            )

        # --- FULL CONFIG: Mutate fleet tree + subtrees + early_phase_window ---
        elif self.config_type == 'full':
            # Mutate fleet variable indices
            n_fleet_features = len(fleet_feature_bounds)
            new_fleet_indices = self.fleet_variable_indices.copy()
            for i in range(len(new_fleet_indices)):
                if rng.random() < mutation_rate:
                    new_fleet_indices[i] = int(rng.integers(0, n_fleet_features))

            # Mutate fleet thresholds
            new_fleet_thresholds = self.fleet_thresholds.copy()
            for i in range(len(new_fleet_thresholds)):
                if rng.random() < mutation_rate:
                    feat_idx = new_fleet_indices[i]
                    min_val, max_val = fleet_feature_bounds[feat_idx]
                    feat_range = max_val - min_val
                    noise = rng.normal(0, mutation_sigma * feat_range)
                    new_fleet_thresholds[i] = float(np.clip(new_fleet_thresholds[i] + noise, min_val, max_val))

            # Recursively mutate subtrees
            new_subtrees = []
            for subtree in self.context_subtrees:
                mutated_subtree = subtree.mutate(
                    mutation_rate=mutation_rate,
                    mutation_sigma=mutation_sigma,
                    feature_bounds=feature_bounds,
                    fleet_feature_bounds=fleet_feature_bounds,
                    rng=rng
                )
                new_subtrees.append(mutated_subtree)

            # Mutate tiebreak_feature
            new_tiebreak = self.tiebreak_feature
            if rng.random() < mutation_rate:
                new_tiebreak = int(rng.integers(0, self.n_features))

            # Mutate early_phase_window (±10 hours, Full config only)
            new_early_phase = self.early_phase_window
            if rng.random() < mutation_rate:
                noise = int(rng.integers(-10, 11))
                new_early_phase = int(np.clip(new_early_phase + noise, 0, 100))

            # Mutate rul_threshold (±10 hours, if present)
            new_rul_threshold = self.rul_threshold
            if self.rul_threshold is not None and rng.random() < mutation_rate:
                noise = int(rng.integers(-10, 11))
                new_rul_threshold = int(np.clip(new_rul_threshold + noise, 5, 100))

            # Repair fleet path constraints (full config has 7 fleet splits)
            new_fleet_indices, new_fleet_thresholds = self._repair_path_constraints(
                new_fleet_indices, new_fleet_thresholds, fleet_feature_bounds,
                len(fleet_feature_bounds), rng
            )

            return Chromosome(
                feature_indices=[],
                thresholds=[],
                tiebreak_feature=new_tiebreak,
                config_type='full',
                tree_depth=self.tree_depth,
                n_features=self.n_features,
                fleet_variable_indices=new_fleet_indices,
                fleet_thresholds=new_fleet_thresholds,
                context_subtrees=new_subtrees,
                early_phase_window=new_early_phase,
                rul_threshold=new_rul_threshold
            )

        else:
            raise ValueError(f"Unknown config_type: {self.config_type}")

    def crossover(
        self,
        other: 'Chromosome',
        crossover_rate: float = 0.70,
        feature_bounds: List[Tuple[float, float]] = None,
        fleet_feature_bounds: List[Tuple[float, float]] = None,
        rng: np.random.Generator = None
    ) -> Tuple['Chromosome', 'Chromosome']:
        """Perform hybrid crossover with another chromosome to create two offspring.

        Supports hierarchical crossover for Medium/Full configs by recursively
        crossing over subtrees while crossing over fleet-level genes independently.

        Crossover strategies:
        - feature_indices: Uniform crossover (per-gene 50% swap)
        - thresholds: Blend crossover (BLX-α with α=0.5)
        - fleet_variable_indices: Uniform crossover
        - fleet_thresholds: Blend crossover
        - context_subtrees: Recursive crossover
        - tiebreak_feature: Random selection from one parent
        - early_phase_window: Arithmetic mean ± noise (Full config only)

        Args:
            other: Second parent chromosome
            crossover_rate: Probability of performing crossover (default: 0.70)
                           If crossover doesn't occur, returns clones of parents
            feature_bounds: Bounds for per-aircraft features (default: standard bounds)
            fleet_feature_bounds: Bounds for fleet features (default: normalized [0, 1])
            rng: Random number generator for determinism

        Returns:
            Tuple of two offspring chromosomes

        Raises:
            ValueError: If chromosomes have incompatible structures
        """
        if rng is None:
            rng = np.random.default_rng()

        if feature_bounds is None:
            # Use self's feature_bounds to stay consistent with validation
            feature_bounds = self.feature_bounds

        if fleet_feature_bounds is None:
            fleet_feature_bounds = [(0, 1) for _ in range(7)]  # 7 fleet features

        # Validate compatibility
        if self.config_type != other.config_type:
            raise ValueError(
                f"Cannot crossover chromosomes with different config types: "
                f"{self.config_type} vs {other.config_type}"
            )
        if self.tree_depth != other.tree_depth:
            raise ValueError(
                f"Cannot crossover chromosomes with different tree depths: "
                f"{self.tree_depth} vs {other.tree_depth}"
            )
        if self.n_features != other.n_features:
            raise ValueError(
                f"Cannot crossover chromosomes with different feature counts: "
                f"{self.n_features} vs {other.n_features}"
            )

        # With probability (1 - crossover_rate), return clones
        if rng.random() > crossover_rate:
            # Use mutate with rate=0 to create proper clones
            return (
                self.mutate(mutation_rate=0.0, feature_bounds=feature_bounds, fleet_feature_bounds=fleet_feature_bounds, rng=rng),
                other.mutate(mutation_rate=0.0, feature_bounds=feature_bounds, fleet_feature_bounds=fleet_feature_bounds, rng=rng)
            )

        # --- SIMPLE CONFIG: Crossover flat tree ---
        if self.config_type == 'simple':
            n_splits = 2 ** self.tree_depth - 1  # 15

            # Uniform crossover for feature_indices (50% swap per gene)
            mask = rng.random(n_splits) < 0.5
            child1_features = np.where(mask, self.feature_indices, other.feature_indices)
            child2_features = np.where(mask, other.feature_indices, self.feature_indices)

            # Blend crossover (BLX-α) for thresholds with α=0.5
            alpha = 0.5
            child1_thresholds = []
            child2_thresholds = []

            for i, (t1, t2) in enumerate(zip(self.thresholds, other.thresholds)):
                # Get feature-specific bounds for each child
                feat_idx_c1 = child1_features[i]
                feat_idx_c2 = child2_features[i]
                min_val_c1, max_val_c1 = feature_bounds[feat_idx_c1]
                min_val_c2, max_val_c2 = feature_bounds[feat_idx_c2]

                t_min = min(t1, t2)
                t_max = max(t1, t2)
                t_range = t_max - t_min

                # Extend interval by α in both directions
                lower = t_min - alpha * t_range
                upper = t_max + alpha * t_range

                # Sample offspring thresholds from extended interval
                offspring1 = rng.uniform(lower, upper)
                offspring2 = rng.uniform(lower, upper)

                # Clip to child-specific feature bounds
                child1_thresholds.append(float(np.clip(offspring1, min_val_c1, max_val_c1)))
                child2_thresholds.append(float(np.clip(offspring2, min_val_c2, max_val_c2)))

            # Random selection for tiebreak_feature
            child1_tiebreak = self.tiebreak_feature if rng.random() < 0.5 else other.tiebreak_feature
            child2_tiebreak = other.tiebreak_feature if rng.random() < 0.5 else self.tiebreak_feature

            # Repair path constraints for both children
            child1_features_repaired, child1_thresholds_repaired = self._repair_path_constraints(
                child1_features.tolist(), child1_thresholds, feature_bounds, self.n_features, rng
            )
            child2_features_repaired, child2_thresholds_repaired = self._repair_path_constraints(
                child2_features.tolist(), child2_thresholds, feature_bounds, self.n_features, rng
            )

            # Conditional rul_threshold crossover (if present in both parents)
            if self.rul_threshold is not None and other.rul_threshold is not None:
                mean_threshold = (self.rul_threshold + other.rul_threshold) / 2.0
                noise1 = rng.integers(-10, 11)
                noise2 = rng.integers(-10, 11)
                child1_rul = int(np.clip(mean_threshold + noise1, 5, 100))
                child2_rul = int(np.clip(mean_threshold + noise2, 5, 100))
            else:
                child1_rul = self.rul_threshold
                child2_rul = other.rul_threshold

            return (
                Chromosome(
                    feature_indices=child1_features_repaired,
                    thresholds=child1_thresholds_repaired,
                    tiebreak_feature=child1_tiebreak,
                    config_type='simple',
                    tree_depth=self.tree_depth,
                    n_features=self.n_features,
                    early_phase_window=None,
                    rul_threshold=child1_rul
                ),
                Chromosome(
                    feature_indices=child2_features_repaired,
                    thresholds=child2_thresholds_repaired,
                    tiebreak_feature=child2_tiebreak,
                    config_type='simple',
                    tree_depth=self.tree_depth,
                    n_features=self.n_features,
                    early_phase_window=None,
                    rul_threshold=child2_rul
                )
            )

        # --- MEDIUM/FULL CONFIG: Hierarchical crossover ---
        elif self.config_type in ['medium', 'full']:
            n_fleet_splits = len(self.fleet_variable_indices)

            # Uniform crossover for fleet_variable_indices
            mask = rng.random(n_fleet_splits) < 0.5
            child1_fleet_indices = np.where(mask, self.fleet_variable_indices, other.fleet_variable_indices)
            child2_fleet_indices = np.where(mask, other.fleet_variable_indices, self.fleet_variable_indices)

            # Blend crossover for fleet_thresholds
            alpha = 0.5
            child1_fleet_thresholds = []
            child2_fleet_thresholds = []

            for i, (t1, t2) in enumerate(zip(self.fleet_thresholds, other.fleet_thresholds)):
                feat_idx = child1_fleet_indices[i]
                min_val, max_val = fleet_feature_bounds[feat_idx]

                t_min = min(t1, t2)
                t_max = max(t1, t2)
                t_range = t_max - t_min

                lower = t_min - alpha * t_range
                upper = t_max + alpha * t_range

                offspring1 = rng.uniform(lower, upper)
                offspring2 = rng.uniform(lower, upper)

                child1_fleet_thresholds.append(float(np.clip(offspring1, min_val, max_val)))
                child2_fleet_thresholds.append(float(np.clip(offspring2, min_val, max_val)))

            # Repair fleet path constraints for full config (which has 7 fleet splits)
            if self.config_type == 'full':
                child1_fleet_indices_repaired, child1_fleet_thresholds_repaired = self._repair_path_constraints(
                    child1_fleet_indices.tolist(), child1_fleet_thresholds,
                    fleet_feature_bounds, len(fleet_feature_bounds), rng
                )
                child2_fleet_indices_repaired, child2_fleet_thresholds_repaired = self._repair_path_constraints(
                    child2_fleet_indices.tolist(), child2_fleet_thresholds,
                    fleet_feature_bounds, len(fleet_feature_bounds), rng
                )
                child1_fleet_indices = child1_fleet_indices_repaired
                child2_fleet_indices = child2_fleet_indices_repaired
                child1_fleet_thresholds = child1_fleet_thresholds_repaired
                child2_fleet_thresholds = child2_fleet_thresholds_repaired
            else:
                # Medium config only has 1 fleet split, no path constraints to repair
                child1_fleet_indices = child1_fleet_indices.tolist()
                child2_fleet_indices = child2_fleet_indices.tolist()

            # Recursively crossover subtrees (already have path repair in simple config crossover)
            child1_subtrees = []
            child2_subtrees = []
            for subtree1, subtree2 in zip(self.context_subtrees, other.context_subtrees):
                offspring1, offspring2 = subtree1.crossover(
                    subtree2,
                    crossover_rate=crossover_rate,
                    feature_bounds=feature_bounds,
                    fleet_feature_bounds=fleet_feature_bounds,
                    rng=rng
                )
                child1_subtrees.append(offspring1)
                child2_subtrees.append(offspring2)

            # Random selection for tiebreak_feature
            child1_tiebreak = self.tiebreak_feature if rng.random() < 0.5 else other.tiebreak_feature
            child2_tiebreak = other.tiebreak_feature if rng.random() < 0.5 else self.tiebreak_feature

            # Conditional early_phase_window crossover (Full config only)
            if self.config_type == 'full':
                mean_window = (self.early_phase_window + other.early_phase_window) / 2.0
                noise1 = rng.integers(-10, 11)
                noise2 = rng.integers(-10, 11)
                child1_window = int(np.clip(mean_window + noise1, 0, 100))
                child2_window = int(np.clip(mean_window + noise2, 0, 100))
            else:
                child1_window = None
                child2_window = None

            # Conditional rul_threshold crossover (if present in both parents)
            if self.rul_threshold is not None and other.rul_threshold is not None:
                mean_threshold = (self.rul_threshold + other.rul_threshold) / 2.0
                noise1 = rng.integers(-10, 11)
                noise2 = rng.integers(-10, 11)
                child1_rul = int(np.clip(mean_threshold + noise1, 5, 100))
                child2_rul = int(np.clip(mean_threshold + noise2, 5, 100))
            else:
                child1_rul = self.rul_threshold
                child2_rul = other.rul_threshold

            return (
                Chromosome(
                    feature_indices=[],
                    thresholds=[],
                    tiebreak_feature=child1_tiebreak,
                    config_type=self.config_type,
                    tree_depth=self.tree_depth,
                    n_features=self.n_features,
                    fleet_variable_indices=child1_fleet_indices,  # Already a list after repair
                    fleet_thresholds=child1_fleet_thresholds,
                    context_subtrees=child1_subtrees,
                    early_phase_window=child1_window,
                    rul_threshold=child1_rul
                ),
                Chromosome(
                    feature_indices=[],
                    thresholds=[],
                    tiebreak_feature=child2_tiebreak,
                    config_type=self.config_type,
                    tree_depth=self.tree_depth,
                    n_features=self.n_features,
                    fleet_variable_indices=child2_fleet_indices,  # Already a list after repair
                    fleet_thresholds=child2_fleet_thresholds,
                    context_subtrees=child2_subtrees,
                    early_phase_window=child2_window,
                    rul_threshold=child2_rul
                )
            )

        else:
            raise ValueError(f"Unknown config_type: {self.config_type}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Chromosome(depth={self.tree_depth}, "
            f"splits={self.n_splits}, "
            f"leaves={self.n_leaves}, "
            f"genes={self.n_genes})"
        )
