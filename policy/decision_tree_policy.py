"""
Decision Tree Policy for Aircraft Maintenance Scheduling

Implements GA-optimized decision tree with NMC bypass and three-sweep greedy
action adjudication. Separates bucket classification (learned) from action
assignment (deterministic) for interpretability and constraint satisfaction.

Architecture:
1. NMC Bypass: Aircraft with status='NMC' → Bucket 0 (deterministic maintenance)
2. Decision Tree: FMC aircraft → Buckets 1-4 based on learned splits
3. Three Sweeps: Greedy action assignment (mandatory → flying → preventive)

Reference: docs/Planning/GREEDY_SWEEP_ADJUDICATION.md
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from policy.base_policy import BasePolicy
from policy.chromosome import Chromosome
from policy.state_encoder import StateEncoder


class DecisionTreePolicy(BasePolicy):
    """
    GA-optimized decision tree policy with bucket classification and greedy sweeps.

    NEW: Direct bucket mapping from tree leaves (0-7) to buckets (1-8).
    Supports three configurations: Simple (flat), Medium (2-context), Full (4-context).

    Bucket Definitions (Direct Mapping):
        - Bucket 0 (NMC): Aircraft with status='NMC' → mandatory maintenance (deterministic)
        - Buckets 1-8 (FMC): Direct mapping from decision tree leaves
            - Lower buckets (1-2): Likely preventive maintenance or early phase
            - Middle buckets (3-5): Mixed eligibility
            - Higher buckets (6-8): Preferred for flying missions

    Early Phase Enforcement (Full config only):
        - Buckets 1-2: Subject to early_phase_window (0-100 hours, binding)
        - Buckets 3-8: NO early phase (purely reactive)
        - Simple/Medium: NO early phase at all

    Example:
        >>> encoder = StateEncoder('config/features/simple_dt.yaml')
        >>> chromosome = Chromosome.random(tree_depth=3, n_features=5, config_type='simple')
        >>> policy = DecisionTreePolicy(chromosome=chromosome, encoder=encoder)
        >>> actions = policy.decide(state)
    """

    # Fleet leaf to context mapping for Full configuration (4 contexts)
    # Distribution: Ctx1=12.5%, Ctx2=37.5%, Ctx3=37.5%, Ctx4=12.5%
    FLEET_LEAF_TO_CONTEXT = [0, 1, 1, 1, 2, 2, 2, 3]  # 0-indexed

    def __init__(
        self,
        chromosome=None,
        encoder=None,
        chromosome_path=None,
        feature_config_path=None,
        rul_threshold: int = 100,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize decision tree policy.

        Can be initialized two ways:

        1. With objects (for programmatic use, testing, GA):
           policy = DecisionTreePolicy(chromosome=chrom, encoder=enc)

        2. With paths (for factory use, scripts):
           policy = DecisionTreePolicy(
               chromosome_path='chromosomes/baseline_dt.json',
               feature_config_path='config/features/simple_dt.yaml'
           )

        Args:
            chromosome: Chromosome object (mutually exclusive with chromosome_path)
            encoder: StateEncoder object (mutually exclusive with feature_config_path)
            chromosome_path: Path to chromosome JSON file
            feature_config_path: Path to feature config YAML file
            rul_threshold: RUL threshold for preventive maintenance (default: 100)
            verbose: Enable detailed logging
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If neither objects nor paths provided, or if incompatible
        """
        super().__init__(verbose=verbose, **kwargs)

        # Load chromosome from path or use provided object
        if chromosome is None and chromosome_path is None:
            raise ValueError(
                "Must provide either 'chromosome' object or 'chromosome_path'. "
                "Example: DecisionTreePolicy(chromosome_path='chromosomes/baseline_dt.json')"
            )

        if chromosome is not None and chromosome_path is not None:
            raise ValueError(
                "Provide either 'chromosome' object OR 'chromosome_path', not both"
            )

        if chromosome is None:
            if self.verbose:
                print(f"Loading chromosome from: {chromosome_path}")
            chromosome = Chromosome.from_json(chromosome_path)

        # Load encoder from path or use provided object
        if encoder is None and feature_config_path is None:
            raise ValueError(
                "Must provide either 'encoder' object or 'feature_config_path'. "
                "Example: DecisionTreePolicy(feature_config_path='config/features/simple_dt.yaml')"
            )

        if encoder is not None and feature_config_path is not None:
            raise ValueError(
                "Provide either 'encoder' object OR 'feature_config_path', not both"
            )

        if encoder is None:
            if self.verbose:
                print(f"Loading feature encoder from: {feature_config_path}")
            encoder = StateEncoder(feature_config_path, verbose=verbose)

        # Store initialized objects
        self.chromosome = chromosome
        self.encoder = encoder
        self.rul_threshold = rul_threshold

        # Validate chromosome compatibility with encoder
        if self.chromosome.n_features != self.encoder.n_per_aircraft:
            raise ValueError(
                f"Chromosome n_features ({self.chromosome.n_features}) must match "
                f"encoder per-aircraft features ({self.encoder.n_per_aircraft})"
            )

        if self.verbose:
            print(f"DecisionTreePolicy initialized:")
            print(f"  Tree depth: {self.chromosome.tree_depth}")
            print(f"  Leaves: {self.chromosome.n_leaves}")
            print(f"  Features: {self.chromosome.n_features}")
            print(f"  Tiebreak feature: {self.chromosome.tiebreak_feature}")
            print(f"  Early phase window: {self.chromosome.early_phase_window} hours")

    def decide(self, state: Dict[str, Any]) -> Dict[int, str]:
        """
        Generate actions using decision tree + three-sweep algorithm.

        Algorithm:
            1. Extract features from state
            2. Classify aircraft into buckets (0-4)
            3. Sweep 1: Mandatory maintenance (NMC aircraft)
            4. Sweep 2: Reverse sweep for flying (buckets 4 → 3 → 2 → 1)
            5. Sweep 3: Forward sweep for preventive (buckets 1 → 2 → 3 → 4)
            6. Fill remaining with 'hold'

        Args:
            state: Simulation state dictionary

        Returns:
            Dictionary mapping aircraft_id (0-7) -> action_string
        """
        # Extract features from encoder
        raw_features = self.encoder.encode(state)

        # Preprocess features based on encoder configuration
        # For medium/full config: raw_features is 1D (flattened per-aircraft + fleet)
        # For simple config: raw_features is 2D (8, n_per_aircraft)
        n_per_aircraft = self.encoder.n_per_aircraft
        n_fleet = self.encoder.n_fleet

        if n_fleet > 0 and raw_features.ndim == 1:
            # Medium/Full: Reshape per-aircraft features to 2D for sweep methods
            n_per_aircraft_total = 8 * n_per_aircraft
            per_aircraft_features = raw_features[:n_per_aircraft_total].reshape(8, n_per_aircraft)
        else:
            # Simple: Already 2D
            per_aircraft_features = raw_features

        # Classify aircraft into buckets (uses raw_features internally to handle fleet)
        bucket_assignments = self._classify_aircraft(state['aircraft'], raw_features)

        # Initialize action dictionary
        actions = {}

        # Track resource consumption
        resources = self._initialize_resources(state)

        # Sweep 1: Mandatory maintenance (NMC → Bucket 0)
        actions = self._sweep_mandatory_maintenance(
            state['aircraft'],
            bucket_assignments,
            resources,
            actions
        )

        # Sweep 2: Reverse sweep for flying (Buckets 4 → 3 → 2 → 1)
        # NOTE: After step() returns state, forecast[0] is the mission just completed.
        # forecast[1] is what the NEXT step() will use, so we read [1] for planning.
        mission_forecast = state['mission_forecast']
        if len(mission_forecast) > 1:
            todays_requirement = mission_forecast[1].required_aircraft
        else:
            todays_requirement = mission_forecast[0].required_aircraft  # Day 1 fallback

        actions = self._sweep_flying_assignment(
            state['aircraft'],
            bucket_assignments,
            todays_requirement,
            per_aircraft_features,  # Always 2D (8, n_per_aircraft)
            actions
        )

        # Sweep 3: Forward sweep for preventive maintenance (Buckets 1 → 2 → 3 → 4)
        actions = self._sweep_preventive_maintenance(
            state['aircraft'],
            bucket_assignments,
            state,
            resources,
            actions
        )

        # Fill remaining aircraft with 'hold'
        for aircraft_id in range(8):
            if aircraft_id not in actions:
                actions[aircraft_id] = 'hold'

        return self._validate_actions(actions)

    def _classify_aircraft(
        self,
        aircraft_list: List,
        features: np.ndarray
    ) -> Dict[int, int]:
        """
        Classify aircraft into buckets using decision tree.

        Supports three configurations:
        - Simple: Direct per-aircraft tree traversal
        - Medium/Full: Fleet context tree → per-aircraft subtree traversal

        Args:
            aircraft_list: List of Aircraft objects
            features: Feature matrix of shape (8, n_per_aircraft) OR (8, n_per_aircraft + n_fleet)
                     depending on encoder configuration

        Returns:
            Dictionary mapping aircraft_id -> bucket (0-8)
                Bucket 0: NMC (mandatory maintenance)
                Buckets 1-8: Direct mapping from leaves (FMC aircraft)
        """
        bucket_assignments = {}

        # Extract fleet features if hierarchical config (Medium/Full)
        fleet_features = None
        n_per_aircraft = self.encoder.n_per_aircraft
        n_fleet = self.encoder.n_fleet

        if self.chromosome.config_type in ['medium', 'full'] and n_fleet > 0:
            # Encoder returns 1D array when fleet features present:
            # [per_aircraft_0, per_aircraft_1, ..., per_aircraft_7, fleet_features]
            # Shape: (8 * n_per_aircraft + n_fleet,)
            n_per_aircraft_total = 8 * n_per_aircraft

            # Extract fleet features from end of flattened array
            fleet_features = features[n_per_aircraft_total:]  # Shape: (n_fleet,)

            # Reshape per-aircraft features back to (8, n_per_aircraft)
            per_aircraft_features = features[:n_per_aircraft_total].reshape(8, n_per_aircraft)
        else:
            # Simple config: features already in (8, n_per_aircraft) shape
            per_aircraft_features = features

        for aircraft in aircraft_list:
            aircraft_id = aircraft.id

            # NMC bypass: Bucket 0 (deterministic)
            if aircraft.status == 'NMC':
                bucket_assignments[aircraft_id] = 0
                continue

            # FMC aircraft: Traverse decision tree
            aircraft_features = per_aircraft_features[aircraft_id, :]  # Shape: (n_per_aircraft,)
            bucket = self._traverse_tree(aircraft_features, fleet_features)
            bucket_assignments[aircraft_id] = bucket

        return bucket_assignments

    def _traverse_tree(
        self,
        per_aircraft_features: np.ndarray,
        fleet_features: np.ndarray = None
    ) -> int:
        """
        Traverse decision tree to determine bucket assignment.

        Supports three configurations:
        - Simple: Direct traversal of per-aircraft tree → bucket
        - Medium/Full: Fleet tree → context selection → per-aircraft subtree → bucket

        Tree structure (breadth-first indexing):
        - Root = 0
        - Left child of node i = 2*i + 1
        - Right child of node i = 2*i + 2
        - Leaf index = node_index - (n_splits)

        Direct bucket mapping (NEW):
        - Leaves 0-7 map directly to buckets 1-8
        - No bucket_assignments array needed

        Args:
            per_aircraft_features: Per-aircraft feature vector (shape: n_per_aircraft)
            fleet_features: Fleet feature vector (shape: n_fleet), required for Medium/Full

        Returns:
            Bucket assignment (1-8 for FMC aircraft)
        """
        # --- SIMPLE CONFIG: Direct per-aircraft tree traversal ---
        if self.chromosome.config_type == 'simple':
            node_idx = 0
            n_splits = 2 ** self.chromosome.tree_depth - 1  # 15

            while node_idx < n_splits:
                feature_idx = self.chromosome.feature_indices[node_idx]
                threshold = self.chromosome.thresholds[node_idx]
                feature_value = per_aircraft_features[feature_idx]

                if feature_value < threshold:
                    node_idx = 2 * node_idx + 1  # Left child
                else:
                    node_idx = 2 * node_idx + 2  # Right child

            # Direct bucket mapping: leaf_idx (0-7) → bucket (1-8)
            leaf_idx = node_idx - n_splits
            bucket = leaf_idx + 1

            return bucket

        # --- MEDIUM/FULL CONFIG: Hierarchical traversal ---
        elif self.chromosome.config_type in ['medium', 'full']:
            if fleet_features is None:
                raise ValueError(f"{self.chromosome.config_type} config requires fleet_features")

            # Step 1: Traverse fleet tree to determine context
            node_idx = 0
            n_fleet_splits = len(self.chromosome.fleet_variable_indices)

            while node_idx < n_fleet_splits:
                fleet_var_idx = self.chromosome.fleet_variable_indices[node_idx]
                fleet_threshold = self.chromosome.fleet_thresholds[node_idx]
                fleet_value = fleet_features[fleet_var_idx]

                if fleet_value < fleet_threshold:
                    node_idx = 2 * node_idx + 1  # Left child
                else:
                    node_idx = 2 * node_idx + 2  # Right child

            # Determine context index from fleet leaf using mapping
            fleet_leaf_idx = node_idx - n_fleet_splits
            # Full config uses FLEET_LEAF_TO_CONTEXT mapping for [1,2,2,2,3,3,3,4] distribution
            # Simple/Medium configs have fewer leaves, but mapping still applies (wraps to available contexts)
            context_idx = self.FLEET_LEAF_TO_CONTEXT[fleet_leaf_idx % len(self.FLEET_LEAF_TO_CONTEXT)]

            # Step 2: Select appropriate per-aircraft subtree
            subtree = self.chromosome.context_subtrees[context_idx]

            # Step 3: Traverse per-aircraft subtree
            node_idx = 0
            n_splits = 2 ** subtree.tree_depth - 1  # 15

            while node_idx < n_splits:
                feature_idx = subtree.feature_indices[node_idx]
                threshold = subtree.thresholds[node_idx]
                feature_value = per_aircraft_features[feature_idx]

                if feature_value < threshold:
                    node_idx = 2 * node_idx + 1
                else:
                    node_idx = 2 * node_idx + 2

            # Direct bucket mapping: leaf_idx (0-7) → bucket (1-8)
            leaf_idx = node_idx - n_splits
            bucket = leaf_idx + 1

            return bucket

        else:
            raise ValueError(f"Unknown config_type: {self.chromosome.config_type}")

    def _initialize_resources(self, state: Dict) -> Dict[str, int]:
        """Initialize resource availability tracking."""
        # Handle tokens_available as integer (single pool) or dict (if dict provided)
        tokens_avail = state.get('tokens_available', 0)
        if isinstance(tokens_avail, dict):
            tokens = tokens_avail.get('total', 0)
        else:
            tokens = int(tokens_avail) if tokens_avail else 0

        # Handle maintenance_slots as dict or default values
        slots = state.get('maintenance_slots', {})

        return {
            'preventive_slots': slots.get('preventive_available', 2),
            'phase_slots': slots.get('phase_available', 1),
            'tokens': tokens  # Single token pool (environment handles allocation)
        }

    def _sweep_mandatory_maintenance(
        self,
        aircraft_list: List,
        buckets: Dict[int, int],
        resources: Dict[str, int],
        actions: Dict[int, str]
    ) -> Dict[int, str]:
        """
        Sweep 1: Assign mandatory maintenance to NMC aircraft (Bucket 0).

        Args:
            aircraft_list: List of Aircraft objects
            buckets: Bucket assignments
            resources: Available resources
            actions: Current actions dictionary

        Returns:
            Updated actions dictionary
        """
        for aircraft in aircraft_list:
            if buckets.get(aircraft.id) != 0:
                continue  # Skip non-NMC aircraft

            aircraft_id = aircraft.id
            nmc_reason = aircraft.nmc_reason

            if nmc_reason == 'nmc_rul':
                # Reactive maintenance needed
                if resources['preventive_slots'] > 0 and resources['tokens'] > 0:
                    actions[aircraft_id] = 'reactive_maintain'
                    resources['preventive_slots'] -= 1
                    resources['tokens'] -= 1
                else:
                    actions[aircraft_id] = 'hold'  # Stuck until resources available

            elif nmc_reason == 'nmc_minor_phase':
                # Minor phase maintenance needed
                if resources['phase_slots'] > 0 and resources['tokens'] > 0:
                    actions[aircraft_id] = 'minor_phase_maintain'
                    resources['phase_slots'] -= 1
                    resources['tokens'] -= 1
                else:
                    actions[aircraft_id] = 'hold'

            elif nmc_reason == 'nmc_major_phase':
                # Major phase maintenance needed
                if resources['phase_slots'] > 0 and resources['tokens'] > 0:
                    actions[aircraft_id] = 'major_phase_maintain'
                    resources['phase_slots'] -= 1
                    resources['tokens'] -= 1
                else:
                    actions[aircraft_id] = 'hold'

            else:
                # Unknown NMC reason
                actions[aircraft_id] = 'hold'

        return actions

    def _sweep_flying_assignment(
        self,
        aircraft_list: List,
        buckets: Dict[int, int],
        mission_requirement: int,
        features: np.ndarray,
        actions: Dict[int, str]
    ) -> Dict[int, str]:
        """
        Sweep 2: Reverse sweep for flying assignment (Buckets 8 → 7 → ... → 2 → 1).

        Note: With direct bucket mapping, now sweeps buckets 8 down to 1 (not 4 to 1).

        Args:
            aircraft_list: List of Aircraft objects
            buckets: Bucket assignments (0-8)
            mission_requirement: Number of aircraft required to fly today
            features: Feature matrix (8, n_per_aircraft) OR (8, n_per_aircraft + n_fleet)
            actions: Current actions dictionary

        Returns:
            Updated actions dictionary
        """
        flying_count = 0

        # Reverse sweep: highest priority bucket first (8 → 1)
        for bucket_num in range(8, 0, -1):  # 8, 7, 6, 5, 4, 3, 2, 1
            # Get aircraft in this bucket
            aircraft_in_bucket = [
                ac for ac in aircraft_list
                if buckets.get(ac.id) == bucket_num
            ]

            # Sort by tiebreaker feature (descending)
            aircraft_in_bucket = self._sort_by_tiebreaker(
                aircraft_in_bucket,
                features
            )

            # Assign fly actions
            for aircraft in aircraft_in_bucket:
                if flying_count >= mission_requirement:
                    break  # Mission requirement met

                # Only fly if not already assigned an action
                if aircraft.id not in actions:
                    actions[aircraft.id] = 'fly'
                    flying_count += 1

            if flying_count >= mission_requirement:
                break  # Stop reverse sweep

        # Log warning if mission not met
        if flying_count < mission_requirement and self.verbose:
            print(f"WARNING: Mission not met: {flying_count}/{mission_requirement} aircraft flying")

        return actions

    def _sort_by_tiebreaker(
        self,
        aircraft_list: List,
        features: np.ndarray
    ) -> List:
        """
        Sort aircraft by tiebreaker feature value (descending).

        Handles both Simple and hierarchical (Medium/Full) feature matrices:
        - Simple: features shape (8, n_per_aircraft)
        - Medium/Full: features shape (8, n_per_aircraft + n_fleet)

        Args:
            aircraft_list: List of Aircraft objects to sort
            features: Feature matrix (8, n_per_aircraft) OR (8, n_per_aircraft + n_fleet)

        Returns:
            Sorted list of Aircraft objects
        """
        tiebreak_feature_idx = self.chromosome.tiebreak_feature

        # Extract tiebreaker values for aircraft in list
        # NOTE: Tiebreak feature is always a per-aircraft feature index (0-4)
        tiebreaker_values = {}
        for aircraft in aircraft_list:
            tiebreaker_values[aircraft.id] = features[aircraft.id, tiebreak_feature_idx]

        # Sort descending by tiebreaker value
        sorted_aircraft = sorted(
            aircraft_list,
            key=lambda ac: tiebreaker_values[ac.id],
            reverse=True
        )

        return sorted_aircraft

    def _sweep_preventive_maintenance(
        self,
        aircraft_list: List,
        buckets: Dict[int, int],
        state: Dict,
        resources: Dict[str, int],
        actions: Dict[int, str]
    ) -> Dict[int, str]:
        """
        Sweep 3: Forward sweep for preventive maintenance (Buckets 1 → 2 → 3 → 4 → ... → 8).

        Note: Bucket range now extends to 8 (not 4) with direct bucket mapping.

        Args:
            aircraft_list: List of Aircraft objects
            buckets: Bucket assignments (0-8)
            state: Full simulation state
            resources: Available resources
            actions: Current actions dictionary

        Returns:
            Updated actions dictionary
        """
        # Forward sweep: preventive-eligible bucket first (buckets 1-8)
        for bucket_num in range(1, 9):  # 1 through 8
            aircraft_in_bucket = [
                ac for ac in aircraft_list
                if buckets.get(ac.id) == bucket_num
            ]

            for aircraft in aircraft_in_bucket:
                # Skip if already assigned action
                if aircraft.id in actions:
                    continue

                # Consider preventive maintenance
                if self._should_do_preventive(aircraft, state, resources):
                    actions[aircraft.id] = 'preventive_maintain'
                    resources['preventive_slots'] -= 1
                    resources['tokens'] -= 1
                    continue

                # Consider early phase maintenance (bucket-conditional)
                phase_type = self._should_do_early_phase(
                    aircraft,
                    state,
                    resources,
                    bucket=bucket_num  # Pass bucket for enforcement check
                )
                if phase_type is not None:
                    if phase_type == 'major':
                        actions[aircraft.id] = 'major_phase_maintain'
                        resources['phase_slots'] -= 1
                        resources['tokens'] -= 1
                    elif phase_type == 'minor':
                        actions[aircraft.id] = 'minor_phase_maintain'
                        resources['phase_slots'] -= 1
                        resources['tokens'] -= 1
                    continue

        return actions

    def _should_do_preventive(
        self,
        aircraft,
        state: Dict,
        resources: Dict[str, int]
    ) -> bool:
        """
        Determine if aircraft should receive preventive maintenance.

        Conditions (all must be true):
        1. observed_rul < rul_threshold
        2. Preventive slots available
        3. Preventive tokens available
        4. Not too close to quarter end (> 10 days remaining)

        Args:
            aircraft: Aircraft object
            state: Simulation state
            resources: Available resources

        Returns:
            True if preventive maintenance should be performed
        """
        # Check RUL threshold
        if aircraft.observed_rul >= self.rul_threshold:
            return False

        # Check resource availability
        if resources['preventive_slots'] <= 0:
            return False

        if resources['tokens'] <= 0:
            return False

        # Check timing (don't deplete resources too close to quarter end)
        fiscal_day = state.get('fiscal_day')
        if fiscal_day:
            days_until_quarter_end = 90 - (fiscal_day.days_into_fy % 90)
            if days_until_quarter_end <= 10:
                return False

        return True

    def _should_do_early_phase(
        self,
        aircraft,
        state: Dict,
        resources: Dict[str, int],
        bucket: int = None
    ) -> str:
        """
        Determine if aircraft should receive early phase maintenance.

        BINDING ENFORCEMENT (Full config only):
        - Early phase window only applies to aircraft in buckets 1-2
        - Buckets 3-8: NO early phase (purely reactive)
        - Simple/Medium configs: NO early phase at all (early_phase_window = None)

        Checks if aircraft is within early_phase_window of phase trigger.
        Major phase takes precedence over minor.

        Args:
            aircraft: Aircraft object
            state: Simulation state
            resources: Available resources
            bucket: Aircraft bucket assignment (optional, for enforcement check)

        Returns:
            'major', 'minor', or None
        """
        # No early phase for Simple/Medium configs
        if self.chromosome.early_phase_window is None:
            return None

        # Bucket-conditional enforcement: Only buckets 1-2
        if bucket is not None and bucket not in [1, 2]:
            return None  # Buckets 3-8 do NOT receive early phase

        early_window = self.chromosome.early_phase_window

        # Calculate hours to phase triggers
        hours_to_major = 500 - aircraft.hours_since_major_phase
        hours_to_minor = 250 - aircraft.hours_since_minor_phase

        # Check major phase (takes precedence)
        if 0 < hours_to_major <= early_window:
            if resources['phase_slots'] > 0 and resources['tokens'] > 0:
                return 'major'

        # Check minor phase
        if 0 < hours_to_minor <= early_window:
            if resources['phase_slots'] > 0 and resources['tokens'] > 0:
                return 'minor'

        return None

    def get_name(self) -> str:
        """Return policy name."""
        return "DecisionTreePolicy"

    def get_info(self) -> Dict[str, Any]:
        """Return policy metadata."""
        info = super().get_info()
        info.update({
            'tree_depth': self.chromosome.tree_depth,
            'n_leaves': self.chromosome.n_leaves,
            'n_features': self.chromosome.n_features,
            'tiebreak_feature': self.chromosome.tiebreak_feature,
            'early_phase_window': self.chromosome.early_phase_window,
            'rul_threshold': self.rul_threshold,
            'encoder_config': self.encoder.config_path.name
        })
        return info

    def get_chromosome(self) -> Chromosome:
        """Return the chromosome (for GA fitness evaluation)."""
        return self.chromosome

    def set_chromosome(self, chromosome: Chromosome) -> None:
        """Update chromosome (for GA optimization)."""
        if chromosome.n_features != self.chromosome.n_features:
            raise ValueError(
                f"Chromosome n_features mismatch: expected {self.chromosome.n_features}, "
                f"got {chromosome.n_features}"
            )
        self.chromosome = chromosome

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DecisionTreePolicy(depth={self.chromosome.tree_depth}, "
            f"features={self.chromosome.n_features}, "
            f"buckets=4)"
        )
