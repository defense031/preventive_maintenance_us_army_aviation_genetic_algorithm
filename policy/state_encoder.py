"""
State Encoder for Feature Extraction

Converts raw simulation state (104 variables) into normalized feature vectors
based on configurable feature definitions. Supports simple, medium, and full
feature sets for progressive GA optimization.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from simulation.aircraft import Aircraft


class StateEncoder:
    """
    Extract and normalize features from simulation state.

    Loads feature configuration from YAML and applies transformations
    to produce feature vectors suitable for decision tree input.

    Example:
        >>> encoder = StateEncoder('config/features/simple.yaml')
        >>> features = encoder.encode(state)  # Returns (8, 3) array
        >>> feature_names = encoder.get_feature_names()
    """

    def __init__(self, feature_config_path: str, verbose: bool = False):
        """
        Initialize encoder with feature configuration.

        Args:
            feature_config_path: Path to feature config YAML file
            verbose: Enable detailed logging

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is malformed
        """
        self.verbose = verbose
        self.config_path = Path(feature_config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Feature config not found: {feature_config_path}"
            )

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract feature definitions
        self.per_aircraft_features = self.config.get('per_aircraft_features', [])
        self.fleet_features = self.config.get('fleet_features', [])
        self.mission_features = self.config.get('mission_features', [])
        self.temporal_features = self.config.get('temporal_features', [])

        # Normalization settings
        norm_config = self.config.get('normalization', {})
        self.norm_method = norm_config.get('method', 'minmax')
        self.handle_missing = norm_config.get('handle_missing', 'zero')
        self.clip_outliers = norm_config.get('clip_outliers', True)

        # Calculate dimensionality
        self.n_per_aircraft = len(self.per_aircraft_features)
        self.n_fleet = len(self.fleet_features)
        self.n_mission = len(self.mission_features)
        self.n_temporal = len(self.temporal_features)
        self.n_total = (self.n_per_aircraft * 8 +
                       self.n_fleet +
                       self.n_mission +
                       self.n_temporal)

        if self.verbose:
            print(f"StateEncoder initialized: {self.config['name']}")
            print(f"  Per-aircraft: {self.n_per_aircraft} × 8 = {self.n_per_aircraft * 8}")
            print(f"  Fleet: {self.n_fleet}")
            print(f"  Mission: {self.n_mission}")
            print(f"  Temporal: {self.n_temporal}")
            print(f"  Total: {self.n_total}")

    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from simulation state.

        Args:
            state: Simulation state dictionary with:
                - aircraft: List[Aircraft]
                - mission_forecast: List[MissionForecast]
                - fleet_metrics: Dict
                - tokens_available: Dict
                - maintenance_slots: Dict
                - fiscal_day: FiscalDay
                - current_or: float

        Returns:
            Feature array of shape (8, n_per_aircraft) for per-aircraft features,
            or (n_total,) if fleet/mission/temporal features included

        Notes:
            For simple config (per-aircraft only): returns (8, 3)
            For medium config (+ mission): returns (8, 6) + (5,) concatenated
            For full config (+ fleet): returns (8, 6) + (5,) + (8,) concatenated
        """
        aircraft_list = state['aircraft']

        # Extract per-aircraft features (always present)
        per_aircraft_matrix = self._extract_per_aircraft_features(aircraft_list)

        # For simple config (per-aircraft only), return (8, n_features)
        if (self.n_fleet == 0 and
            self.n_mission == 0 and
            self.n_temporal == 0):
            return per_aircraft_matrix

        # Otherwise, flatten and concatenate with other feature types
        all_features = []

        # Flatten per-aircraft features: (8, n) → (8*n,)
        all_features.append(per_aircraft_matrix.flatten())

        # Extract fleet-level features
        if self.n_fleet > 0:
            fleet_features = self._extract_fleet_features(state)
            all_features.append(fleet_features)

        # Extract mission features
        if self.n_mission > 0:
            mission_features = self._extract_mission_features(state)
            all_features.append(mission_features)

        # Extract temporal features
        if self.n_temporal > 0:
            temporal_features = self._extract_temporal_features(state)
            all_features.append(temporal_features)

        # Concatenate all feature vectors
        return np.concatenate(all_features)

    def _extract_per_aircraft_features(
        self,
        aircraft_list: List[Aircraft]
    ) -> np.ndarray:
        """
        Extract per-aircraft features.

        Args:
            aircraft_list: List of Aircraft objects

        Returns:
            Array of shape (8, n_per_aircraft)
        """
        # First calculate DA line ideal positions (needed for deviation features)
        ideal_hours = self._calculate_ideal_da_line_positions(aircraft_list)

        features_matrix = []

        for aircraft in sorted(aircraft_list, key=lambda a: a.id):
            aircraft_features = []

            for feature_def in self.per_aircraft_features:
                value = self._extract_single_feature(
                    feature_def,
                    aircraft=aircraft,
                    ideal_hours=ideal_hours
                )
                aircraft_features.append(value)

            features_matrix.append(aircraft_features)

        return np.array(features_matrix, dtype=np.float32)

    def _extract_fleet_features(self, state: Dict) -> np.ndarray:
        """Extract fleet-level features."""
        features = []

        for feature_def in self.fleet_features:
            # Pass aircraft_list for fleet features that need it (e.g., fmc_count)
            value = self._extract_single_feature(
                feature_def,
                state=state,
                aircraft_list=state.get('aircraft', [])
            )
            features.append(value)

        return np.array(features, dtype=np.float32)

    def _extract_mission_features(self, state: Dict) -> np.ndarray:
        """Extract mission-related features."""
        features = []

        for feature_def in self.mission_features:
            value = self._extract_single_feature(feature_def, state=state)
            features.append(value)

        return np.array(features, dtype=np.float32)

    def _extract_temporal_features(self, state: Dict) -> np.ndarray:
        """Extract temporal/fiscal features."""
        features = []

        for feature_def in self.temporal_features:
            value = self._extract_single_feature(feature_def, state=state)
            features.append(value)

        return np.array(features, dtype=np.float32)

    def _calculate_ideal_da_line_positions(
        self,
        aircraft_list: List[Aircraft],
        major_phase_hours: int = 500
    ) -> Dict[int, float]:
        """
        Calculate ideal DA line positions for fleet synchronization.

        Based on environment.R:1387-1403 DA line calculation.

        Args:
            aircraft_list: List of Aircraft objects
            major_phase_hours: Trigger for major phase (default: 500)

        Returns:
            Dict mapping aircraft_id to ideal hours_to_major for its rank
        """
        # Calculate hours until major phase for each aircraft
        hours_until_phase = {}
        for aircraft in aircraft_list:
            hours_remaining = major_phase_hours - aircraft.hours_since_major_phase
            hours_until_phase[aircraft.id] = max(0, hours_remaining)

        # Rank aircraft by hours_to_major (descending order)
        # Aircraft with most hours remaining gets rank 0 (top)
        sorted_aircraft = sorted(
            aircraft_list,
            key=lambda a: hours_until_phase[a.id],
            reverse=True
        )

        # Assign ranks (0 = most hours, 7 = least hours)
        aircraft_ranks = {ac.id: rank for rank, ac in enumerate(sorted_aircraft)}

        # Calculate ideal hours for each rank
        # Ideal spacing: evenly distributed across [0, 500]
        # Rank 0 (healthiest): 500 * (8-0)/8 = 500 hrs
        # Rank 7 (sickest):   500 * (8-7)/8 = 62.5 hrs
        n_aircraft = len(aircraft_list)
        ideal_for_rank = {}
        for aircraft_id, rank in aircraft_ranks.items():
            ideal_for_rank[aircraft_id] = major_phase_hours * (n_aircraft - rank) / n_aircraft

        return ideal_for_rank

    def _extract_single_feature(
        self,
        feature_def: Dict,
        aircraft: Aircraft = None,
        state: Dict = None,
        ideal_hours: Dict[int, float] = None,
        aircraft_list: List[Aircraft] = None
    ) -> float:
        """
        Extract a single feature value.

        Args:
            feature_def: Feature definition from config
            aircraft: Aircraft object (for per-aircraft features)
            state: State dict (for fleet/mission/temporal features)

        Returns:
            Normalized feature value
        """
        feature_name = feature_def['name']
        feature_type = feature_def['type']
        source = feature_def['source']

        # Extract raw value based on source
        if source == 'derived':
            # Evaluate formula
            raw_value = self._evaluate_formula(
                feature_def['formula'],
                aircraft=aircraft,
                state=state,
                ideal_hours=ideal_hours,
                aircraft_list=aircraft_list
            )
        elif source.startswith('aircraft.'):
            # Direct aircraft attribute
            if aircraft is None:
                raise ValueError(f"Aircraft required for feature: {feature_name}")
            attr_name = source.split('aircraft.')[1]
            raw_value = getattr(aircraft, attr_name)
        elif source.startswith('state['):
            # Direct state access: state['current_or']
            if state is None:
                raise ValueError(f"State required for feature: {feature_name}")
            key = source.split("'")[1]  # Extract key from state['key']
            raw_value = state[key]
        elif source.startswith('fleet_metrics['):
            # Fleet metrics access
            if state is None:
                raise ValueError(f"State required for feature: {feature_name}")
            key = source.split("'")[1]
            raw_value = state['fleet_metrics'][key]
        elif source.startswith('tokens_available['):
            # Token access
            if state is None:
                raise ValueError(f"State required for feature: {feature_name}")
            key = source.split("'")[1]
            raw_value = state['tokens_available'][key]
        elif source.startswith('maintenance_slots['):
            # Slot access
            if state is None:
                raise ValueError(f"State required for feature: {feature_name}")
            key = source.split("'")[1]
            raw_value = state['maintenance_slots'][key]
        elif source.startswith('mission_forecast['):
            # Mission forecast access: mission_forecast[0].required_aircraft
            if state is None:
                raise ValueError(f"State required for feature: {feature_name}")
            # Parse index and attribute
            # Example: "mission_forecast[0].required_aircraft"
            parts = source.split('.')
            index_part = parts[0]  # "mission_forecast[0]"
            index = int(index_part.split('[')[1].split(']')[0])
            attr = parts[1] if len(parts) > 1 else 'required_aircraft'

            mission_forecast = state.get('mission_forecast', [])
            if index < len(mission_forecast):
                raw_value = getattr(mission_forecast[index], attr)
            else:
                raw_value = 0  # Default if forecast not available
        elif source.startswith('fiscal_day.'):
            # Fiscal day attribute
            if state is None:
                raise ValueError(f"State required for feature: {feature_name}")
            attr = source.split('fiscal_day.')[1]
            raw_value = getattr(state['fiscal_day'], attr)
        else:
            raise ValueError(f"Unknown source format: {source}")

        # Handle missing values
        if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
            raw_value = self._handle_missing_value(feature_def)

        # Normalize
        normalized_value = self._normalize_value(raw_value, feature_def)

        return normalized_value

    def _evaluate_formula(
        self,
        formula: str,
        aircraft: Aircraft = None,
        state: Dict = None,
        ideal_hours: Dict[int, float] = None,
        aircraft_list: List[Aircraft] = None
    ) -> float:
        """
        Evaluate a derived feature formula.

        Supports simple arithmetic expressions like:
        - "250 - aircraft.hours_since_minor_phase"
        - "500 - aircraft.hours_since_major_phase"
        - "max([m.required_aircraft for m in mission_forecast[:3]])"
        - DA line deviation calculations
        - Fleet-level formulas: fmc_count, days_until_report

        Args:
            formula: Formula string from config
            aircraft: Aircraft object (if per-aircraft formula)
            state: State dict (if fleet/mission formula)
            ideal_hours: Dict of ideal DA line positions (for deviation features)
            aircraft_list: List of Aircraft objects (for fleet formulas)

        Returns:
            Evaluated numeric value
        """
        # Special case: days_until_reporting_period_end (16th to 15th cycle)
        if formula == 'days_until_reporting_period_end' and state is not None:
            # Reporting period runs from 16th of month M to 15th of month M+1
            # Calculate day of month from fiscal_day
            fiscal_day_obj = state.get('fiscal_day', None)
            if fiscal_day_obj is None:
                return 0.0

            days_into_fy = fiscal_day_obj.days_into_fy

            # Calculate day of month (assuming 30-day months for simplicity)
            # More accurate: track actual calendar days, but this is approximate
            days_in_month = 30
            day_of_month = ((days_into_fy - 1) % days_in_month) + 1

            # Calculate days until 15th (end of reporting period)
            if day_of_month >= 16:
                # Currently in period that ends on 15th of next month
                days_until_report = (days_in_month - day_of_month) + 15
            else:
                # Currently in period that ends on 15th of this month
                days_until_report = 15 - day_of_month

            return float(max(0, days_until_report))

        # DA line deviation features (special handling)
        if 'da_line_deviation' in formula and aircraft is not None and ideal_hours is not None:
            # Calculate actual hours_to_major
            actual_hours = 500 - aircraft.hours_since_major_phase
            actual_hours = max(0, actual_hours)

            # Get ideal hours for this aircraft
            ideal = ideal_hours.get(aircraft.id, 0)

            # Calculate deviation
            if 'positive' in formula:
                # Positive deviation: ahead of schedule (more capacity than ideal)
                deviation = max(0, actual_hours - ideal)
                return float(deviation)  # Return raw hours
            elif 'negative' in formula:
                # Negative deviation: behind schedule (less capacity than ideal)
                deviation = max(0, ideal - actual_hours)
                return float(deviation)  # Return raw hours
            elif formula == 'da_line_deviation':
                # Signed deviation: actual - ideal (hours_to_major)
                # Positive = BEHIND schedule (more bank hours than ideal → low utilization → should fly)
                # Negative = AHEAD of schedule (fewer bank hours than ideal → high utilization → protect)
                deviation = actual_hours - ideal
                return float(deviation)  # Range: [-500, +437] approximately
            else:
                return 0.0

        # Fleet-level formulas with aircraft list
        if aircraft_list is not None and 'aircraft' in formula:
            # Formulas like: sum([1 for ac in aircraft if ac.status == 'FMC'])
            local_vars = {
                'aircraft': aircraft_list,
                'sum': sum,
                'max': max,
                'min': min,
                'len': len
            }
            try:
                result = eval(formula, {"__builtins__": {}}, local_vars)
                return float(result)
            except Exception as e:
                if self.verbose:
                    print(f"Fleet formula evaluation error: {formula}, {e}")
                return 0.0

        # Simple derivations for common patterns
        if 'aircraft.' in formula and aircraft is not None:
            # Replace aircraft attributes
            local_vars = {'aircraft': aircraft}
            try:
                result = eval(formula, {"__builtins__": {}}, local_vars)
                return float(result)
            except Exception as e:
                if self.verbose:
                    print(f"Formula evaluation error: {formula}, {e}")
                return 0.0

        elif 'mission_forecast' in formula and state is not None:
            # Mission forecast formulas
            mission_forecast = state.get('mission_forecast', [])
            local_vars = {
                'mission_forecast': mission_forecast,
                'max': max,
                'min': min,
                'sum': sum,
                'len': len,
                'mean': lambda x: sum(x) / len(x) if len(x) > 0 else 0
            }
            try:
                result = eval(formula, {"__builtins__": {}}, local_vars)
                return float(result)
            except Exception as e:
                if self.verbose:
                    print(f"Formula evaluation error: {formula}, {e}")
                return 0.0

        else:
            if self.verbose:
                print(f"Unknown formula pattern: {formula}")
            return 0.0

    def _normalize_value(self, value: float, feature_def: Dict) -> float:
        """
        Normalize feature value based on method and bounds.

        Args:
            value: Raw feature value
            feature_def: Feature definition with normalization settings

        Returns:
            Normalized value
        """
        norm_method = feature_def.get('normalization', self.norm_method)
        bounds = feature_def.get('bounds', None)

        if norm_method == 'none':
            return float(value)

        elif norm_method == 'minmax':
            if bounds is None:
                # No bounds specified, return as-is
                return float(value)

            min_val, max_val = bounds
            if max_val == min_val:
                return 0.5  # Avoid division by zero

            # Clip to bounds if enabled
            if self.clip_outliers:
                value = np.clip(value, min_val, max_val)

            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            return float(np.clip(normalized, 0.0, 1.0))

        elif norm_method == 'zscore':
            # Z-score normalization (would need statistics)
            # For now, just return as-is
            if self.verbose:
                print("Warning: zscore normalization not fully implemented")
            return float(value)

        else:
            return float(value)

    def _handle_missing_value(self, feature_def: Dict) -> float:
        """Handle missing/NaN values based on config."""
        if self.handle_missing == 'zero':
            return 0.0
        elif self.handle_missing == 'mean':
            # Would need statistics tracking
            return 0.5  # Placeholder
        else:
            return 0.0

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names in order.

        Returns:
            List of feature name strings
        """
        names = []

        # Per-aircraft features (repeated for each aircraft)
        for aircraft_id in range(8):
            for feature_def in self.per_aircraft_features:
                names.append(f"aircraft_{aircraft_id}_{feature_def['name']}")

        # Fleet features
        for feature_def in self.fleet_features:
            names.append(f"fleet_{feature_def['name']}")

        # Mission features
        for feature_def in self.mission_features:
            names.append(f"mission_{feature_def['name']}")

        # Temporal features
        for feature_def in self.temporal_features:
            names.append(f"temporal_{feature_def['name']}")

        return names

    def get_dimensionality(self) -> Tuple[int, str]:
        """
        Get feature dimensionality and shape description.

        Returns:
            (total_dims, shape_description)
        """
        if (self.n_fleet == 0 and
            self.n_mission == 0 and
            self.n_temporal == 0):
            shape = f"({8}, {self.n_per_aircraft})"
        else:
            shape = f"({self.n_total},)"

        return self.n_total, shape

    def __repr__(self) -> str:
        """String representation."""
        return (f"StateEncoder(config='{self.config['name']}', "
                f"dims={self.n_total})")
