"""Fleet-level metrics calculator for DA line analysis.

Ported from: aviation_hierarchical_sim_v2/core_v2/environment.R (lines 1340-1395)

Calculates depot activation (DA) line metrics to assess fleet health:
- Total bank hours: Fleet capacity remaining until major phase maintenance
- Clustering: How well aircraft are distributed along DA line (ideal spacing)
- Capacity strain: Exponential penalty when fleet capacity is low
- Per-aircraft deviance: Individual aircraft contributions to clustering
"""

from typing import List, Dict
import numpy as np


class FleetMetrics:
    """Calculator for fleet-level DA line metrics.

    Metrics help assess:
    1. Fleet capacity health (total bank hours)
    2. Maintenance scheduling efficiency (clustering)
    3. Risk of multiple simultaneous depot activations
    """

    def __init__(self, major_phase_threshold: float = 500.0, num_aircraft: int = 8):
        """Initialize fleet metrics calculator.

        Args:
            major_phase_threshold: Hours until major phase maintenance
            num_aircraft: Size of aircraft fleet
        """
        self.major_phase_threshold = major_phase_threshold
        self.num_aircraft = num_aircraft

        # Capacity threshold: 1000 hours (25% of max, ~2 aircraft worth of buffer)
        # Below this, clustering becomes a serious risk
        self.capacity_threshold = 1000.0

        # Ideal spacing between aircraft on DA line
        # For 8 aircraft with 500hr threshold: 500/8 = 62.5 hours apart
        self.ideal_spacing = major_phase_threshold / num_aircraft

    def calculate_bank_hours(self, aircraft_list: List) -> Dict[str, float]:
        """Calculate fleet bank hours (capacity remaining until major phase).

        Args:
            aircraft_list: List of Aircraft objects with hours_since_major_phase

        Returns:
            Dict with:
                - total_bank_hours: Sum of hours until major phase across fleet
                - bank_hours_per_aircraft: List of individual aircraft bank hours
                - avg_bank_hours: Mean bank hours per aircraft
        """
        # Calculate hours until major phase for each aircraft
        bank_hours_per_aircraft = []
        for aircraft in aircraft_list:
            hours_remaining = self.major_phase_threshold - aircraft.hours_since_major_phase
            bank_hours_per_aircraft.append(max(0.0, hours_remaining))

        total_bank_hours = sum(bank_hours_per_aircraft)
        avg_bank_hours = total_bank_hours / len(aircraft_list) if aircraft_list else 0.0

        return {
            "total_bank_hours": total_bank_hours,
            "bank_hours_per_aircraft": bank_hours_per_aircraft,
            "avg_bank_hours": avg_bank_hours,
        }

    def calculate_clustering_metric(self, bank_hours_per_aircraft: List[float]) -> Dict[str, float]:
        """Calculate clustering metric based on deviation from ideal DA line spacing.

        Clustering measures how poorly aircraft are distributed along the DA line.
        - 0.0 = Perfect spacing (aircraft evenly distributed)
        - 1.0 = Worst clustering (all aircraft bunched together)

        Method:
        1. Sort aircraft by bank hours (descending)
        2. Compare to ideal evenly-spaced distribution
        3. Calculate MSE normalized by max possible MSE

        Args:
            bank_hours_per_aircraft: List of bank hours for each aircraft

        Returns:
            Dict with:
                - clustering: Normalized clustering metric [0, 1]
                - mse: Mean squared error from ideal spacing
                - max_possible_mse: Maximum possible MSE (normalization factor)
                - ideal_positions: Ideal bank hours for each rank position
        """
        n = len(bank_hours_per_aircraft)

        # Edge case: Single aircraft has no clustering problem
        if n == 1:
            return {
                "clustering": 0.0,
                "mse": 0.0,
                "max_possible_mse": 0.0,
                "ideal_positions": [self.major_phase_threshold],
            }

        # Sort aircraft by bank hours (descending) - highest capacity first
        sorted_hours = sorted(bank_hours_per_aircraft, reverse=True)

        # Generate ideal evenly-spaced positions
        # E.g., for 8 aircraft: [500, 437.5, 375, 312.5, 250, 187.5, 125, 62.5]
        ideal_positions = np.linspace(
            self.major_phase_threshold,
            self.ideal_spacing,
            n
        ).tolist()

        # Calculate squared deviations from ideal positions
        deviations_squared = [(actual - ideal)**2
                              for actual, ideal in zip(sorted_hours, ideal_positions)]
        mse = np.mean(deviations_squared)

        # Normalize by maximum possible MSE
        # Max MSE occurs when all aircraft are at one extreme (all at 0 or all at 500)
        max_possible_mse = np.mean([ideal**2 for ideal in ideal_positions])

        # Clustering metric: normalized MSE, clamped to [0, 1]
        clustering = min(mse / max_possible_mse, 1.0) if max_possible_mse > 0 else 0.0

        return {
            "clustering": clustering,
            "mse": mse,
            "max_possible_mse": max_possible_mse,
            "ideal_positions": ideal_positions,
        }

    def calculate_capacity_strain(self, total_bank_hours: float) -> float:
        """Calculate capacity strain metric based on total fleet bank hours.

        Capacity strain increases exponentially as total bank hours drops below
        the capacity threshold (1000 hours). This motivates proactive maintenance
        before the fleet becomes critically strained.

        Formula: 1 - exp((total_bank_hours - 1000) / 500)

        Args:
            total_bank_hours: Sum of bank hours across entire fleet

        Returns:
            Capacity strain metric [0, 1] where:
                - 0.0 = Plenty of capacity (>= 1000 hours)
                - 1.0 = Maximum strain (near 0 hours)
        """
        if total_bank_hours >= self.capacity_threshold:
            return 0.0  # Plenty of capacity - no strain

        # Exponential strain as bank hours drop below threshold
        # Denominator (500) controls steepness of exponential curve
        strain = 1.0 - np.exp((total_bank_hours - self.capacity_threshold) / 500.0)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, strain))

    def calculate_per_aircraft_deviance(
        self,
        bank_hours_per_aircraft: List[float],
        ideal_positions: List[float],
        max_possible_mse: float
    ) -> List[float]:
        """Calculate per-aircraft DA line deviance for agent attribution.

        Enables agents to understand which specific aircraft are causing
        clustering problems by showing each aircraft's squared deviation
        from its ideal DA line position.

        Args:
            bank_hours_per_aircraft: Bank hours for each aircraft (original order)
            ideal_positions: Ideal bank hours for each rank position
            max_possible_mse: Normalization factor

        Returns:
            List of deviance values (one per aircraft) normalized by max_possible_mse
        """
        n = len(bank_hours_per_aircraft)

        # Rank aircraft by bank hours (highest = rank 1)
        # Ties broken by first occurrence
        ranks = self._rank_descending(bank_hours_per_aircraft)

        # Calculate deviance for each aircraft
        per_aircraft_deviance = []
        for i in range(n):
            rank = ranks[i]  # 0-indexed rank
            ideal_for_rank = ideal_positions[rank]

            # Squared deviation from ideal position, normalized
            deviation = (bank_hours_per_aircraft[i] - ideal_for_rank)**2
            normalized_deviation = deviation / max_possible_mse if max_possible_mse > 0 else 0.0
            per_aircraft_deviance.append(normalized_deviation)

        return per_aircraft_deviance

    def calculate_all_metrics(self, aircraft_list: List) -> Dict:
        """Calculate all fleet-level DA line metrics.

        Args:
            aircraft_list: List of Aircraft objects

        Returns:
            Dict with all fleet metrics:
                - total_bank_hours: Sum of hours until major phase
                - bank_hours_per_aircraft: List of individual bank hours
                - avg_bank_hours: Mean bank hours per aircraft
                - clustering: Clustering metric [0, 1]
                - capacity_strain: Capacity strain metric [0, 1]
                - penalty: Combined DA line penalty (clustering × strain)
                - per_aircraft_deviance: List of deviance values per aircraft
        """
        # Edge case: Empty fleet
        if not aircraft_list:
            return {
                "total_bank_hours": 0.0,
                "bank_hours_per_aircraft": [],
                "avg_bank_hours": 0.0,
                "clustering": 0.0,
                "capacity_strain": 0.0,
                "penalty": 0.0,
                "per_aircraft_deviance": [],
            }

        # Step 1: Calculate bank hours
        bank_metrics = self.calculate_bank_hours(aircraft_list)
        bank_hours_per_aircraft = bank_metrics["bank_hours_per_aircraft"]
        total_bank_hours = bank_metrics["total_bank_hours"]

        # Step 2: Calculate clustering
        clustering_metrics = self.calculate_clustering_metric(bank_hours_per_aircraft)
        clustering = clustering_metrics["clustering"]

        # Step 3: Calculate capacity strain
        capacity_strain = self.calculate_capacity_strain(total_bank_hours)

        # Step 4: Combined penalty (only penalize clustering when capacity is strained)
        penalty = clustering * capacity_strain

        # Step 5: Per-aircraft deviance for agent attribution
        per_aircraft_deviance = self.calculate_per_aircraft_deviance(
            bank_hours_per_aircraft,
            clustering_metrics["ideal_positions"],
            clustering_metrics["max_possible_mse"]
        )

        return {
            "total_bank_hours": total_bank_hours,
            "bank_hours_per_aircraft": bank_hours_per_aircraft,
            "avg_bank_hours": bank_metrics["avg_bank_hours"],
            "clustering": clustering,
            "capacity_strain": capacity_strain,
            "penalty": penalty,
            "per_aircraft_deviance": per_aircraft_deviance,
        }

    @staticmethod
    def _rank_descending(values: List[float]) -> List[int]:
        """Rank values in descending order (highest = rank 0).

        Args:
            values: List of numeric values

        Returns:
            List of ranks (0-indexed), ties broken by first occurrence
        """
        # Create list of (value, original_index) pairs
        indexed_values = [(val, idx) for idx, val in enumerate(values)]

        # Sort by value (descending), breaking ties by original index
        sorted_indexed = sorted(indexed_values, key=lambda x: (-x[0], x[1]))

        # Create rank mapping
        ranks = [0] * len(values)
        for rank, (_, original_idx) in enumerate(sorted_indexed):
            ranks[original_idx] = rank

        return ranks


def calculate_bank_hours_category(total_bank_hours: float) -> str:
    """Categorize fleet bank hours into discrete levels.

    Used for state discretization in RL agents.

    Args:
        total_bank_hours: Sum of bank hours across fleet

    Returns:
        Category string: "High", "Medium", or "Low"
    """
    if total_bank_hours > 2750:
        return "High"
    elif total_bank_hours >= 1250:
        return "Medium"
    else:
        return "Low"
