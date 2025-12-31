"""Baseline rule-based policy for benchmarking.

Policy Logic:
- Fly exactly the mission requirement using stalest aircraft (wear leveling)
- Staleness = highest hours_since_major_phase (closest to major phase)
- NMC aircraft get appropriate maintenance based on failure reason
- Other FMC aircraft hold
- Rationale: Flying stalest first prevents maintenance clustering and smooths DA line
"""

from typing import Dict, List
from simulation.aircraft import Aircraft
from policy.base_policy import BasePolicy


class BaselinePolicy(BasePolicy):
    """Simple rule-based baseline policy.

    Serves as benchmark for GA-optimized policies.

    Decision Logic:
    1. NMC aircraft → Appropriate maintenance (reactive/minor_phase/major_phase)
    2. FMC aircraft → Sort by hours_since_major_phase (descending = stalest first)
    3. Select top M aircraft for flying (M = mission requirement)
    4. Remaining FMC aircraft → Hold

    Rationale: Flying stalest aircraft provides wear leveling across the fleet,
    prevents maintenance clustering, and smooths DA line distribution.
    """

    def __init__(self, verbose: bool = False):
        """Initialize baseline policy.

        Args:
            verbose: Enable detailed logging
        """
        super().__init__(verbose=verbose)

    def decide(self, state: Dict) -> Dict[int, str]:
        """Generate actions for all aircraft.

        Args:
            state: Environment state dictionary

        Returns:
            Dict mapping aircraft_id to action string
        """
        aircraft_list = state["aircraft"]
        mission_forecast = state["mission_forecast"]

        # Get today's mission requirement
        # NOTE: After step() returns state, forecast[0] is the mission just completed.
        # forecast[1] is what the NEXT step() will use, so we read [1] for planning.
        if mission_forecast and len(mission_forecast) > 1:
            required_aircraft = mission_forecast[1].required_aircraft
        elif mission_forecast and len(mission_forecast) > 0:
            required_aircraft = mission_forecast[0].required_aircraft  # Day 1 fallback
        else:
            required_aircraft = 0

        actions = {}

        # Step 1: Handle NMC aircraft (they need maintenance)
        nmc_aircraft = [a for a in aircraft_list if a.status == "NMC"]
        for aircraft in nmc_aircraft:
            action = self._get_nmc_action(aircraft)
            actions[aircraft.id] = action

        # Step 2: Handle FMC aircraft
        fmc_aircraft = [a for a in aircraft_list if a.status == "FMC" and not a.in_maintenance]

        # Sort FMC aircraft by staleness (highest hours_since_major_phase first)
        # This prioritizes aircraft closest to major phase maintenance (wear leveling)
        fmc_aircraft_sorted = sorted(fmc_aircraft, key=lambda a: a.hours_since_major_phase, reverse=True)

        # Step 3: Assign flying to stalest M aircraft
        for i, aircraft in enumerate(fmc_aircraft_sorted):
            if i < required_aircraft:
                # Stale aircraft flies (wear leveling)
                actions[aircraft.id] = "fly"
            else:
                # Remaining aircraft hold
                actions[aircraft.id] = "hold"

        # Step 4: Aircraft in maintenance automatically hold (no action needed, handled by environment)
        for aircraft in aircraft_list:
            if aircraft.in_maintenance and aircraft.id not in actions:
                actions[aircraft.id] = "hold"  # Placeholder (environment will ignore)

        if self.verbose:
            print(f"Policy: Flying {min(required_aircraft, len(fmc_aircraft))}/{required_aircraft} required aircraft")

        return actions

    def _get_nmc_action(self, aircraft: Aircraft) -> str:
        """Determine appropriate maintenance action for NMC aircraft.

        Args:
            aircraft: NMC aircraft

        Returns:
            Maintenance action string
        """
        if aircraft.nmc_reason == "nmc_rul":
            return "reactive_maintain"
        elif aircraft.nmc_reason == "nmc_minor_phase":
            return "minor_phase_maintain"
        elif aircraft.nmc_reason == "nmc_major_phase":
            return "major_phase_maintain"
        else:
            # Fallback: hold (shouldn't happen)
            if self.verbose:
                print(f"⚠️  Aircraft {aircraft.id} is NMC but nmc_reason is {aircraft.nmc_reason}")
            return "hold"

    def get_name(self) -> str:
        """Return policy name for logging."""
        return "Baseline_FlyStalest"

    def __repr__(self) -> str:
        """String representation."""
        return "BaselinePolicy(rule='fly_stalest_aircraft_wear_leveling')"
