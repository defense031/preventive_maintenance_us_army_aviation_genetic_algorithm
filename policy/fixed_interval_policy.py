"""Fixed-interval preventive maintenance policy.

Forces preventive maintenance every N flying hours regardless of RUL.
This represents a traditional time-based maintenance approach that
doesn't use sensor information - the baseline for comparison.
"""

from typing import Dict
from simulation.aircraft import Aircraft
from policy.base_policy import BasePolicy


class FixedIntervalPolicy(BasePolicy):
    """Fixed-interval preventive maintenance policy.

    Triggers preventive maintenance every `interval_hours` of flying time,
    regardless of actual component condition (RUL). This represents the
    traditional fixed-interval approach that predates condition-based maintenance.

    Decision Logic:
    1. NMC aircraft → Appropriate mandatory maintenance
    2. FMC aircraft past interval → Preventive maintenance (if slot available)
    3. Remaining FMC → Fly stalest first (wear leveling) up to mission requirement
    4. Others → Hold
    """

    def __init__(self, interval_hours: float = 25.0, verbose: bool = False):
        """Initialize fixed-interval policy.

        Args:
            interval_hours: Hours between mandatory preventive maintenance
            verbose: Enable detailed logging
        """
        super().__init__(verbose=verbose)
        self.interval_hours = interval_hours

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
        if mission_forecast and len(mission_forecast) > 1:
            required_aircraft = mission_forecast[1].required_aircraft
        elif mission_forecast and len(mission_forecast) > 0:
            required_aircraft = mission_forecast[0].required_aircraft
        else:
            required_aircraft = 0

        actions = {}

        # Step 1: Handle NMC aircraft (mandatory maintenance)
        nmc_aircraft = [a for a in aircraft_list if a.status == "NMC"]
        for aircraft in nmc_aircraft:
            action = self._get_nmc_action(aircraft)
            actions[aircraft.id] = action

        # Step 2: Get FMC aircraft not in maintenance
        fmc_aircraft = [a for a in aircraft_list if a.status == "FMC" and not a.in_maintenance]

        # Step 3: Check which FMC aircraft are due for fixed-interval preventive
        due_for_preventive = []
        not_due = []

        for aircraft in fmc_aircraft:
            if aircraft.hours_since_preventive >= self.interval_hours:
                due_for_preventive.append(aircraft)
            else:
                not_due.append(aircraft)

        # Sort due aircraft by most overdue first
        due_for_preventive.sort(key=lambda a: a.hours_since_preventive, reverse=True)

        # Assign preventive maintenance to due aircraft (limited by slots - 2 available)
        # Aircraft past interval are GROUNDED until maintained - they cannot fly
        preventive_assigned = 0
        max_preventive_slots = 2

        for aircraft in due_for_preventive:
            if preventive_assigned < max_preventive_slots:
                actions[aircraft.id] = "preventive_maintain"
                preventive_assigned += 1
                if self.verbose:
                    print(f"  Aircraft {aircraft.id}: preventive ({aircraft.hours_since_preventive:.1f}h >= {self.interval_hours}h)")
            else:
                # No slot available - aircraft is GROUNDED (cannot fly until maintained)
                actions[aircraft.id] = "hold"
                if self.verbose:
                    print(f"  Aircraft {aircraft.id}: GROUNDED ({aircraft.hours_since_preventive:.1f}h >= {self.interval_hours}h, no slot)")

        # Step 4: Assign flying to remaining FMC aircraft that are NOT past interval (wear leveling)
        not_due_sorted = sorted(not_due, key=lambda a: a.hours_since_major_phase, reverse=True)

        flying_assigned = 0
        for aircraft in not_due_sorted:
            if aircraft.id not in actions:
                if flying_assigned < required_aircraft:
                    actions[aircraft.id] = "fly"
                    flying_assigned += 1
                else:
                    actions[aircraft.id] = "hold"

        # Step 5: Aircraft in maintenance hold
        for aircraft in aircraft_list:
            if aircraft.in_maintenance and aircraft.id not in actions:
                actions[aircraft.id] = "hold"

        if self.verbose:
            print(f"FixedInterval({self.interval_hours}h): {preventive_assigned} preventive, {flying_assigned}/{required_aircraft} flying")

        return actions

    def _get_nmc_action(self, aircraft: Aircraft) -> str:
        """Determine appropriate maintenance action for NMC aircraft."""
        if aircraft.nmc_reason == "nmc_rul":
            return "reactive_maintain"
        elif aircraft.nmc_reason == "nmc_minor_phase":
            return "minor_phase_maintain"
        elif aircraft.nmc_reason == "nmc_major_phase":
            return "major_phase_maintain"
        else:
            return "hold"

    def get_name(self) -> str:
        """Return policy name for logging."""
        return f"FixedInterval_{self.interval_hours}h"

    def __repr__(self) -> str:
        return f"FixedIntervalPolicy(interval={self.interval_hours}h)"
