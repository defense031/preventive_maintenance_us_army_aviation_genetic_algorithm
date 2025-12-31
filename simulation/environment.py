"""Aviation maintenance simulation environment.

Ported from: aviation_hierarchical_sim_v2/core_v2/environment.R

Main simulation engine with daily step loop and state management.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from copy import deepcopy

from simulation.aircraft import Aircraft
from simulation.fiscal_calendar import FiscalCalendar
from simulation.token_tracker import TokenTracker
from simulation.mission_generator import MissionGenerator
from simulation.maintenance_system import MaintenanceSystem
from simulation.fleet_metrics import FleetMetrics
from utils.config import SimulationConfig


class Environment:
    """Aviation maintenance simulation environment.

    Manages daily simulation loop with all subsystems.
    """

    def __init__(self, config: SimulationConfig, rng: Optional[np.random.Generator] = None):
        """Initialize environment.

        Args:
            config: Simulation configuration
            rng: NumPy random generator
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.verbose = config.verbose

        # Fleet metrics calculator
        self.fleet_metrics = FleetMetrics(
            major_phase_threshold=config.phase.major_threshold,
            num_aircraft=config.num_aircraft
        )

        # Will be initialized in reset()
        self.fiscal_calendar: Optional[FiscalCalendar] = None
        self.token_tracker: Optional[TokenTracker] = None
        self.mission_generator: Optional[MissionGenerator] = None
        self.maintenance_system: Optional[MaintenanceSystem] = None
        self.aircraft: List[Aircraft] = []

        # Simulation state
        self.current_day = 0
        self.current_or = 0.0
        self.or_history: List[float] = []
        self.mission_success_history: List[bool] = []
        self.total_flight_hours = 0.0
        self.total_inflight_failures = 0

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset environment to initial state.

        Args:
            seed: Random seed

        Returns:
            Initial state dictionary
        """
        # Set seed if provided
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Initialize subsystems
        self.fiscal_calendar = FiscalCalendar(
            start_fy=2025,  # Default FY2025
            sim_days=self.config.sim_days
        )

        self.token_tracker = TokenTracker(
            annual_token_budget=self.config.tokens.annual_budget,
            fiscal_calendar=self.fiscal_calendar,
            verbose=self.verbose
        )

        self.mission_generator = MissionGenerator(
            num_aircraft=self.config.num_aircraft,
            tempo_regime=self.config.mission.tempo_regime,
            forecast_horizon=self.config.mission.forecast_days,
            min_sortie_hours=self.config.mission.min_sortie_hours,
            max_sortie_hours=self.config.mission.max_sortie_hours,
            entropy_scale=self.config.mission.transition_entropy_scale,
            rng=self.rng,
            verbose=self.verbose
        )

        self.maintenance_system = MaintenanceSystem(
            config=self.config,
            rng=self.rng,
            verbose=self.verbose
        )

        # Initialize aircraft fleet with realistic mid-lifecycle conditions
        self.aircraft = []
        for i in range(self.config.num_aircraft):
            # Sample hours since major phase from config range
            hours_since_major = self.rng.uniform(
                self.config.fleet_init.major_hours_min,
                self.config.fleet_init.major_hours_max
            )

            # Calculate hours since minor phase based on 250-hour cycle
            # - If hours_since_major < 250: haven't done minor yet, so minor hours = major hours
            # - If hours_since_major >= 250: did minor at 250, so minor hours = major hours - 250
            if hours_since_major < self.config.phase.minor_threshold:
                hours_since_minor = hours_since_major
                hours_since_last_maintenance = hours_since_major  # Last maintenance was at start (or previous major)
            else:
                hours_since_minor = hours_since_major - self.config.phase.minor_threshold
                hours_since_last_maintenance = hours_since_minor  # Last maintenance was minor phase at 250 hours

            # Sample fresh RUL and apply degradation correlation
            fresh_rul = self.config.rul.sample_reset_rul(self.rng)

            # Apply wear based on hours since last RUL reset (last maintenance)
            # correlation_factor: 0 = no correlation, 1 = fully correlated (1:1 wear)
            wear = hours_since_last_maintenance * self.config.fleet_init.rul_correlation
            initial_rul = max(10.0, fresh_rul - wear)  # Ensure at least 10 hours RUL

            observed_rul = self.config.rul.add_observation_noise(initial_rul, self.rng)

            # Set total_flight_hours = hours_since_major for realism
            total_flight_hours = hours_since_major

            aircraft = Aircraft(
                id=i,
                status="FMC",
                true_rul=initial_rul,
                observed_rul=observed_rul,
                total_flight_hours=total_flight_hours,
                hours_since_minor_phase=hours_since_minor,
                hours_since_major_phase=hours_since_major
            )
            self.aircraft.append(aircraft)

        # Reset simulation state
        self.current_day = 0
        self.current_or = 1.0  # All FMC at start
        self.or_history = []
        self.mission_success_history = []
        self.total_flight_hours = 0.0
        self.total_inflight_failures = 0

        # Release Q1 tokens at start
        self.token_tracker.release_quarterly_tokens(quarter=1, sim_day=1)

        if self.verbose:
            print(f"🚀 Environment reset: {self.config.num_aircraft} aircraft, {self.config.sim_days} days")

        return self.get_state()

    def step(self, actions: Dict[int, str]) -> Tuple[Dict, float, bool, Dict]:
        """Execute one day of simulation.

        Args:
            actions: Dict mapping aircraft_id to action string

        Returns:
            (state, reward, done, info) tuple
        """
        self.current_day += 1

        # Get current fiscal day
        fiscal_day = self.fiscal_calendar.get_day(self.current_day)

        # 1. Reset daily aircraft flags
        for aircraft in self.aircraft:
            aircraft.reset_daily_state()

        # 2. Process maintenance completions (must happen before new maintenance starts)
        completed_ids = self.maintenance_system.process_daily_maintenance(self.aircraft)

        # 3. Get today's mission
        mission = self.mission_generator.get_daily_mission(self.current_day)

        # 4. Execute actions
        flying_aircraft = []
        for aircraft_id, action in actions.items():
            aircraft = self.aircraft[aircraft_id]
            success = self._execute_action(aircraft, action, mission)

            if success and action == "fly":
                flying_aircraft.append(aircraft)

        # 5. Process flights (RUL degradation)
        mission_success = len(flying_aircraft) >= mission["required_aircraft"]
        self.mission_success_history.append(mission_success)

        for aircraft in flying_aircraft:
            flight_hours = mission["hours_per_sortie"]

            # Add flight hours (this decrements RUL)
            aircraft.add_flight_hours(flight_hours)
            self.total_flight_hours += flight_hours

            # Update observed RUL with noise
            aircraft.observed_rul = self.config.rul.add_observation_noise(aircraft.true_rul, self.rng)

        # 6. Check maintenance triggers (NMC conditions)
        self._check_maintenance_triggers()

        # 7. Check for quarterly token release (exact first day of quarter)
        # Tokens release on: Oct 1 (Q1), Jan 1 (Q2), Apr 1 (Q3), Jul 1 (Q4)
        if fiscal_day and fiscal_day.days_into_quarter == 1 and self.current_day > 1:
            quarter = fiscal_day.fiscal_quarter
            result = self.token_tracker.release_quarterly_tokens(
                quarter=quarter,
                sim_day=self.current_day
            )
            if self.verbose and result["success"]:
                print(f"💰 Q{quarter} tokens released: {result['tokens_released']} tokens")

        # 8. Calculate daily metrics
        metrics = self._calculate_daily_metrics(mission, mission_success)
        self.or_history.append(metrics["current_or"])
        self.current_or = metrics["current_or"]

        # 9. Calculate reward (simple: mission success + OR)
        reward = 1.0 if mission_success else 0.0
        reward += metrics["current_or"]

        # 10. Check if done
        done = self.current_day >= self.config.sim_days

        # 11. Prepare info dict
        info = {
            "day": self.current_day,
            "mission_success": mission_success,
            "flying_aircraft": len(flying_aircraft),
            "required_aircraft": mission["required_aircraft"],
            "metrics": metrics,
            "completed_maintenance": completed_ids
        }

        return self.get_state(), reward, done, info

    def _execute_action(self, aircraft: Aircraft, action: str, mission: Dict) -> bool:
        """Execute and validate a single aircraft action.

        Args:
            aircraft: Aircraft to act
            action: Action string
            mission: Current mission dict

        Returns:
            True if action executed successfully
        """
        # Validate action
        if not self._validate_action(aircraft, action):
            # Invalid action - default to hold
            action = "hold"

        aircraft.todays_decision = action

        # Execute action
        if action == "fly":
            # Aircraft will fly (hours added later in step())
            return True

        elif action == "hold":
            # Do nothing
            return True

        elif action.endswith("_maintain"):
            # Extract maintenance type
            maintenance_type = action.replace("_maintain", "")

            # Check token availability
            if not self.token_tracker.check_token_available(maintenance_type):
                # No tokens - downgrade to hold
                aircraft.todays_decision = "hold"
                return False

            # Try to start maintenance
            success = self.maintenance_system.start_maintenance(aircraft, maintenance_type)

            if success:
                # Set maintenance started flag (for detailed_daily_operations.csv)
                aircraft.maintenance_started_today = maintenance_type

                # Consume token
                self.token_tracker.consume_token(
                    maintenance_type, aircraft.id, self.current_day
                )
                return True
            else:
                # No slot available - downgrade to hold
                aircraft.todays_decision = "hold"
                return False

        return False

    def _validate_action(self, aircraft: Aircraft, action: str) -> bool:
        """Validate if action is legal for aircraft.

        Args:
            aircraft: Aircraft
            action: Action string

        Returns:
            True if action is valid
        """
        # NMC aircraft cannot fly
        if aircraft.status == "NMC" and action == "fly":
            return False

        # Aircraft in maintenance cannot take actions
        if aircraft.in_maintenance:
            return False

        # Reactive maintenance only for RUL failures
        if action == "reactive_maintain" and aircraft.nmc_reason != "nmc_rul":
            return False

        # Phase maintenance only when triggered
        if action == "minor_phase_maintain" and aircraft.nmc_reason != "nmc_minor_phase":
            return False

        if action == "major_phase_maintain" and aircraft.nmc_reason != "nmc_major_phase":
            return False

        return True

    def _check_maintenance_triggers(self) -> None:
        """Check for mandatory maintenance triggers (NMC conditions)."""
        for aircraft in self.aircraft:
            # Skip if already NMC or in maintenance
            if aircraft.status == "NMC" or aircraft.in_maintenance:
                continue

            # Priority 1: RUL failure (true_rul <= 0)
            if aircraft.true_rul <= 0:
                aircraft.flight_failure_today = True  # Mark in-flight failure for CSV tracking
                aircraft.set_nmc("nmc_rul")
                self.total_inflight_failures += 1

                if self.verbose:
                    print(f"❌ Aircraft {aircraft.id} NMC: RUL failure ({aircraft.true_rul:.1f} hrs)")

            # Priority 2: Major phase threshold (500 hours)
            elif aircraft.hours_since_major_phase >= self.config.phase.major_threshold:
                aircraft.set_nmc("nmc_major_phase")

                if self.verbose:
                    print(f"⚠️  Aircraft {aircraft.id} NMC: Major phase threshold ({aircraft.hours_since_major_phase:.1f} hrs)")

            # Priority 3: Minor phase threshold (250 hours)
            elif aircraft.hours_since_minor_phase >= self.config.phase.minor_threshold:
                aircraft.set_nmc("nmc_minor_phase")

                if self.verbose:
                    print(f"⚠️  Aircraft {aircraft.id} NMC: Minor phase threshold ({aircraft.hours_since_minor_phase:.1f} hrs)")

    def _calculate_daily_metrics(self, mission: Dict, mission_success: bool) -> Dict:
        """Calculate daily performance metrics.

        Args:
            mission: Mission dict
            mission_success: Whether mission was successful

        Returns:
            Dict with daily metrics
        """
        # Count aircraft by status
        fmc_count = sum(1 for a in self.aircraft if a.status == "FMC")
        nmc_count = sum(1 for a in self.aircraft if a.status == "NMC")
        maintenance_count = sum(1 for a in self.aircraft if a.in_maintenance)

        # Calculate OR (FMC / Total)
        current_or = fmc_count / self.config.num_aircraft

        # Flight hours today
        daily_flight_hours = sum(a.todays_flight_hours for a in self.aircraft)

        return {
            "current_or": current_or,
            "fmc_count": fmc_count,
            "nmc_count": nmc_count,
            "maintenance_count": maintenance_count,
            "mission_success": mission_success,
            "mission_required": mission["required_aircraft"],
            "mission_tempo": mission["tempo"],
            "daily_flight_hours": daily_flight_hours,
        }

    def get_state(self) -> Dict:
        """Get current observable state.

        Returns:
            State dictionary with fleet-level and per-aircraft information
        """
        fiscal_day = self.fiscal_calendar.get_day(self.current_day) if self.current_day > 0 else None
        mission_forecast = self.mission_generator.get_forecast() if self.mission_generator else []

        # Calculate fleet-level DA line metrics
        fleet_da_metrics = self.fleet_metrics.calculate_all_metrics(self.aircraft)

        return {
            "sim_day": self.current_day,
            "fiscal_day": fiscal_day,
            "aircraft": [deepcopy(a) for a in self.aircraft],
            "tokens_available": self.token_tracker.tokens_available if self.token_tracker else {},
            "maintenance_slots": self.maintenance_system.get_slot_status() if self.maintenance_system else {},
            "mission_forecast": mission_forecast,
            "current_or": self.current_or,
            # Fleet-level DA line metrics
            "fleet_metrics": {
                "total_bank_hours": fleet_da_metrics["total_bank_hours"],
                "avg_bank_hours": fleet_da_metrics["avg_bank_hours"],
                "clustering": fleet_da_metrics["clustering"],
                "capacity_strain": fleet_da_metrics["capacity_strain"],
                "da_line_penalty": fleet_da_metrics["penalty"],
                "per_aircraft_deviance": fleet_da_metrics["per_aircraft_deviance"],
            },
        }

    def get_final_metrics(self) -> Dict:
        """Calculate final episode metrics.

        Returns:
            Dict with episode summary
        """
        mean_or = np.mean(self.or_history) if self.or_history else 0.0
        final_or = self.or_history[-1] if self.or_history else 0.0
        mission_success_rate = np.mean(self.mission_success_history) if self.mission_success_history else 0.0

        # Count NMC days
        total_nmc_days = sum(
            sum(1 for a in self.aircraft if a.status == "NMC")
            for _ in range(len(self.or_history))
        )

        return {
            "sim_days": self.current_day,
            "mean_or": mean_or,
            "final_or": final_or,
            "mission_success_rate": mission_success_rate,
            "total_flight_hours": self.total_flight_hours,
            "total_inflight_failures": self.total_inflight_failures,
            "total_nmc_days": total_nmc_days,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Environment(day={self.current_day}/{self.config.sim_days}, "
            f"OR={self.current_or:.2%}, "
            f"aircraft={self.config.num_aircraft})"
        )
