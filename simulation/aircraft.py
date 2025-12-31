"""Aircraft state dataclass.

Ported from: aviation_hierarchical_sim_v2/core_v2/environment.R (lines 200-250)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Aircraft:
    """Single aircraft state container.

    Represents the complete state of one aircraft in the 8-aircraft fleet.
    All times/hours are in flight hours unless otherwise noted.

    Attributes:
        id: Aircraft identifier (0-7)
        status: FMC (Fully Mission Capable) or NMC (Not Mission Capable)

        # RUL (Remaining Useful Life) - Component failure modeling
        true_rul: Ground truth RUL (hidden from policy, only for simulation)
        observed_rul: Noisy sensor reading (what policy observes)

        # Flight hours tracking
        total_flight_hours: Cumulative flight hours since fleet inception
        hours_since_minor_phase: Hours flown since last minor phase maintenance
        hours_since_major_phase: Hours flown since last major phase maintenance
        hours_since_preventive: Hours flown since last preventive/reactive maintenance

        # NMC reason tracking
        nmc_reason: Why aircraft is NMC (None if FMC)
                    Valid values: 'nmc_rul', 'nmc_minor_phase', 'nmc_major_phase'

        # Daily activity tracking (reset each day)
        flew_today: Whether aircraft flew today
        todays_flight_hours: Hours flown today (0 if didn't fly)
        todays_decision: Action taken today (fly, hold, maintenance type)

        # Maintenance state tracking
        in_maintenance: Whether aircraft is currently in maintenance
        maintenance_type: Type of maintenance if in_maintenance=True
                         Valid values: 'preventive', 'reactive', 'minor_phase', 'major_phase'
        maintenance_days_remaining: Days until maintenance completes

        # Queue tracking (for maintenance slot management)
        in_queue: Whether aircraft is waiting for maintenance slot
        queue_position: Position in queue (None if not in queue)
        queue_type: Type of queue ('preventive' or 'phase')
    """

    # Identity
    id: int

    # Status
    status: str = "FMC"  # 'FMC' or 'NMC'

    # RUL (Component failure modeling)
    true_rul: float = 150.0  # Ground truth (hidden from policy)
    observed_rul: float = 150.0  # Noisy observation (policy sees this)

    # Flight hours tracking
    total_flight_hours: float = 0.0
    hours_since_minor_phase: float = 0.0
    hours_since_major_phase: float = 0.0
    hours_since_preventive: float = 0.0  # For fixed-interval baseline comparison

    # NMC reason
    nmc_reason: Optional[str] = None  # 'nmc_rul', 'nmc_minor_phase', 'nmc_major_phase'

    # Daily activity (reset each day)
    flew_today: bool = False
    todays_flight_hours: float = 0.0
    todays_decision: Optional[str] = None  # 'fly', 'hold', 'preventive_maintain', etc.

    # Daily maintenance event tracking (for detailed_daily_operations.csv)
    flight_failure_today: bool = False  # In-flight RUL failure
    maintenance_started_today: str = "none"  # "none", "reactive", "minor_phase", "major_phase", "preventive"
    maintenance_completed_today: str = "none"  # "none", "reactive", "minor_phase", "major_phase", "preventive"

    # Maintenance state
    in_maintenance: bool = False
    maintenance_type: Optional[str] = None  # 'preventive', 'reactive', 'minor_phase', 'major_phase'
    maintenance_days_remaining: int = 0

    # Queue state
    in_queue: bool = False
    queue_position: Optional[int] = None
    queue_type: Optional[str] = None  # 'preventive' or 'phase'

    def reset_daily_state(self) -> None:
        """Reset daily tracking variables at start of new day."""
        self.flew_today = False
        self.todays_flight_hours = 0.0
        self.todays_decision = None
        self.flight_failure_today = False
        self.maintenance_started_today = "none"
        self.maintenance_completed_today = "none"

    def add_flight_hours(self, hours: float) -> None:
        """Add flight hours and update all counters.

        Args:
            hours: Flight hours to add (typically 4-8 hours per sortie)
        """
        self.total_flight_hours += hours
        self.hours_since_minor_phase += hours
        self.hours_since_major_phase += hours
        self.hours_since_preventive += hours
        self.todays_flight_hours += hours
        self.flew_today = True

        # Decrement RUL (component wear)
        self.true_rul -= hours
        # Note: observed_rul will be recalculated with noise by environment

    def reset_phase_counters(self, maintenance_type: str) -> None:
        """Reset phase hour counters after phase maintenance.

        Args:
            maintenance_type: 'minor_phase' or 'major_phase'
        """
        if maintenance_type == 'minor_phase':
            self.hours_since_minor_phase = 0.0
        elif maintenance_type == 'major_phase':
            # Major phase resets both counters
            self.hours_since_minor_phase = 0.0
            self.hours_since_major_phase = 0.0

    def set_nmc(self, reason: str) -> None:
        """Mark aircraft as NMC with reason.

        Args:
            reason: Why aircraft is NMC ('nmc_rul', 'nmc_minor_phase', 'nmc_major_phase')
        """
        self.status = "NMC"
        self.nmc_reason = reason

    def set_fmc(self) -> None:
        """Mark aircraft as FMC (clear NMC status)."""
        self.status = "FMC"
        self.nmc_reason = None

    def start_maintenance(self, maintenance_type: str, duration_days: int) -> None:
        """Begin maintenance period.

        Args:
            maintenance_type: Type of maintenance
            duration_days: Number of days maintenance will take
        """
        self.in_maintenance = True
        self.maintenance_type = maintenance_type
        self.maintenance_days_remaining = duration_days

        # Remove from queue if in one
        self.in_queue = False
        self.queue_position = None
        self.queue_type = None

    def advance_maintenance(self) -> bool:
        """Advance maintenance by one day.

        Returns:
            True if maintenance completed, False if still in progress
        """
        if not self.in_maintenance:
            return False

        self.maintenance_days_remaining -= 1

        if self.maintenance_days_remaining <= 0:
            # Maintenance complete
            self.in_maintenance = False
            maint_type = self.maintenance_type
            self.maintenance_type = None
            self.maintenance_days_remaining = 0
            return True

        return False

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Aircraft(id={self.id}, status={self.status}, "
            f"true_rul={self.true_rul:.1f}, observed_rul={self.observed_rul:.1f}, "
            f"hours_major={self.hours_since_major_phase:.1f})"
        )
