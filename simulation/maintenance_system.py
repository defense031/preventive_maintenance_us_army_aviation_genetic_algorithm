"""Slot-based maintenance processing system.

Ported from: aviation_hierarchical_sim_v2/core_v2/environment.R (lines 2603-2888)

Architecture:
- 2 Preventive/Reactive slots (shared pool for both types)
- 1 Phase slot (handles both minor and major phase maintenance)
- NO QUEUEING - Immediate allocation only (action masking prevents invalid requests)
- Uniform duration sampling by maintenance type
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from simulation.aircraft import Aircraft
from utils.config import SimulationConfig


@dataclass
class MaintenanceSlot:
    """Single maintenance slot state."""

    aircraft_id: Optional[int] = None
    maintenance_type: Optional[str] = None
    days_remaining: int = 0

    def is_occupied(self) -> bool:
        """Check if slot is currently occupied."""
        return self.aircraft_id is not None

    def clear(self) -> None:
        """Clear slot after maintenance completion."""
        self.aircraft_id = None
        self.maintenance_type = None
        self.days_remaining = 0


@dataclass
class MaintenanceSlots:
    """All maintenance slots for the fleet."""

    # 2 preventive/reactive slots (dual-purpose)
    preventive_slot1: MaintenanceSlot = field(default_factory=MaintenanceSlot)
    preventive_slot2: MaintenanceSlot = field(default_factory=MaintenanceSlot)

    # 1 phase slot (handles minor and major)
    phase_slot: MaintenanceSlot = field(default_factory=MaintenanceSlot)

    def get_available_preventive_slot(self) -> Optional[str]:
        """Find first available preventive/reactive slot.

        Returns:
            Slot name ('preventive_slot1' or 'preventive_slot2') or None
        """
        if not self.preventive_slot1.is_occupied():
            return "preventive_slot1"
        elif not self.preventive_slot2.is_occupied():
            return "preventive_slot2"
        return None

    def get_slot(self, slot_name: str) -> MaintenanceSlot:
        """Get slot by name."""
        return getattr(self, slot_name)

    def count_occupied(self) -> int:
        """Count total occupied slots."""
        return sum(
            [
                self.preventive_slot1.is_occupied(),
                self.preventive_slot2.is_occupied(),
                self.phase_slot.is_occupied(),
            ]
        )


class MaintenanceSystem:
    """Slot-based maintenance processing system.

    Manages maintenance slot allocation, duration tracking, and completion.
    """

    def __init__(self, config: SimulationConfig, rng: Optional[np.random.Generator] = None, verbose: bool = False):
        """Initialize maintenance system.

        Args:
            config: Simulation configuration
            rng: NumPy random generator
            verbose: Enable detailed logging
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.verbose = verbose

        # Initialize slots
        self.slots = MaintenanceSlots()

        # Maintenance type to slot type mapping
        self.slot_type_map = {
            "preventive": "preventive",
            "reactive": "preventive",
            "minor_phase": "phase",
            "major_phase": "phase",
        }

    def _sample_maintenance_duration(self, maintenance_type: str) -> int:
        """Sample maintenance duration from uniform distribution.

        Args:
            maintenance_type: Type of maintenance

        Returns:
            Duration in days
        """
        return self.config.maintenance_durations.sample_duration(maintenance_type, self.rng)

    def check_slot_available(self, maintenance_type: str) -> bool:
        """Check if a slot is available for given maintenance type.

        Args:
            maintenance_type: Type of maintenance

        Returns:
            True if slot available
        """
        slot_type = self.slot_type_map.get(maintenance_type)

        if slot_type == "preventive":
            return self.slots.get_available_preventive_slot() is not None
        elif slot_type == "phase":
            return not self.slots.phase_slot.is_occupied()

        return False

    def start_maintenance(
        self, aircraft: Aircraft, maintenance_type: str
    ) -> bool:
        """Start maintenance for an aircraft (immediate allocation).

        Args:
            aircraft: Aircraft to enter maintenance
            maintenance_type: Type of maintenance

        Returns:
            True if successfully started, False if slot unavailable
        """
        # Determine slot type
        slot_type = self.slot_type_map.get(maintenance_type)
        if slot_type is None:
            if self.verbose:
                print(f"⚠️  Invalid maintenance type: {maintenance_type}")
            return False

        # Check slot availability
        if not self.check_slot_available(maintenance_type):
            if self.verbose:
                print(f"⚠️  No available {slot_type} slot for aircraft {aircraft.id}")
            return False

        # Sample duration
        duration = self._sample_maintenance_duration(maintenance_type)

        # Allocate slot
        if slot_type == "preventive":
            # Find first available preventive/reactive slot
            slot_name = self.slots.get_available_preventive_slot()
            if slot_name is None:
                return False

            slot = self.slots.get_slot(slot_name)
            slot.aircraft_id = aircraft.id
            slot.maintenance_type = maintenance_type
            slot.days_remaining = duration

        elif slot_type == "phase":
            # Use dedicated phase slot
            self.slots.phase_slot.aircraft_id = aircraft.id
            self.slots.phase_slot.maintenance_type = maintenance_type
            self.slots.phase_slot.days_remaining = duration

        # Update aircraft state
        aircraft.start_maintenance(maintenance_type, duration)

        if self.verbose:
            print(
                f"🔧 Aircraft {aircraft.id} starting {maintenance_type} "
                f"maintenance ({duration} days)"
            )

        return True

    def process_daily_maintenance(
        self, aircraft_list: List[Aircraft]
    ) -> List[int]:
        """Process daily maintenance operations (countdown and completions).

        Args:
            aircraft_list: List of all aircraft

        Returns:
            List of aircraft IDs that completed maintenance today
        """
        completed_aircraft_ids = []

        # Process all slots
        all_slots = [
            ("preventive_slot1", self.slots.preventive_slot1),
            ("preventive_slot2", self.slots.preventive_slot2),
            ("phase_slot", self.slots.phase_slot),
        ]

        for slot_name, slot in all_slots:
            if not slot.is_occupied():
                continue

            # Decrement days remaining
            slot.days_remaining -= 1

            # Check for completion
            if slot.days_remaining <= 0:
                aircraft_id = slot.aircraft_id
                maintenance_type = slot.maintenance_type

                # Find aircraft
                aircraft = aircraft_list[aircraft_id]

                # Complete maintenance
                self._complete_maintenance(aircraft, maintenance_type)
                completed_aircraft_ids.append(aircraft_id)

                if self.verbose:
                    print(
                        f"✅ Aircraft {aircraft_id} completed {maintenance_type} maintenance"
                    )

                # Clear slot
                slot.clear()

        return completed_aircraft_ids

    def _complete_maintenance(self, aircraft: Aircraft, maintenance_type: str) -> None:
        """Complete maintenance for an aircraft.

        Args:
            aircraft: Aircraft completing maintenance
            maintenance_type: Type of maintenance completed
        """
        # Set completion flag (for detailed_daily_operations.csv)
        aircraft.maintenance_completed_today = maintenance_type

        # Reset RUL (all maintenance types reset RUL)
        new_rul = self.config.rul.sample_reset_rul(self.rng)
        aircraft.true_rul = new_rul
        aircraft.observed_rul = self.config.rul.add_observation_noise(new_rul, self.rng)

        # Reset phase counters
        aircraft.reset_phase_counters(maintenance_type)

        # Reset preventive interval counter (any maintenance resets the clock)
        aircraft.hours_since_preventive = 0.0

        # Set aircraft back to FMC
        aircraft.set_fmc()

        # Mark maintenance completion
        aircraft.in_maintenance = False
        aircraft.maintenance_type = None
        aircraft.maintenance_days_remaining = 0

    def get_slot_status(self) -> Dict:
        """Get current slot occupancy status.

        Returns:
            Dict with slot availability counts
        """
        return {
            "preventive_available": 2 - sum(
                [
                    self.slots.preventive_slot1.is_occupied(),
                    self.slots.preventive_slot2.is_occupied(),
                ]
            ),
            "phase_available": 0 if self.slots.phase_slot.is_occupied() else 1,
            "total_occupied": self.slots.count_occupied(),
            "total_capacity": 3,
        }

    def get_aircraft_in_maintenance(self) -> List[int]:
        """Get list of aircraft currently in maintenance.

        Returns:
            List of aircraft IDs
        """
        aircraft_ids = []

        if self.slots.preventive_slot1.is_occupied():
            aircraft_ids.append(self.slots.preventive_slot1.aircraft_id)
        if self.slots.preventive_slot2.is_occupied():
            aircraft_ids.append(self.slots.preventive_slot2.aircraft_id)
        if self.slots.phase_slot.is_occupied():
            aircraft_ids.append(self.slots.phase_slot.aircraft_id)

        return aircraft_ids

    def __repr__(self) -> str:
        """String representation."""
        status = self.get_slot_status()
        return (
            f"MaintenanceSystem(preventive_slots: {status['preventive_available']}/2, "
            f"phase_slots: {status['phase_available']}/1, "
            f"occupied: {status['total_occupied']}/3)"
        )
