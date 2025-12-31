"""
Abstract Base Class for Aircraft Maintenance Policies

Defines the interface contract that all policies must implement.
This enables plug-in architecture where policies can be swapped via configuration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BasePolicy(ABC):
    """
    Abstract base class for maintenance scheduling policies.

    All policy implementations must inherit from this class and implement
    the decide() method. This ensures a consistent interface for the
    simulation environment.

    Policy Lifecycle:
        1. Instantiate policy: policy = SomePolicy(**config)
        2. Optionally reset: policy.reset() (before each episode)
        3. Decide actions: actions = policy.decide(state)
        4. Repeat step 3 for each simulation day

    State Format (see STATE_VARIABLE_REFERENCE.md for complete details):
        {
            'sim_day': int,                      # Current day [1-365]
            'fiscal_day': FiscalDay,             # Fiscal calendar object
            'aircraft': List[Aircraft],          # 8 Aircraft objects (deepcopy)
            'current_or': float,                 # Operational Readiness [0.0-1.0]
            'tokens_available': Dict[str, int],  # Budget by maintenance type
            'maintenance_slots': Dict[str, int], # Available slots
            'mission_forecast': List[MissionForecast],  # 14-day rolling forecast
            'fleet_metrics': Dict[str, float]    # DA line clustering/capacity
        }

    Action Format:
        Dict[int, str] mapping aircraft_id (0-7) -> action_string

        Valid actions:
            - "fly"                    : Execute mission sortie
            - "hold"                   : No action (idle)
            - "reactive_maintain"      : Fix RUL failure
            - "minor_phase_maintain"   : 250-hour phase maintenance
            - "major_phase_maintain"   : 500-hour phase maintenance
            - "preventive_maintain"    : Proactive maintenance

    Constraints (validated by Environment):
        - NMC aircraft cannot fly
        - Aircraft in maintenance cannot take new actions
        - Reactive maintenance only valid if nmc_reason == "nmc_rul"
        - Minor phase only valid if nmc_reason == "nmc_minor_phase"
        - Major phase only valid if nmc_reason == "nmc_major_phase"
        - Invalid actions automatically downgrade to "hold"

    Example Implementation:
        >>> class MyPolicy(BasePolicy):
        ...     def decide(self, state: Dict) -> Dict[int, str]:
        ...         actions = {}
        ...         for aircraft in state['aircraft']:
        ...             if aircraft.status == 'FMC':
        ...                 actions[aircraft.id] = 'fly'
        ...             else:
        ...                 actions[aircraft.id] = 'hold'
        ...         return actions
    """

    def __init__(self, verbose: bool = False, **kwargs):
        """
        Initialize the policy.

        Args:
            verbose: If True, print decision-making information
            **kwargs: Policy-specific configuration parameters
        """
        self.verbose = verbose
        self.config = kwargs

    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> Dict[int, str]:
        """
        Generate maintenance and flight actions for all aircraft.

        This is the core method that all policies must implement. It takes
        the current simulation state and returns a dictionary mapping each
        aircraft ID to an action string.

        Args:
            state: Complete simulation state dictionary containing:
                - aircraft: List of Aircraft objects
                - mission_forecast: Mission demand forecast
                - tokens_available: Token budget
                - maintenance_slots: Available slots
                - fleet_metrics: Fleet health metrics
                - sim_day, fiscal_day: Temporal context
                - current_or: Current operational readiness

        Returns:
            actions: Dictionary mapping aircraft_id -> action_string
                Example: {0: "fly", 1: "fly", 2: "hold", 3: "reactive_maintain", ...}

        Raises:
            NotImplementedError: If subclass does not implement this method

        Notes:
            - Must return an action for ALL 8 aircraft (IDs 0-7)
            - Invalid actions will be automatically downgraded to "hold"
            - Environment handles token/slot availability checking
            - Decision time budget: ~1ms per call (365 calls per episode)
        """
        raise NotImplementedError("Subclasses must implement decide() method")

    def reset(self) -> None:
        """
        Reset policy state between episodes (optional).

        Override this method if your policy maintains internal state that
        should be reset between episodes (e.g., learning algorithms,
        memory buffers, cached statistics).

        Default implementation does nothing (stateless policies).

        Called automatically by evaluation scripts before each episode.
        """
        pass

    def get_name(self) -> str:
        """
        Return human-readable policy name.

        Override to provide descriptive name for logging and visualization.
        Default returns class name.

        Returns:
            Policy name string
        """
        return self.__class__.__name__

    def get_info(self) -> Dict[str, Any]:
        """
        Return policy metadata and configuration.

        Override to provide policy-specific information for logging,
        debugging, and reproducibility.

        Returns:
            Dictionary with policy metadata
        """
        return {
            'name': self.get_name(),
            'verbose': self.verbose,
            'config': self.config
        }

    def _validate_actions(self, actions: Dict[int, str]) -> Dict[int, str]:
        """
        Validate that actions dictionary is well-formed (optional helper).

        This is a convenience method for policy implementations to use.
        The Environment will also perform validation, but checking here
        can help with debugging during policy development.

        Args:
            actions: Proposed actions dictionary

        Returns:
            Validated actions (fills missing aircraft with "hold")

        Warnings:
            Logs warning if actions are malformed
        """
        # Ensure all 8 aircraft have actions
        validated = {}
        for aircraft_id in range(8):
            if aircraft_id in actions:
                validated[aircraft_id] = actions[aircraft_id]
            else:
                if self.verbose:
                    print(f"WARNING: No action for aircraft {aircraft_id}, defaulting to 'hold'")
                validated[aircraft_id] = "hold"

        return validated

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.get_name()}(verbose={self.verbose})"
