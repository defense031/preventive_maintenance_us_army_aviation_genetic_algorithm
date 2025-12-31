"""Mission tempo generator with stochastic demand.

Ported from: aviation_hierarchical_sim_v2/core_v2/environment.R (lines 241-1779)

Implements 8-state Markov chain for mission tempo with tempo-dependent aircraft requirements.

States: none, very_low, low, moderate_low, moderate, moderate_high, high, very_high
Tempo regimes: baseline, high_optempo, low_optempo
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from math import ceil


@dataclass
class MissionForecast:
    """Single day in mission forecast."""

    day_offset: int  # Days ahead (0 = today, 1 = tomorrow, ..., 13 = day 14)
    tempo: str  # Tempo state
    required_aircraft: int  # Number of aircraft needed
    aircraft_fraction: float  # Fraction used to calculate requirement


class MissionGenerator:
    """Stochastic mission tempo generator.

    Uses 8-state Markov chain to generate time-varying mission demand.
    """

    # Tempo states
    TEMPO_STATES = [
        "none",
        "very_low",
        "low",
        "moderate_low",
        "moderate",
        "moderate_high",
        "high",
        "very_high",
    ]

    # Tempo to aircraft requirement mapping
    # Fractions chosen so ceiling(fraction × 8) produces 0-7 aircraft
    TEMPO_REQUIREMENTS = {
        "none": 0.0,  # ceil(0.0 × 8) = 0 aircraft
        "very_low": 0.0625,  # ceil(0.5 × 8) = 1 aircraft
        "low": 0.1875,  # ceil(1.5 × 8) = 2 aircraft
        "moderate_low": 0.3125,  # ceil(2.5 × 8) = 3 aircraft
        "moderate": 0.4375,  # ceil(3.5 × 8) = 4 aircraft
        "moderate_high": 0.5625,  # ceil(4.5 × 8) = 5 aircraft
        "high": 0.6875,  # ceil(5.5 × 8) = 6 aircraft
        "very_high": 0.8125,  # ceil(6.5 × 8) = 7 aircraft
    }

    def __init__(
        self,
        num_aircraft: int = 8,
        tempo_regime: str = "baseline",
        forecast_horizon: int = 14,
        min_sortie_hours: float = 4.0,
        max_sortie_hours: float = 8.0,
        entropy_scale: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,
    ):
        """Initialize mission generator.

        Args:
            num_aircraft: Total fleet size
            tempo_regime: 'baseline', 'high_optempo', or 'low_optempo'
            forecast_horizon: Days of mission forecast (default 14)
            min_sortie_hours: Minimum flight hours per sortie
            max_sortie_hours: Maximum flight hours per sortie
            entropy_scale: Scale factor for transition matrix entropy (default 1.0).
                          Values > 1.0 flatten toward uniform (more uncertainty).
                          1.0 = original matrix, 2.0 = approx 2x entropy.
            rng: NumPy random generator
            verbose: Enable detailed logging
        """
        self.num_aircraft = num_aircraft
        self.tempo_regime = tempo_regime
        self.forecast_horizon = forecast_horizon
        self.min_sortie_hours = min_sortie_hours
        self.max_sortie_hours = max_sortie_hours
        self.entropy_scale = entropy_scale
        self.rng = rng if rng is not None else np.random.default_rng()
        self.verbose = verbose

        # Create transition matrix
        self.transition_matrix = self._create_transition_matrix(tempo_regime)

        # Apply entropy scaling if requested
        if entropy_scale != 1.0:
            self.transition_matrix = self._scale_transition_entropy(
                self.transition_matrix, entropy_scale
            )

        # Initialize tempo state (start at "low" - middle-low starting point)
        self.current_tempo_state = "low"

        # Mission forecast (rolling 14-day window)
        self.mission_forecast: List[MissionForecast] = []

    def _scale_transition_entropy(
        self,
        P: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """Scale transition matrix entropy by blending toward uniform.

        For each row, blends original probabilities toward uniform distribution:
            new_row = alpha * original + (1 - alpha) * uniform

        where alpha = 1 / scale. This increases entropy (more uncertainty).

        Args:
            P: Original 8x8 transition matrix
            scale: Entropy scale factor (1.0 = unchanged, 2.0 = approx 2x entropy)

        Returns:
            Scaled transition matrix with higher entropy

        Example:
            scale=2.0 gives alpha=0.5, so:
            new_row = 0.5 * original + 0.5 * uniform
        """
        if scale <= 0:
            raise ValueError(f"entropy_scale must be positive, got {scale}")

        if scale == 1.0:
            return P

        n_states = P.shape[0]
        uniform = np.ones(n_states) / n_states  # [0.125, 0.125, ..., 0.125]

        alpha = 1.0 / scale  # alpha=0.5 for scale=2.0
        P_scaled = np.zeros_like(P)

        for i in range(n_states):
            P_scaled[i, :] = alpha * P[i, :] + (1 - alpha) * uniform

        return P_scaled

    def _create_transition_matrix(self, tempo_regime: str) -> np.ndarray:
        """Create mission tempo transition matrix.

        Args:
            tempo_regime: 'baseline', 'high_optempo', or 'low_optempo'

        Returns:
            8x8 transition probability matrix
        """
        P = np.zeros((8, 8))

        if tempo_regime == "high_optempo":
            # HIGH OPTEMPO: E[demand] = 3.0 aircraft/day (1.5x baseline)
            # Steady-state: 37.6% utilization
            P[0, :] = [0.650, 0.200, 0.100, 0.050, 0.000, 0.000, 0.000, 0.000]
            P[1, :] = [0.200, 0.425, 0.250, 0.075, 0.050, 0.000, 0.000, 0.000]
            P[2, :] = [0.100, 0.225, 0.350, 0.200, 0.075, 0.050, 0.000, 0.000]
            P[3, :] = [0.050, 0.100, 0.200, 0.300, 0.200, 0.100, 0.050, 0.000]
            P[4, :] = [0.000, 0.050, 0.100, 0.175, 0.350, 0.175, 0.100, 0.050]
            P[5, :] = [0.000, 0.025, 0.050, 0.100, 0.200, 0.350, 0.200, 0.075]
            P[6, :] = [0.000, 0.000, 0.025, 0.050, 0.100, 0.250, 0.425, 0.150]
            P[7, :] = [0.000, 0.000, 0.000, 0.025, 0.050, 0.150, 0.400, 0.375]

        elif tempo_regime == "high_variance":
            # HIGH VARIANCE: E[demand] = 2.0 aircraft/day (same as baseline)
            # Entropy: 1.32x baseline (91% of max) - more unpredictable transitions
            # All state transitions possible (can jump from 0→7 or 7→0)
            P[0, :] = [0.375, 0.250, 0.125, 0.075, 0.050, 0.050, 0.050, 0.025]
            P[1, :] = [0.350, 0.275, 0.125, 0.075, 0.050, 0.050, 0.050, 0.025]
            P[2, :] = [0.325, 0.250, 0.125, 0.075, 0.075, 0.050, 0.050, 0.050]
            P[3, :] = [0.300, 0.200, 0.125, 0.125, 0.075, 0.075, 0.050, 0.050]
            P[4, :] = [0.225, 0.200, 0.125, 0.125, 0.125, 0.075, 0.075, 0.050]
            P[5, :] = [0.225, 0.175, 0.125, 0.100, 0.075, 0.125, 0.075, 0.100]
            P[6, :] = [0.225, 0.150, 0.125, 0.100, 0.075, 0.100, 0.100, 0.125]
            P[7, :] = [0.200, 0.150, 0.100, 0.100, 0.100, 0.100, 0.100, 0.150]

        else:  # baseline
            # BASELINE: E[demand] = 2.0 aircraft/day
            # Steady-state: 25.0% utilization
            P[0, :] = [0.700, 0.200, 0.075, 0.025, 0.000, 0.000, 0.000, 0.000]
            P[1, :] = [0.250, 0.450, 0.225, 0.050, 0.025, 0.000, 0.000, 0.000]
            P[2, :] = [0.125, 0.275, 0.375, 0.150, 0.050, 0.025, 0.000, 0.000]
            P[3, :] = [0.050, 0.100, 0.225, 0.325, 0.175, 0.100, 0.025, 0.000]
            P[4, :] = [0.025, 0.050, 0.125, 0.200, 0.350, 0.150, 0.075, 0.025]
            P[5, :] = [0.000, 0.025, 0.050, 0.125, 0.200, 0.350, 0.200, 0.050]
            P[6, :] = [0.000, 0.000, 0.025, 0.050, 0.125, 0.250, 0.400, 0.150]
            P[7, :] = [0.000, 0.000, 0.000, 0.025, 0.050, 0.150, 0.400, 0.375]

        return P

    def update_tempo_state(self) -> str:
        """Update mission tempo using transition matrix.

        Returns:
            New tempo state
        """
        # Get current state index
        current_idx = self.TEMPO_STATES.index(self.current_tempo_state)

        # Sample next state
        probs = self.transition_matrix[current_idx, :]
        new_idx = self.rng.choice(len(self.TEMPO_STATES), p=probs)
        new_tempo = self.TEMPO_STATES[new_idx]

        if self.verbose:
            print(f"Mission tempo: {self.current_tempo_state} -> {new_tempo}")

        self.current_tempo_state = new_tempo
        return new_tempo

    def generate_full_forecast(self) -> List[MissionForecast]:
        """Generate complete forecast (initialization or reset).

        Returns:
            List of MissionForecast for next forecast_horizon days
        """
        forecast = []
        tempo = self.current_tempo_state

        for day_offset in range(self.forecast_horizon):
            if day_offset > 0:
                # Sample next tempo state for future days
                tempo_idx = self.TEMPO_STATES.index(tempo)
                probs = self.transition_matrix[tempo_idx, :]
                next_idx = self.rng.choice(len(self.TEMPO_STATES), p=probs)
                tempo = self.TEMPO_STATES[next_idx]

            # Calculate required aircraft
            aircraft_fraction = self.TEMPO_REQUIREMENTS[tempo]
            required_aircraft = ceil(aircraft_fraction * self.num_aircraft)

            forecast.append(
                MissionForecast(
                    day_offset=day_offset,
                    tempo=tempo,
                    required_aircraft=required_aircraft,
                    aircraft_fraction=aircraft_fraction,
                )
            )

        return forecast

    def update_rolling_forecast(self) -> List[MissionForecast]:
        """Update rolling forecast (shift and add one new day).

        Returns:
            Updated forecast
        """
        if not self.mission_forecast:
            # No existing forecast, generate full one
            return self.generate_full_forecast()

        # Shift existing forecast: day 2 becomes day 1, etc.
        new_forecast = []
        for i in range(1, len(self.mission_forecast)):
            shifted = self.mission_forecast[i]
            # Create new instance with updated offset
            new_forecast.append(
                MissionForecast(
                    day_offset=shifted.day_offset - 1,
                    tempo=shifted.tempo,
                    required_aircraft=shifted.required_aircraft,
                    aircraft_fraction=shifted.aircraft_fraction,
                )
            )

        # Generate one new day at the end (day 14)
        if len(self.mission_forecast) >= self.forecast_horizon:
            last_tempo = self.mission_forecast[-1].tempo
        else:
            last_tempo = self.current_tempo_state

        # Sample next tempo for new day
        tempo_idx = self.TEMPO_STATES.index(last_tempo)
        probs = self.transition_matrix[tempo_idx, :]
        next_idx = self.rng.choice(len(self.TEMPO_STATES), p=probs)
        new_tempo = self.TEMPO_STATES[next_idx]

        # Add new day
        aircraft_fraction = self.TEMPO_REQUIREMENTS[new_tempo]
        required_aircraft = ceil(aircraft_fraction * self.num_aircraft)

        new_forecast.append(
            MissionForecast(
                day_offset=self.forecast_horizon - 1,
                tempo=new_tempo,
                required_aircraft=required_aircraft,
                aircraft_fraction=aircraft_fraction,
            )
        )

        return new_forecast

    def get_daily_mission(self, sim_day: int) -> Dict:
        """Get mission requirement for today.

        Args:
            sim_day: Current simulation day

        Returns:
            Dict with 'required_aircraft', 'tempo', 'hours_per_sortie'
        """
        # Initialize or update forecast
        if not self.mission_forecast or sim_day == 1:
            self.mission_forecast = self.generate_full_forecast()
        else:
            # Update tempo state and rolling forecast
            self.update_tempo_state()
            self.mission_forecast = self.update_rolling_forecast()

        # Today's mission is day_offset=0
        today = self.mission_forecast[0]

        # Sample flight hours per sortie
        hours_per_sortie = self.rng.uniform(self.min_sortie_hours, self.max_sortie_hours)

        return {
            "required_aircraft": today.required_aircraft,
            "tempo": today.tempo,
            "hours_per_sortie": hours_per_sortie,
            "aircraft_fraction": today.aircraft_fraction,
        }

    def get_forecast(self, days_ahead: int = 14) -> List[MissionForecast]:
        """Get mission forecast.

        Args:
            days_ahead: Number of days to return (default 14)

        Returns:
            List of MissionForecast
        """
        if not self.mission_forecast:
            self.mission_forecast = self.generate_full_forecast()

        return self.mission_forecast[: min(days_ahead, len(self.mission_forecast))]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MissionGenerator(regime={self.tempo_regime}, "
            f"current_tempo={self.current_tempo_state}, "
            f"forecast_days={len(self.mission_forecast)})"
        )
