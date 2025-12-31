"""Simulation configuration with Pydantic validation.

Ported from: aviation_hierarchical_sim_v2/parameters_v2/parameter_registry_shell.R
"""

from typing import Dict, Tuple, Optional, ClassVar
from pydantic import BaseModel, Field, field_validator


class MaintenanceDurationConfig(BaseModel):
    """Maintenance duration ranges (in days).

    Preventive: Uniform[1, 4] days - predictable, parts pre-ordered
    Reactive: Log-Normal(μ=2.30, σ=0.43) - right-skewed for supply chain uncertainty
              Mean ≈ 11 days, Median ≈ 10 days, Mode ≈ 8 days
    Phase: Uniform distributions as before
    """

    preventive_min: int = Field(default=1, ge=1, le=4)
    preventive_max: int = Field(default=4, ge=1, le=4)

    # Reactive uses log-normal, but keep these for backwards compatibility
    reactive_min: int = Field(default=5, ge=5, le=15)
    reactive_max: int = Field(default=15, ge=5, le=15)

    # Log-normal parameters for reactive maintenance
    reactive_lognormal_mu: float = Field(default=2.30, description="Log-normal μ parameter")
    reactive_lognormal_sigma: float = Field(default=0.43, description="Log-normal σ parameter")

    minor_phase_min: int = Field(default=8, ge=7, le=14)
    minor_phase_max: int = Field(default=14, ge=7, le=14)

    major_phase_min: int = Field(default=38, ge=35, le=50)
    major_phase_max: int = Field(default=50, ge=35, le=50)

    def sample_duration(self, maintenance_type: str, rng=None) -> int:
        """Sample maintenance duration.

        Preventive/Phase: Uniform distribution
        Reactive: Log-normal distribution (right-skewed for supply chain uncertainty)

        Args:
            maintenance_type: 'preventive', 'reactive', 'minor_phase', 'major_phase'
            rng: NumPy random generator (optional)

        Returns:
            Duration in days
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        if maintenance_type == "preventive":
            return rng.integers(self.preventive_min, self.preventive_max + 1)
        elif maintenance_type == "reactive":
            # Log-normal: right-skewed for parts delays
            # μ=2.30, σ=0.43 gives mean≈11, median≈10, mode≈8
            duration = rng.lognormal(self.reactive_lognormal_mu, self.reactive_lognormal_sigma)
            return max(1, int(round(duration)))  # Minimum 1 day
        elif maintenance_type == "minor_phase":
            return rng.integers(self.minor_phase_min, self.minor_phase_max + 1)
        elif maintenance_type == "major_phase":
            return rng.integers(self.major_phase_min, self.major_phase_max + 1)
        else:
            raise ValueError(f"Unknown maintenance type: {maintenance_type}")


class RULConfig(BaseModel):
    """RUL (Remaining Useful Life) parameters."""

    reset_min: float = Field(default=25.0, ge=10.0, le=300.0, description="Min RUL after maintenance (hours)")
    reset_max: float = Field(default=300.0, ge=10.0, le=300.0, description="Max RUL after maintenance (hours)")
    accuracy_mode: str = Field(
        default="very_high",
        description="RUL observation accuracy mode"
    )

    # Accuracy mode mapping
    ACCURACY_MODES: ClassVar[Dict[str, Optional[float]]] = {
        "perfect": 0.00,      # No noise
        "very_high": 0.05,    # ±5% noise
        "high": 0.10,         # ±10% noise
        "low": 0.25,          # ±25% noise
        "very_low": 0.50,     # ±50% noise
        "not_visible": None   # RUL not observable (policy blind to RUL)
    }

    @field_validator("accuracy_mode")
    @classmethod
    def validate_accuracy_mode(cls, v):
        """Ensure accuracy_mode is valid."""
        if v not in cls.ACCURACY_MODES:
            raise ValueError(
                f"Invalid accuracy_mode '{v}'. Must be one of: {list(cls.ACCURACY_MODES.keys())}"
            )
        return v

    def get_accuracy_value(self) -> Optional[float]:
        """Get the numeric accuracy value for current mode.

        Returns:
            Accuracy value (0.0-0.5) or None for 'not_visible'
        """
        return self.ACCURACY_MODES[self.accuracy_mode]

    def sample_reset_rul(self, rng=None) -> float:
        """Sample new RUL after maintenance completion.

        Args:
            rng: NumPy random generator (optional)

        Returns:
            RUL in flight hours
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        return rng.uniform(self.reset_min, self.reset_max)

    def add_observation_noise(self, true_rul: float, rng=None) -> float:
        """Add observation noise to true RUL using Gamma distribution.

        Noise Model: Gamma distribution with CV-based parameterization
        - Naturally bounded at 0 (no clipping needed)
        - Right-skewed (overestimates possible, underestimates limited)
        - Standard in reliability/lifetime modeling
        - Maintains differentiation at all RUL values

        Parameterization:
        - CV = accuracy (coefficient of variation equals sensor error rate)
        - Shape k = 1/CV², Scale θ = mean * CV² (so mean = k * θ)

        Accuracy Levels:
        - perfect (0.00): No noise, observed = true
        - very_high (0.05): ±5% sensor error
        - high (0.10): ±10% sensor error
        - low (0.25): ±25% sensor error
        - very_low (0.50): ±50% sensor error

        Examples (accuracy=0.10, CV=0.10):
        - true_rul=25 hrs  → std=2.5 hrs
        - true_rul=100 hrs → std=10.0 hrs
        - true_rul=200 hrs → std=20.0 hrs

        Args:
            true_rul: Ground truth RUL (hours)
            rng: NumPy random generator (optional)

        Returns:
            Observed RUL with noise (always >= 0, or None if not_visible mode)
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        accuracy = self.get_accuracy_value()

        # Not visible mode - return None (policy should handle this)
        if accuracy is None:
            return None

        # Perfect mode - return exact value
        if accuracy == 0.0:
            return true_rul

        # Handle edge case: true_rul <= 0
        if true_rul <= 0:
            return 0.0

        # Gamma parameterization via CV (coefficient of variation)
        # CV equals accuracy directly (no baseline added)
        cv = accuracy

        k = 1 / (cv ** 2)      # Shape parameter
        theta = true_rul / k   # Scale parameter (mean = k * theta = true_rul)

        # Sample from Gamma distribution
        observed = rng.gamma(k, theta)

        return observed


class PhaseMaintenanceConfig(BaseModel):
    """Phase maintenance thresholds."""

    minor_threshold: float = Field(default=250.0, description="Minor phase hours threshold")
    major_threshold: float = Field(default=500.0, description="Major phase hours threshold")

    minor_window: float = Field(default=100.0, description="Hours before minor threshold to open window")
    major_window: float = Field(default=100.0, description="Hours before major threshold to open window")


class FleetInitConfig(BaseModel):
    """Fleet initialization parameters for realistic mid-lifecycle starting conditions.

    Creates heterogeneous fleet with aircraft at different points in maintenance cycle.
    """

    major_hours_min: float = Field(default=25.0, ge=0.0, le=500.0, description="Min hours since major phase at start")
    major_hours_max: float = Field(default=300.0, ge=0.0, le=500.0, description="Max hours since major phase at start")
    rul_correlation: float = Field(default=0.3, ge=0.0, le=1.0, description="RUL degradation correlation factor (0=independent, 1=fully correlated)")


class TokenConfig(BaseModel):
    """Maintenance token budget parameters.

    Single pool with weighted consumption:
    - Preventive/Reactive: 1 token
    - Minor Phase: 3 tokens
    - Major Phase: 10 tokens
    - Quarterly allocation: annual_budget / 4 (remainder to Q1)
    - Tokens accumulate across quarters (don't expire)
    """

    annual_budget: int = Field(default=120, ge=30, le=500, description="Total annual token budget")
    preventive_weight: int = Field(default=1, ge=1, le=5, description="Tokens consumed per preventive maintenance")
    reactive_weight: int = Field(default=1, ge=1, le=5, description="Tokens consumed per reactive maintenance")
    minor_phase_weight: int = Field(default=3, ge=1, le=10, description="Tokens consumed per minor phase maintenance")
    major_phase_weight: int = Field(default=10, ge=5, le=20, description="Tokens consumed per major phase maintenance")


class MissionConfig(BaseModel):
    """Mission tempo and demand parameters."""

    tempo_regime: str = Field(default="baseline", description="Mission tempo regime (baseline, high_optempo, low_optempo)")

    # Flying hours per sortie range
    min_sortie_hours: float = Field(default=4.0, ge=2.0, le=6.0)
    max_sortie_hours: float = Field(default=8.0, ge=6.0, le=10.0)

    # Forecast parameters
    forecast_days: int = Field(default=14, description="Days of mission forecast to generate")

    # Transition matrix entropy scaling (for sensitivity analysis)
    transition_entropy_scale: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Scale factor for transition matrix entropy. 1.0=original, 2.0=approx 2x entropy (flatter)"
    )


class SimulationConfig(BaseModel):
    """Main simulation configuration."""

    # Fleet parameters
    num_aircraft: int = Field(default=8, ge=4, le=12, description="Number of aircraft in fleet")
    sim_days: int = Field(default=365, ge=30, le=1095, description="Simulation duration in days")

    # Random seed (None = random)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    # Fiscal year start
    fiscal_year_start_month: int = Field(default=10, ge=1, le=12, description="FY start month (10=October)")

    # Sub-configurations
    maintenance_durations: MaintenanceDurationConfig = Field(default_factory=MaintenanceDurationConfig)
    rul: RULConfig = Field(default_factory=RULConfig)
    phase: PhaseMaintenanceConfig = Field(default_factory=PhaseMaintenanceConfig)
    fleet_init: FleetInitConfig = Field(default_factory=FleetInitConfig)
    tokens: TokenConfig = Field(default_factory=TokenConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)

    # Maintenance capacity (slots)
    preventive_slots: int = Field(default=2, ge=1, le=4, description="Concurrent preventive maintenance slots")
    phase_slots: int = Field(default=1, ge=1, le=2, description="Concurrent phase maintenance slots")

    # Performance flags
    verbose: bool = Field(default=False, description="Print detailed logging")
    collect_detailed_data: bool = Field(default=False, description="Save detailed daily data to database")

    # Policy parameters (used by DecisionTreePolicy)
    rul_threshold: int = Field(default=100, ge=10, le=200, description="RUL threshold for preventive maintenance (hours)")

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v):
        """Ensure seed is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("Seed must be non-negative")
        return v

    def get_quarterly_tokens(self) -> int:
        """Calculate quarterly token allocation.

        Returns:
            Quarter token weight (annual_budget / 4)
        """
        return self.tokens.annual_budget // 4


class GAConfig(BaseModel):
    """Genetic Algorithm optimization configuration."""

    # Population parameters
    population_size: int = Field(default=20, ge=5, le=100)
    max_generations: int = Field(default=20, ge=5, le=200)
    elite_count: int = Field(default=2, ge=1, le=10)

    # Genetic operators
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.15, ge=0.0, le=0.5)

    # Early stopping
    stagnation_limit: int = Field(default=5, ge=2, le=20, description="Generations without improvement before stopping")
    min_fitness_delta: float = Field(default=0.001, ge=0.0, le=0.1, description="Minimum improvement threshold")

    # Evaluation
    episodes_per_policy: int = Field(default=50, ge=10, le=500, description="Monte Carlo episodes per policy")

    # Parallelization
    n_cores: int = Field(default=4, ge=1, le=64, description="Parallel cores for evaluation")

    # Fitness weights (must sum to 1.0)
    fitness_mission_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    fitness_or_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    fitness_hours_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("fitness_mission_weight", "fitness_or_weight", "fitness_hours_weight")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure fitness weights sum to 1.0."""
        # Note: This validator runs per-field, full validation done after all fields set
        return v

    def model_post_init(self, __context) -> None:
        """Validate that fitness weights sum to 1.0."""
        total = self.fitness_mission_weight + self.fitness_or_weight + self.fitness_hours_weight
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Fitness weights must sum to 1.0, got {total:.4f} "
                f"({self.fitness_mission_weight} + {self.fitness_or_weight} + {self.fitness_hours_weight})"
            )


def load_config_from_yaml(path: str) -> SimulationConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated SimulationConfig object
    """
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return SimulationConfig(**data)


def save_config_to_yaml(config: SimulationConfig, path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: SimulationConfig object
        path: Output YAML path
    """
    import yaml

    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
