"""
Fitness Evaluator for Genetic Algorithm

Evaluates chromosome fitness by running simulation episodes and computing
a weighted combination of mission success, operational readiness, and flight hours.

The fitness function prioritizes mission success (70%) while balancing
operational readiness (15%) and flying efficiency (15%).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

from simulation.environment import Environment
from policy.chromosome import Chromosome
from policy.decision_tree_policy import DecisionTreePolicy
from policy.state_encoder import StateEncoder
from utils.config import SimulationConfig


class FitnessEvaluator:
    """Evaluate chromosome fitness through simulation.

    Runs multiple episodes with a chromosome-based policy and computes weighted
    fitness from mission success, operational readiness, and flight hours.

    Fitness Formula:
        fitness = w_ms × mission_success_rate +
                  w_or × mean_or +
                  w_fh × (total_flight_hours / baseline_max_flight_hours)

    Where:
        - w_ms = 0.70 (mission success weight)
        - w_or = 0.15 (operational readiness weight)
        - w_fh = 0.15 (flight hours weight)
        - baseline_max_flight_hours = 4563.55 (normalization constant)

    Attributes:
        sim_config: Simulation configuration
        feature_config_path: Path to feature encoder configuration
        weight_mission_success: Weight for mission success (default: 0.70)
        weight_or: Weight for operational readiness (default: 0.15)
        weight_flight_hours: Weight for flight hours (default: 0.15)
        baseline_max_flight_hours: Max flight hours for normalization (default: 4563.55)
        n_episodes: Number of evaluation episodes (default: 100)
        seed: Base random seed for reproducibility
        verbose: Enable detailed logging

    Example:
        >>> evaluator = FitnessEvaluator(
        ...     sim_config=config,
        ...     feature_config_path='config/features/simple_dt.yaml',
        ...     n_episodes=100,
        ...     seed=42
        ... )
        >>> chromosome = Chromosome.random(tree_depth=3, n_features=5)
        >>> fitness, metrics = evaluator.evaluate(chromosome)
        >>> print(f"Fitness: {fitness:.4f}")
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        feature_config_path: str = "config/features/simple_dt.yaml",
        weight_mission_success: float = 0.70,
        weight_or: float = 0.15,
        weight_flight_hours: float = 0.15,
        baseline_max_flight_hours: float = 4563.55,
        n_episodes: int = 100,
        seed: int = 42,
        verbose: bool = False
    ):
        """Initialize fitness evaluator.

        Args:
            sim_config: Simulation configuration object
            feature_config_path: Path to feature encoder YAML (default: simple_dt.yaml)
            weight_mission_success: Weight for mission success (default: 0.70)
            weight_or: Weight for mean OR (default: 0.15)
            weight_flight_hours: Weight for flight hours (default: 0.15)
            baseline_max_flight_hours: Normalizer for flight hours (default: 4563.55)
            n_episodes: Number of evaluation episodes (default: 100)
            seed: Base random seed for reproducibility
            verbose: Enable detailed logging

        Raises:
            ValueError: If weights don't sum to 1.0 or invalid parameters
            FileNotFoundError: If feature config not found
        """
        # Validate weights
        total_weight = weight_mission_success + weight_or + weight_flight_hours
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Fitness weights must sum to 1.0, got {total_weight:.4f}"
            )

        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")

        if baseline_max_flight_hours <= 0:
            raise ValueError(
                f"baseline_max_flight_hours must be positive, got {baseline_max_flight_hours}"
            )

        # Validate feature config exists
        feature_path = Path(feature_config_path)
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature config not found: {feature_config_path}"
            )

        self.sim_config = sim_config
        self.feature_config_path = str(feature_path)
        self.weight_mission_success = weight_mission_success
        self.weight_or = weight_or
        self.weight_flight_hours = weight_flight_hours
        self.baseline_max_flight_hours = baseline_max_flight_hours
        self.n_episodes = n_episodes
        self.seed = seed
        self.verbose = verbose

        # Create state encoder (reused across evaluations)
        self.encoder = StateEncoder(
            feature_config_path=self.feature_config_path,
            verbose=False  # Suppress encoder logging during fitness eval
        )

        # Random number generator for episode seeds
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        chromosome: Chromosome,
        episode_offset: int = 0
    ) -> Tuple[float, Dict]:
        """Evaluate chromosome fitness by running simulation episodes.

        Args:
            chromosome: Chromosome to evaluate
            episode_offset: Offset for episode numbering (for parallel eval)

        Returns:
            Tuple of (fitness_score, detailed_metrics)

            fitness_score: Scalar fitness in [0, 1] (higher is better)
            detailed_metrics: Dictionary containing:
                - mean_mission_success: Mean mission success rate across episodes
                - mean_or: Mean operational readiness across episodes
                - mean_flight_hours: Mean total flight hours per episode
                - std_mission_success: Std dev of mission success
                - std_or: Std dev of operational readiness
                - std_flight_hours: Std dev of flight hours
                - fitness: Overall fitness score
                - n_episodes: Number of episodes evaluated

        Example:
            >>> fitness, metrics = evaluator.evaluate(chromosome)
            >>> print(f"Mission Success: {metrics['mean_mission_success']:.2%}")
            >>> print(f"Mean OR: {metrics['mean_or']:.2%}")
        """
        # Validate chromosome
        try:
            chromosome.validate()
        except Exception as e:
            raise ValueError(f"Invalid chromosome: {e}")

        # Create policy from chromosome
        # Use rul_threshold from sim_config (required field - will fail if missing)
        rul_threshold = self.sim_config.rul_threshold
        policy = DecisionTreePolicy(
            chromosome=chromosome,
            encoder=self.encoder,
            rul_threshold=rul_threshold,
            verbose=False  # Suppress policy logging during fitness eval
        )

        # Run episodes and collect metrics
        episode_metrics = []

        for episode_idx in range(self.n_episodes):
            # Generate episode-specific seed
            episode_seed = self.seed + episode_offset + episode_idx

            # Create environment (fresh for each episode)
            env = Environment(self.sim_config, rng=np.random.default_rng(episode_seed))

            # Run episode
            metrics = self._run_episode(env, policy, episode_seed)
            episode_metrics.append(metrics)

            # Log progress periodically
            if self.verbose and (episode_idx + 1) % 10 == 0:
                print(
                    f"  Episode {episode_idx + 1}/{self.n_episodes}: "
                    f"MS={metrics['mission_success_rate']:.2%}, "
                    f"OR={metrics['mean_or']:.2%}, "
                    f"FH={metrics['total_flight_hours']:.1f}"
                )

        # Aggregate metrics across episodes
        mission_success_rates = [m['mission_success_rate'] for m in episode_metrics]
        mean_ors = [m['mean_or'] for m in episode_metrics]
        flight_hours = [m['total_flight_hours'] for m in episode_metrics]

        mean_mission_success = np.mean(mission_success_rates)
        mean_or = np.mean(mean_ors)
        mean_flight_hours = np.mean(flight_hours)

        std_mission_success = np.std(mission_success_rates)
        std_or = np.std(mean_ors)
        std_flight_hours = np.std(flight_hours)

        # Compute fitness
        fitness = self._compute_fitness(
            mission_success_rate=mean_mission_success,
            mean_or=mean_or,
            total_flight_hours=mean_flight_hours
        )

        # Compile detailed metrics
        detailed_metrics = {
            'mean_mission_success': mean_mission_success,
            'mean_or': mean_or,
            'mean_flight_hours': mean_flight_hours,
            'std_mission_success': std_mission_success,
            'std_or': std_or,
            'std_flight_hours': std_flight_hours,
            'fitness': fitness,
            'n_episodes': self.n_episodes,
            # Include raw episode data for debugging
            'episode_metrics': episode_metrics if self.verbose else None
        }

        if self.verbose:
            print(f"\n✅ Evaluation complete:")
            print(f"   Fitness: {fitness:.4f}")
            print(f"   Mission Success: {mean_mission_success:.2%} ± {std_mission_success:.2%}")
            print(f"   Mean OR: {mean_or:.2%} ± {std_or:.2%}")
            print(f"   Mean Flight Hours: {mean_flight_hours:.1f} ± {std_flight_hours:.1f}")

        return fitness, detailed_metrics

    def _run_episode(
        self,
        env: Environment,
        policy: DecisionTreePolicy,
        episode_seed: int
    ) -> Dict:
        """Run a single simulation episode.

        Args:
            env: Environment instance
            policy: Policy to use for decisions
            episode_seed: Random seed for this episode

        Returns:
            Episode metrics dictionary
        """
        # Reset environment
        state = env.reset(seed=episode_seed)
        done = False

        # Run episode loop
        while not done:
            # Policy decides actions
            actions = policy.decide(state)

            # Execute step
            state, reward, done, info = env.step(actions)

        # Get final metrics
        final_metrics = env.get_final_metrics()

        return final_metrics

    def _compute_fitness(
        self,
        mission_success_rate: float,
        mean_or: float,
        total_flight_hours: float
    ) -> float:
        """Compute weighted fitness score.

        Args:
            mission_success_rate: Mission success rate in [0, 1]
            mean_or: Mean operational readiness in [0, 1]
            total_flight_hours: Total flight hours (unnormalized)

        Returns:
            Fitness score in [0, 1]

        Formula:
            fitness = w_ms × mission_success +
                      w_or × mean_or +
                      w_fh × (flight_hours / baseline_max)
        """
        # Normalize flight hours
        normalized_flight_hours = total_flight_hours / self.baseline_max_flight_hours

        # Clip to [0, 1] range (in case exceeds baseline)
        normalized_flight_hours = np.clip(normalized_flight_hours, 0.0, 1.0)

        # Compute weighted sum
        fitness = (
            self.weight_mission_success * mission_success_rate +
            self.weight_or * mean_or +
            self.weight_flight_hours * normalized_flight_hours
        )

        return fitness

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"FitnessEvaluator("
            f"n_episodes={self.n_episodes}, "
            f"weights=[MS:{self.weight_mission_success:.2f}, "
            f"OR:{self.weight_or:.2f}, FH:{self.weight_flight_hours:.2f}], "
            f"baseline_fh={self.baseline_max_flight_hours:.1f})"
        )
