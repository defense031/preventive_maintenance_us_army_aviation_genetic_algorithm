#!/usr/bin/env python3
"""Run baseline simulation with default configuration.

Usage:
    python scripts/run_baseline.py --config config/default.yaml --output-dir results/baseline
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.environment import Environment
from policy.baseline_policy import BaselinePolicy
from utils.config import load_config_from_yaml
from utils.data_collector import create_data_collector


def run_episode(env: Environment, policy: BaselinePolicy, episode_num: int,
                data_collector=None, verbose: bool = False) -> dict:
    """Run a single episode.

    Args:
        env: Environment instance
        policy: Policy instance
        episode_num: Episode number
        data_collector: Optional DataCollector for logging
        verbose: Enable progress logging

    Returns:
        Episode metrics dict
    """
    # Start episode data collection
    if data_collector:
        data_collector.start_episode(episode_num, env.config.sim_days)

    state = env.reset()
    done = False
    day = 0

    if verbose:
        print(f"\n🚀 Starting episode: {env.config.sim_days} days")

    while not done:
        day += 1

        # Policy decides actions
        actions = policy.decide(state)

        # Execute step
        state, reward, done, info = env.step(actions)

        # Collect daily data
        if data_collector:
            data_collector.collect_daily_data(
                day=day,
                state=state,
                actions=actions,
                tokens_available=env.token_tracker.tokens_available
            )

        # Progress logging
        if verbose and day % 30 == 0:
            metrics = info["metrics"]
            print(
                f"Day {day:3d}/{env.config.sim_days}: "
                f"OR={metrics['current_or']:.1%}, "
                f"Mission={'✓' if info['mission_success'] else '✗'} "
                f"({info['flying_aircraft']}/{info['required_aircraft']}), "
                f"Flight hrs={metrics['daily_flight_hours']:.1f}"
            )

    # Get final metrics
    final_metrics = env.get_final_metrics()

    # Finish episode data collection
    if data_collector:
        data_collector.finish_episode(final_metrics)

    if verbose:
        print(f"\n✅ Episode complete!")
        print(f"   Mean OR: {final_metrics['mean_or']:.2%}")
        print(f"   Final OR: {final_metrics['final_or']:.2%}")
        print(f"   Mission Success Rate: {final_metrics['mission_success_rate']:.2%}")
        print(f"   Total Flight Hours: {final_metrics['total_flight_hours']:.1f}")
        print(f"   Inflight Failures: {final_metrics['total_inflight_failures']}")

    return final_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run baseline aviation simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed progress logging"
    )
    parser.add_argument(
        "--use-database",
        action="store_true",
        help="Log to SQLite database"
    )

    args = parser.parse_args()

    # Create timestamped output directory for this test run
    # Format: results/baseline_20251119_220532_10ep_seed42
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{Path(args.output_dir).name}_{timestamp}_{args.episodes}ep"
    if args.seed is not None:
        run_name += f"_seed{args.seed}"

    base_dir = Path(args.output_dir).parent
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")

    # Load configuration
    print(f"📋 Loading configuration from {args.config}")
    config = load_config_from_yaml(args.config)

    # Override seed if provided
    if args.seed is not None:
        config.seed = args.seed

    # Create DataCollector if database logging requested
    data_collector = None
    if args.use_database:
        db_path = str(output_dir / "simulation_data.db")
        data_collector = create_data_collector(
            db_path=db_path,
            session_id=1,
            enabled=True,
            verbose=args.verbose
        )
        print(f"🗄️  Database logging enabled: {db_path}")

    # Initialize environment (no database code - DataCollector handles it)
    print(f"🏗️  Initializing environment ({config.num_aircraft} aircraft, {config.sim_days} days)")
    env = Environment(config)

    # Initialize policy
    policy = BaselinePolicy(verbose=args.verbose)
    print(f"🎯 Using policy: {policy}")

    # Run episodes
    all_metrics = []
    for episode_num in range(1, args.episodes + 1):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_num}/{args.episodes}")
        print(f"{'='*60}")

        # Run episode with data collector
        metrics = run_episode(env, policy, episode_num,
                             data_collector=data_collector,
                             verbose=args.verbose)
        all_metrics.append(metrics)

        # Export episode JSON
        json_path = output_dir / f"episode_{episode_num:03d}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Saved metrics to {json_path}")

    # Close data collector
    if data_collector:
        data_collector.close()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes run: {len(all_metrics)}")
    if len(all_metrics) > 1:
        import statistics
        mean_ors = [m["mean_or"] for m in all_metrics]
        mission_rates = [m["total_mission_success_rate"] for m in all_metrics]
        print(f"Mean OR: {statistics.mean(mean_ors):.2%} ± {statistics.stdev(mean_ors):.2%}")
        print(f"Mission Success: {statistics.mean(mission_rates):.2%} ± {statistics.stdev(mission_rates):.2%}")

    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
