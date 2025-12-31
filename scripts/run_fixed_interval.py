#!/usr/bin/env python3
"""Run fixed-interval preventive maintenance baseline evaluation.

Usage:
    python scripts/run_fixed_interval.py --interval 25 --episodes 1000
"""

import argparse
import json
import sys
import statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.environment import Environment
from policy.fixed_interval_policy import FixedIntervalPolicy
from utils.config import load_config_from_yaml


def run_episode(env: Environment, policy: FixedIntervalPolicy, verbose: bool = False) -> dict:
    """Run a single episode."""
    state = env.reset()
    done = False

    while not done:
        actions = policy.decide(state)
        state, reward, done, info = env.step(actions)

    return env.get_final_metrics()


def main():
    parser = argparse.ArgumentParser(description="Run fixed-interval baseline")
    parser.add_argument("--interval", type=float, default=25.0, help="Preventive interval in hours")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Sim config")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output-dir", type=str, default="results/fixed_interval", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"interval_{args.interval}h_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fixed-Interval Baseline Evaluation")
    print(f"=" * 50)
    print(f"Interval: {args.interval} hours")
    print(f"Episodes: {args.episodes}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"=" * 50)

    # Load config and create environment
    config = load_config_from_yaml(args.config)
    env = Environment(config)
    policy = FixedIntervalPolicy(interval_hours=args.interval, verbose=args.verbose)

    # Run episodes
    all_metrics = []
    for ep in range(1, args.episodes + 1):
        metrics = run_episode(env, policy, verbose=args.verbose)
        all_metrics.append(metrics)

        if ep % 100 == 0 or ep == args.episodes:
            mean_or = statistics.mean([m["mean_or"] for m in all_metrics])
            mean_fp = statistics.mean([m["mission_success_rate"] for m in all_metrics])
            print(f"Episode {ep:4d}/{args.episodes}: OR={mean_or:.2%}, FP={mean_fp:.2%}")

    # Calculate final statistics
    mean_ors = [m["mean_or"] for m in all_metrics]
    mission_rates = [m["mission_success_rate"] for m in all_metrics]
    inflight_failures = [m["total_inflight_failures"] for m in all_metrics]

    results = {
        "policy": f"FixedInterval_{args.interval}h",
        "interval_hours": args.interval,
        "episodes": args.episodes,
        "config": args.config,
        "mean_or": {
            "mean": statistics.mean(mean_ors),
            "std": statistics.stdev(mean_ors) if len(mean_ors) > 1 else 0,
            "min": min(mean_ors),
            "max": max(mean_ors)
        },
        "mission_success_rate": {
            "mean": statistics.mean(mission_rates),
            "std": statistics.stdev(mission_rates) if len(mission_rates) > 1 else 0,
            "min": min(mission_rates),
            "max": max(mission_rates)
        },
        "inflight_failures": {
            "mean": statistics.mean(inflight_failures),
            "std": statistics.stdev(inflight_failures) if len(inflight_failures) > 1 else 0,
            "total": sum(inflight_failures)
        }
    }

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"FINAL RESULTS: Fixed Interval = {args.interval}h")
    print(f"{'=' * 50}")
    print(f"Mean OR:           {results['mean_or']['mean']:.2%} +/- {results['mean_or']['std']:.2%}")
    print(f"Mission Success:   {results['mission_success_rate']['mean']:.2%} +/- {results['mission_success_rate']['std']:.2%}")
    print(f"Inflight Failures: {results['inflight_failures']['mean']:.1f} per episode")
    print(f"{'=' * 50}")

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
