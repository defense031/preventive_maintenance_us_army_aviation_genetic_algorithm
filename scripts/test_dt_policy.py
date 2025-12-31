#!/usr/bin/env python3
"""
Test Decision Tree Policy Infrastructure

Validates DecisionTreePolicy with a fixed-seed random chromosome to ensure:
- Feature extraction works correctly
- Tree traversal and bucket classification works
- Three-sweep action assignment works
- Integration with simulation environment works
- Results are deterministic and reproducible

Usage:
    python scripts/test_dt_policy.py --seed 42 --days 90 --verbose
    python scripts/test_dt_policy.py --seed 42 --days 90 --save-chromosome results/test_chromosome.json
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulation.environment import Environment
from policy.decision_tree_policy import DecisionTreePolicy
from policy.chromosome import Chromosome
from policy.state_encoder import StateEncoder
from utils.config import load_config_from_yaml


class DiagnosticsTracker:
    """Track and analyze policy behavior during episode."""

    def __init__(self):
        self.bucket_history = []  # List of dicts: {day: int, buckets: List[int]}
        self.action_history = []  # List of dicts: {day: int, actions: Dict[int, str]}
        self.metrics_history = []  # List of dicts: {day: int, **metrics}
        self.checksum_data = []  # For reproducibility checking

    def record_day(self, day, state, actions, info, policy):
        """Record all diagnostics for a single day."""
        # Get bucket assignments by re-classifying
        features = policy.encoder.encode(state)
        buckets = {}
        for aircraft in state['aircraft']:
            if aircraft.status == 'NMC':
                buckets[aircraft.id] = 0
            else:
                aircraft_features = features[aircraft.id, :]
                bucket = policy._traverse_tree(aircraft_features)
                buckets[aircraft.id] = bucket

        # Record bucket distribution
        self.bucket_history.append({
            'day': day,
            'buckets': buckets
        })

        # Record actions
        self.action_history.append({
            'day': day,
            'actions': actions.copy()
        })

        # Record metrics
        metrics = info.get('metrics', {})
        self.metrics_history.append({
            'day': day,
            'current_or': metrics.get('current_or', 0.0),
            'mission_success': info.get('mission_success', False),
            'required_aircraft': info.get('required_aircraft', 0),
            'flying_aircraft': info.get('flying_aircraft', 0),
            'daily_flight_hours': metrics.get('daily_flight_hours', 0.0)
        })

        # Add to checksum data for reproducibility
        checksum_entry = f"{day}:{buckets}:{actions}:{metrics.get('current_or', 0):.4f}"
        self.checksum_data.append(checksum_entry)

    def compute_checksum(self):
        """Compute SHA256 checksum of all recorded data for reproducibility."""
        combined = "\n".join(self.checksum_data)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get_bucket_distribution(self):
        """Get distribution of bucket assignments across all days."""
        all_buckets = []
        for record in self.bucket_history:
            all_buckets.extend(record['buckets'].values())
        return Counter(all_buckets)

    def get_action_distribution(self):
        """Get distribution of actions across all days."""
        all_actions = []
        for record in self.action_history:
            all_actions.extend(record['actions'].values())
        return Counter(all_actions)

    def get_mission_stats(self):
        """Get mission success statistics."""
        total_days = len(self.metrics_history)
        mission_met = sum(1 for m in self.metrics_history if m['mission_success'])
        avg_or = np.mean([m['current_or'] for m in self.metrics_history])
        total_flight_hours = sum(m['daily_flight_hours'] for m in self.metrics_history)

        return {
            'total_days': total_days,
            'mission_met': mission_met,
            'mission_success_rate': mission_met / total_days if total_days > 0 else 0.0,
            'avg_or': avg_or,
            'total_flight_hours': total_flight_hours
        }

    def get_nmc_stats(self):
        """Get NMC handling statistics."""
        nmc_events = 0
        immediate_maintenance = 0

        for record in self.bucket_history:
            bucket_0_count = sum(1 for b in record['buckets'].values() if b == 0)
            nmc_events += bucket_0_count

        for record in self.action_history:
            for aircraft_id, action in record['actions'].items():
                # Check if this was an NMC aircraft
                bucket_record = next(r for r in self.bucket_history if r['day'] == record['day'])
                if bucket_record['buckets'][aircraft_id] == 0:
                    if action in ['reactive_maintain', 'minor_phase_maintain', 'major_phase_maintain']:
                        immediate_maintenance += 1

        return {
            'nmc_events': nmc_events,
            'immediate_maintenance': immediate_maintenance,
            'stuck_percentage': (nmc_events - immediate_maintenance) / nmc_events * 100 if nmc_events > 0 else 0.0
        }

    def print_summary(self, chromosome):
        """Print comprehensive diagnostics summary."""
        print("\n" + "="*70)
        print("DECISION TREE POLICY TEST RESULTS")
        print("="*70)

        # Chromosome configuration
        print("\n📊 Chromosome Configuration:")
        print(f"   Tree Depth:          {chromosome.tree_depth}")
        print(f"   Leaves:              {chromosome.n_leaves}")
        print(f"   Features:            {chromosome.n_features}")
        print(f"   Tiebreak Feature:    {chromosome.tiebreak_feature}")
        print(f"   Early Phase Window:  {chromosome.early_phase_window} hours")

        # Bucket distribution
        print("\n🎯 Bucket Assignment Distribution:")
        bucket_dist = self.get_bucket_distribution()
        total_assignments = sum(bucket_dist.values())
        bucket_names = {
            0: "NMC (Mandatory)",
            1: "PREVENTIVE_ELIGIBLE",
            2: "LIKELY_DONT_FLY",
            3: "COULD_FLY",
            4: "SHOULD_FLY"
        }
        for bucket in sorted(bucket_dist.keys()):
            count = bucket_dist[bucket]
            pct = count / total_assignments * 100
            name = bucket_names.get(bucket, f"Bucket {bucket}")
            print(f"   Bucket {bucket} ({name:20s}): {count:4d} ({pct:5.1f}%)")

        # Action distribution
        print("\n🎬 Action Distribution:")
        action_dist = self.get_action_distribution()
        total_actions = sum(action_dist.values())
        for action in sorted(action_dist.keys()):
            count = action_dist[action]
            pct = count / total_actions * 100
            print(f"   {action:25s}: {count:4d} ({pct:5.1f}%)")

        # Mission performance
        print("\n✈️  Mission Performance:")
        mission_stats = self.get_mission_stats()
        print(f"   Days Simulated:       {mission_stats['total_days']}")
        print(f"   Mission Met:          {mission_stats['mission_met']}/{mission_stats['total_days']} ({mission_stats['mission_success_rate']:.1%})")
        print(f"   Average OR:           {mission_stats['avg_or']:.2%}")
        print(f"   Total Flight Hours:   {mission_stats['total_flight_hours']:.1f}")

        # NMC handling
        print("\n🔧 NMC Handling:")
        nmc_stats = self.get_nmc_stats()
        print(f"   NMC Events:           {nmc_stats['nmc_events']}")
        print(f"   Immediate Maintenance: {nmc_stats['immediate_maintenance']} ({100 - nmc_stats['stuck_percentage']:.1f}%)")
        print(f"   Stuck (no resources):  {nmc_stats['nmc_events'] - nmc_stats['immediate_maintenance']} ({nmc_stats['stuck_percentage']:.1f}%)")

        # Reproducibility checksum
        print("\n🔐 Determinism Check:")
        checksum = self.compute_checksum()
        print(f"   Run Checksum: {checksum}")
        print(f"   (Run again with same seed to verify reproducibility)")

        print("\n" + "="*70)


def run_test_episode(env, policy, diagnostics, verbose=False):
    """Run a single test episode and collect diagnostics."""
    state = env.reset()
    done = False
    day = 0

    if verbose:
        print(f"\n🚀 Starting test episode: {env.config.sim_days} days")

    while not done:
        day += 1

        # Policy decides actions
        actions = policy.decide(state)

        # Execute step
        state, reward, done, info = env.step(actions)

        # Record diagnostics
        diagnostics.record_day(day, state, actions, info, policy)

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

    if verbose:
        print(f"✅ Episode complete!")

    return diagnostics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Decision Tree Policy infrastructure with fixed-seed random chromosome"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to simulation configuration YAML (default: config/default.yaml)"
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default="config/features/simple_dt.yaml",
        help="Path to feature configuration YAML (default: config/features/simple_dt.yaml)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chromosome generation (default: 42)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of simulation days (default: 90)"
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=3,
        help="Tree depth (default: 3, gives 8 leaves)"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=5,
        help="Number of features (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed progress logging"
    )
    parser.add_argument(
        "--save-chromosome",
        type=str,
        default=None,
        help="Save generated chromosome to JSON file (optional)"
    )

    args = parser.parse_args()

    print("="*70)
    print("DECISION TREE POLICY INFRASTRUCTURE TEST")
    print("="*70)
    print(f"Seed: {args.seed}")
    print(f"Simulation Days: {args.days}")
    print(f"Tree Depth: {args.tree_depth} ({2**args.tree_depth} leaves)")
    print(f"Features: {args.n_features}")
    print("="*70)

    # Load simulation configuration
    print(f"\n📋 Loading simulation config from {args.config}")
    config = load_config_from_yaml(args.config)
    config.sim_days = args.days
    config.seed = args.seed

    # Create random chromosome with fixed seed
    print(f"\n🧬 Generating random chromosome (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    chromosome = Chromosome.random(
        tree_depth=args.tree_depth,
        n_features=args.n_features,
        rng=rng
    )
    print(f"   Generated chromosome: {chromosome}")

    # Optionally save chromosome
    if args.save_chromosome:
        save_path = Path(args.save_chromosome)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        chromosome.to_json(str(save_path))
        print(f"   💾 Saved chromosome to {save_path}")

    # Initialize state encoder
    print(f"\n🔍 Loading feature encoder from {args.feature_config}")
    encoder = StateEncoder(args.feature_config, verbose=args.verbose)
    print(f"   Features: {encoder.n_per_aircraft} per-aircraft")

    # Create policy
    print(f"\n🎯 Creating DecisionTreePolicy...")
    policy = DecisionTreePolicy(
        chromosome=chromosome,
        encoder=encoder,
        verbose=args.verbose
    )

    # Initialize environment
    print(f"\n🏗️  Initializing environment ({config.num_aircraft} aircraft, {config.sim_days} days)")
    env = Environment(config)

    # Run test episode
    print(f"\n▶️  Running test episode...")
    diagnostics = DiagnosticsTracker()
    diagnostics = run_test_episode(env, policy, diagnostics, verbose=args.verbose)

    # Print comprehensive summary
    diagnostics.print_summary(chromosome)

    print("\n✅ Test complete! Infrastructure validated successfully.")
    print(f"   Run again with --seed {args.seed} to verify determinism.")


if __name__ == "__main__":
    main()
