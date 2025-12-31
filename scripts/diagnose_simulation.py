#!/usr/bin/env python3
"""Diagnostic script to analyze mission success patterns.

This script runs a simulation and collects detailed statistics about:
- Mission requirements vs FMC counts
- Success/failure patterns by demand level
- OR and aircraft status over time
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.environment import Environment
from policy.baseline_policy import BaselinePolicy
from utils.config import load_config_from_yaml


def run_diagnostic(n_episodes: int = 10, verbose: bool = True):
    """Run diagnostic simulation and collect statistics."""

    # Load config
    config = load_config_from_yaml(str(PROJECT_ROOT / "config" / "default.yaml"))

    # Statistics collectors
    stats = {
        'mission_success_by_requirement': defaultdict(lambda: {'success': 0, 'total': 0}),
        'fmc_counts': [],
        'requirements': [],
        'planned_requirements': [],  # What policy thought it was planning for
        'flying_counts': [],
        'or_values': [],
        'in_maintenance_counts': [],
        'nmc_counts': [],
        'mismatch_count': 0,
        'total_days': 0,
    }

    for episode in range(n_episodes):
        # Create fresh environment and policy
        env = Environment(config, rng=np.random.default_rng(42 + episode))
        policy = BaselinePolicy(verbose=False)

        state = env.reset(seed=42 + episode)
        done = False

        while not done:
            # Get current state info (BEFORE step)
            fmc_count = sum(1 for a in state['aircraft'] if a.status == 'FMC')
            nmc_count = sum(1 for a in state['aircraft'] if a.status == 'NMC')
            in_maint_count = sum(1 for a in state['aircraft'] if a.in_maintenance)

            # What the policy THINKS the requirement is (from state's forecast[1] after fix)
            mission_forecast = state['mission_forecast']
            if len(mission_forecast) > 1:
                planned_req = mission_forecast[1].required_aircraft
            elif len(mission_forecast) > 0:
                planned_req = mission_forecast[0].required_aircraft
            else:
                planned_req = 0

            # Get actions
            actions = policy.decide(state)

            # Count flying actions requested by policy
            flying_count = sum(1 for a in actions.values() if a == 'fly')

            # Execute step
            state, reward, done, info = env.step(actions)

            # Get ACTUAL requirement from info (this is what step() used)
            actual_req = info['required_aircraft']
            actual_flying = info['flying_aircraft']

            # Track mismatch between planned and actual
            stats['total_days'] += 1
            if planned_req != actual_req:
                stats['mismatch_count'] += 1

            # Record stats using the CORRECT requirement
            mission_success = info['mission_success']
            stats['mission_success_by_requirement'][actual_req]['total'] += 1
            if mission_success:
                stats['mission_success_by_requirement'][actual_req]['success'] += 1

            stats['fmc_counts'].append(fmc_count)
            stats['requirements'].append(actual_req)
            stats['planned_requirements'].append(planned_req)
            stats['flying_counts'].append(actual_flying)
            stats['or_values'].append(state['current_or'])
            stats['in_maintenance_counts'].append(in_maint_count)
            stats['nmc_counts'].append(nmc_count)

        if verbose:
            final = env.get_final_metrics()
            print(f"Episode {episode+1}: MS={final['mission_success_rate']:.2%}, OR={final['mean_or']:.2%}")

    return stats


def analyze_stats(stats):
    """Analyze and print diagnostic statistics."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)

    # CRITICAL: Planned vs Actual mismatch
    print("\n--- ⚠️  PLANNED vs ACTUAL REQUIREMENT MISMATCH ---")
    mismatch_rate = stats['mismatch_count'] / stats['total_days'] * 100
    print(f"Days where planned ≠ actual: {stats['mismatch_count']} / {stats['total_days']} ({mismatch_rate:.1f}%)")

    # Analyze mismatch patterns
    planned = np.array(stats['planned_requirements'])
    actual = np.array(stats['requirements'])
    diff = actual - planned

    under_planned = (diff > 0).sum()  # Actual > planned (flew too few)
    over_planned = (diff < 0).sum()   # Actual < planned (flew too many, but OK)
    print(f"  Under-planned (actual > planned): {under_planned} ({under_planned/len(diff)*100:.1f}%) → Potential failures")
    print(f"  Over-planned (actual < planned): {over_planned} ({over_planned/len(diff)*100:.1f}%) → OK (flew extra)")
    print(f"  Mean difference (actual - planned): {diff.mean():.2f}")

    # Mission success by requirement
    print("\n--- Mission Success by Requirement ---")
    print(f"{'Req':>4} | {'Success':>8} | {'Total':>8} | {'Rate':>8} | {'Avg FMC on Fail'}")
    print("-" * 55)

    for req in sorted(stats['mission_success_by_requirement'].keys()):
        data = stats['mission_success_by_requirement'][req]
        rate = data['success'] / data['total'] if data['total'] > 0 else 0

        # Calculate avg FMC when failing at this requirement level
        fail_indices = [i for i, (r, f) in enumerate(zip(stats['requirements'],
                        [stats['fmc_counts'][i] for i in range(len(stats['requirements']))]))
                       if r == req and not (f >= req)]
        # Actually need to check success/fail differently - just show basic stats for now

        print(f"{req:>4} | {data['success']:>8} | {data['total']:>8} | {rate:>7.1%} |")

    # Overall distribution
    print("\n--- Overall Statistics ---")
    print(f"Mean FMC count: {np.mean(stats['fmc_counts']):.2f}")
    print(f"Std FMC count: {np.std(stats['fmc_counts']):.2f}")
    print(f"Min FMC count: {np.min(stats['fmc_counts'])}")
    print(f"Max FMC count: {np.max(stats['fmc_counts'])}")

    print(f"\nMean requirement: {np.mean(stats['requirements']):.2f}")
    print(f"Mean flying count: {np.mean(stats['flying_counts']):.2f}")
    print(f"Mean OR: {np.mean(stats['or_values']):.2%}")

    print(f"\nMean NMC count: {np.mean(stats['nmc_counts']):.2f}")
    print(f"Mean in-maintenance: {np.mean(stats['in_maintenance_counts']):.2f}")

    # FMC distribution
    print("\n--- FMC Count Distribution ---")
    fmc_counts = np.array(stats['fmc_counts'])
    for fmc in range(9):
        pct = np.mean(fmc_counts == fmc) * 100
        print(f"FMC={fmc}: {pct:5.1f}%")

    # Cross-tabulation: requirement vs FMC
    print("\n--- Failure Analysis ---")
    print("Days where requirement > FMC (mission failures):")
    reqs = np.array(stats['requirements'])
    fmcs = np.array(stats['fmc_counts'])

    failures = reqs > fmcs
    print(f"Total failure days: {failures.sum()} / {len(failures)} ({failures.mean()*100:.1f}%)")

    # Breakdown by requirement
    print("\nFailures by requirement level:")
    for req in range(8):
        mask = reqs == req
        if mask.sum() > 0:
            fail_rate = (reqs[mask] > fmcs[mask]).mean() * 100
            avg_fmc_when_req = fmcs[mask].mean()
            print(f"  Req={req}: {fail_rate:5.1f}% fail, avg FMC={avg_fmc_when_req:.1f}")


if __name__ == "__main__":
    print("Running simulation diagnostic...")
    stats = run_diagnostic(n_episodes=10, verbose=True)
    analyze_stats(stats)
