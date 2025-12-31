#!/usr/bin/env python3
"""
Batch Experiment Runner for Cloud Deployment
=============================================
Runs multiple GA experiments in parallel on a high-CPU instance.

Usage:
    python scripts/run_batch_experiments.py --experiments experiments/ga_configs/exp_*.yaml
    python scripts/run_batch_experiments.py --experiment-list experiments/batch_ms70.txt
    python scripts/run_batch_experiments.py --experiments experiments/ga_configs/exp_simple_*_ms70.yaml --workers-per-exp 12

The script will:
1. Detect available CPUs
2. Calculate how many experiments can run in parallel
3. Run experiments in batches, respecting CPU limits
4. Log progress and handle failures gracefully
"""

import argparse
import glob
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
import re


def get_available_cpus():
    """Get the number of available CPUs."""
    return multiprocessing.cpu_count()


# Thread-safe print lock
_print_lock = threading.Lock()


def safe_print(msg: str):
    """Thread-safe printing."""
    with _print_lock:
        print(msg, flush=True)


def run_single_experiment(config_path: str, workers: int) -> dict:
    """
    Run a single GA experiment with streaming output.

    Returns dict with status, runtime, and any error messages.
    """
    start_time = time.time()
    experiment_name = Path(config_path).stem

    # Short prefix for this experiment (first 12 chars)
    prefix = experiment_name[:20].ljust(20)

    safe_print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {experiment_name} (workers={workers})")

    try:
        # Run the GA with specified workers - stream output
        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/run_island_ga.py",
                "--config", config_path,
                "--no-confirm",
                "--parallel-workers", str(workers)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            cwd=Path(__file__).parent.parent
        )

        # Pattern to match generation progress lines
        gen_pattern = re.compile(r'Gen\s+(\d+).*Best=([0-9.]+)')
        migration_pattern = re.compile(r'Migration at generation (\d+)')
        early_stop_pattern = re.compile(r'Early stopping')

        last_output = []

        # Stream output, filtering for interesting lines
        for line in process.stdout:
            line = line.rstrip()
            last_output.append(line)
            if len(last_output) > 100:
                last_output.pop(0)

            # Print generation updates with experiment prefix
            if gen_pattern.search(line):
                safe_print(f"  [{prefix}] {line}")
            elif migration_pattern.search(line):
                safe_print(f"  [{prefix}] {line}")
            elif early_stop_pattern.search(line):
                safe_print(f"  [{prefix}] {line}")
            elif 'Island GA Complete' in line:
                safe_print(f"  [{prefix}] {line}")
            elif 'Global Best Fitness' in line:
                safe_print(f"  [{prefix}] {line}")

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            safe_print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {experiment_name} ({elapsed/60:.1f} min)")
            return {
                "experiment": experiment_name,
                "config": config_path,
                "status": "success",
                "runtime_seconds": elapsed,
                "stdout": "\n".join(last_output)
            }
        else:
            safe_print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED: {experiment_name}")
            return {
                "experiment": experiment_name,
                "config": config_path,
                "status": "failed",
                "runtime_seconds": elapsed,
                "error": "\n".join(last_output[-20:])
            }

    except Exception as e:
        elapsed = time.time() - start_time
        safe_print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {experiment_name} - {str(e)}")
        return {
            "experiment": experiment_name,
            "config": config_path,
            "status": "error",
            "runtime_seconds": elapsed,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple GA experiments in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ms70 experiments with auto-detected parallelism
  python scripts/run_batch_experiments.py --experiments "experiments/ga_configs/exp_simple_*_ms70.yaml"

  # Run from a list file
  python scripts/run_batch_experiments.py --experiment-list experiments/batch_list.txt

  # Specify workers per experiment (default: 12)
  python scripts/run_batch_experiments.py --experiments "experiments/ga_configs/*.yaml" --workers-per-exp 16

  # Limit concurrent experiments
  python scripts/run_batch_experiments.py --experiments "experiments/ga_configs/*.yaml" --max-concurrent 8
        """
    )

    parser.add_argument(
        "--experiments",
        type=str,
        help="Glob pattern for experiment configs (e.g., 'experiments/ga_configs/exp_*.yaml')"
    )
    parser.add_argument(
        "--experiment-list",
        type=str,
        help="Path to file containing experiment config paths (one per line)"
    )
    parser.add_argument(
        "--workers-per-exp",
        type=int,
        default=12,
        help="Number of parallel workers per experiment (default: 12)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent experiments (default: auto-calculate from CPUs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt (for automated runs)"
    )

    args = parser.parse_args()

    # Gather experiment configs
    configs = []
    if args.experiments:
        configs.extend(sorted(glob.glob(args.experiments)))
    if args.experiment_list:
        with open(args.experiment_list) as f:
            configs.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])

    if not configs:
        print("Error: No experiment configs specified or found")
        print("Use --experiments 'pattern' or --experiment-list file.txt")
        sys.exit(1)

    # Remove duplicates while preserving order
    configs = list(dict.fromkeys(configs))

    # Verify all configs exist
    missing = [c for c in configs if not os.path.exists(c)]
    if missing:
        print(f"Error: {len(missing)} config files not found:")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        sys.exit(1)

    # Calculate parallelism
    total_cpus = get_available_cpus()
    workers_per_exp = args.workers_per_exp

    if args.max_concurrent:
        max_concurrent = args.max_concurrent
    else:
        # Leave 2-4 CPUs for OS overhead
        available_cpus = max(total_cpus - 4, total_cpus * 0.95)
        max_concurrent = max(1, int(available_cpus // workers_per_exp))

    # Print summary
    print("=" * 70)
    print("BATCH EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Total CPUs available:     {total_cpus}")
    print(f"Workers per experiment:   {workers_per_exp}")
    print(f"Max concurrent experiments: {max_concurrent}")
    print(f"Total experiments:        {len(configs)}")
    print(f"Estimated CPU utilization: {min(len(configs), max_concurrent) * workers_per_exp}/{total_cpus}")
    print("=" * 70)
    print("\nExperiments to run:")
    for i, config in enumerate(configs, 1):
        print(f"  {i:2d}. {Path(config).stem}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would execute above experiments. Exiting.")
        sys.exit(0)

    # Confirm (unless --no-confirm)
    if not args.no_confirm:
        response = input(f"Run {len(configs)} experiments with {max_concurrent} concurrent? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Run experiments with dynamic worker allocation
    start_time = time.time()
    results = []
    pending_configs = list(configs)  # Configs waiting to run
    running_futures = {}  # future -> (config, workers)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting batch run (dynamic worker allocation)...")
    print("-" * 70)

    # Use ThreadPoolExecutor so output goes to stdout properly
    def calculate_workers(remaining_count: int, total_cpus: int) -> tuple:
        """Calculate concurrent experiments and workers per experiment.

        Always try to use all available CPUs.
        Returns (concurrent_count, workers_per_exp)
        """
        usable_cpus = total_cpus - 2  # Leave 2 for OS

        if remaining_count >= 3:
            # Run 3 concurrent with ~10 workers each
            concurrent = 3
            workers = usable_cpus // concurrent
        elif remaining_count == 2:
            # Run 2 concurrent with ~15 workers each
            concurrent = 2
            workers = usable_cpus // concurrent
        elif remaining_count == 1:
            # Run 1 with all workers
            concurrent = 1
            workers = usable_cpus
        else:
            concurrent = 0
            workers = 0

        return concurrent, workers

    with ThreadPoolExecutor(max_workers=10) as executor:  # Max 10 concurrent experiments
        while pending_configs or running_futures:
            # Calculate how many experiments to run and with how many workers
            remaining = len(pending_configs) + len(running_futures)
            target_concurrent, workers = calculate_workers(remaining, total_cpus)

            # Launch new experiments if we have capacity
            while pending_configs and len(running_futures) < target_concurrent:
                config = pending_configs.pop(0)
                # Recalculate workers based on what will be running
                running_after = len(running_futures) + 1
                remaining_after = len(pending_configs) + running_after
                _, workers = calculate_workers(remaining_after, total_cpus)

                future = executor.submit(run_single_experiment, config, workers)
                running_futures[future] = (config, workers)

            # Wait for at least one to complete
            if running_futures:
                done, _ = wait(running_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    config, workers_used = running_futures.pop(future)
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "experiment": Path(config).stem,
                            "config": config,
                            "status": "error",
                            "error": str(e)
                        })

    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print("-" * 70)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] BATCH COMPLETE")
    print("=" * 70)
    print(f"Total time:      {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful:      {len(successful)}/{len(results)}")
    print(f"Failed:          {len(failed)}/{len(results)}")

    if successful:
        avg_time = sum(r.get("runtime_seconds", 0) for r in successful) / len(successful)
        print(f"Avg time/exp:    {avg_time/60:.1f} minutes")

    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['experiment']}: {r.get('error', 'Unknown error')[:100]}")

    # Save results log
    log_path = f"results/batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("results", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Batch Run Summary\n")
        f.write(f"=" * 70 + "\n")
        f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"CPUs: {total_cpus}, Workers/exp: {workers_per_exp}, Concurrent: {max_concurrent}\n")
        f.write(f"\nResults:\n")
        for r in results:
            status = r["status"]
            runtime = r.get("runtime_seconds", 0) / 60
            f.write(f"  {r['experiment']}: {status} ({runtime:.1f} min)\n")
            if status != "success" and "error" in r:
                f.write(f"    Error: {r['error'][:200]}\n")

    print(f"\nLog saved to: {log_path}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
