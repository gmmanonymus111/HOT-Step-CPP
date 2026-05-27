#!/usr/bin/env python3
"""
uber_benchmark.py — HOT-Step combinatorial solver/scheduler/guidance benchmark
================================================================================

Runs steps_benchmark.py for every combination of solver × scheduler × guidance,
each in its own timestamped results folder.

Usage:
  python uber_benchmark.py <params.json> [--dry-run]

Requires steps_benchmark.py in the same directory.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── Matrix definition ──

SOLVERS = [
    "euler",
    "dpm2m",
    "dpm3m",
    "heun",
    "stork2",
    "rk4",
    "storm",
    "jkass_fast",
]

SCHEDULERS = [
    "linear",
    "cosine",
    "md_hap",
    "sgm_uniform",
    "bong_tangent",
]

GUIDANCE_MODES = [
    "apg",
    "md_storm_guidance",
    "dynamic_cfg",
]

STEP_COUNTS = "8,25,50,100,150,200"

# ── NFE lookup for time estimation ──
NFE_MAP = {
    "euler": 1, "dpm2m": 1, "dpm3m": 1, "storm": 1, "jkass_fast": 1,
    "heun": 2, "stork2": 2, "stork4": 4, "rk4": 4,
    "dpm2m_ada": 1, "rfsolver": 1, "unipc": 1, "unipc_p": 1,
    "aflops": 1, "aflops2": 1, "gl2s": 2, "rk5": 5,
    "dop853": 8, "dopri5": 5, "md_pingpong_simple": 2,
    "jkass_quality": 1, "sde": 1,
}


def estimate_time(solvers, steps_list):
    """Estimate total generation time in seconds."""
    base = 5  # ~5s overhead with keep_loaded
    per_step_1nfe = 0.59
    pp_overhead = 2  # PP-VAE + tiled decoder

    total = 0
    n_combos = len(SCHEDULERS) * len(GUIDANCE_MODES)
    for solver in solvers:
        nfe = NFE_MAP.get(solver, 1)
        for steps in steps_list:
            run_time = base + (per_step_1nfe * nfe * steps) + pp_overhead
            total += run_time * n_combos
    return total


def main():
    parser = argparse.ArgumentParser(
        description="HOT-Step uber benchmark — combinatorial solver/scheduler/guidance sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("params_json", help="Path to base params JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without running anything")
    parser.add_argument("--seeds", default="42",
                        help="Seeds to pass to each benchmark run (default: 42)")
    parser.add_argument("--steps", default=STEP_COUNTS,
                        help=f"Step counts (default: {STEP_COUNTS})")
    parser.add_argument("--engine", default="http://127.0.0.1:8085",
                        help="ace-server URL (default: http://127.0.0.1:8085)")
    parser.add_argument("--format", default="wav16",
                        help="Audio format (default: wav16)")
    parser.add_argument("--base-dir", default="./uber_benchmark",
                        help="Base output directory (default: ./uber_benchmark)")
    parser.add_argument("--solvers", default=None,
                        help=f"Override solvers (comma-separated, default: {','.join(SOLVERS)})")
    parser.add_argument("--schedulers", default=None,
                        help=f"Override schedulers (comma-separated, default: {','.join(SCHEDULERS)})")
    parser.add_argument("--guidance", default=None,
                        help=f"Override guidance modes (comma-separated, default: {','.join(GUIDANCE_MODES)})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combos that already have a results folder")

    args = parser.parse_args()

    # Load and validate params
    params_path = Path(args.params_json)
    if not params_path.exists():
        print(f"ERROR: Params file not found: {params_path}")
        sys.exit(1)

    with open(params_path, "r", encoding="utf-8") as f:
        base_params = json.load(f)

    # Resolve matrix
    solvers = [s.strip() for s in args.solvers.split(",")] if args.solvers else SOLVERS
    schedulers = [s.strip() for s in args.schedulers.split(",")] if args.schedulers else SCHEDULERS
    guidance_modes = [g.strip() for g in args.guidance.split(",")] if args.guidance else GUIDANCE_MODES
    steps_list = [int(s.strip()) for s in args.steps.split(",")]

    # Create base dir with timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    base_dir = Path(args.base_dir) / f"run_{timestamp}"

    total_combos = len(solvers) * len(schedulers) * len(guidance_modes)
    total_runs = total_combos * len(steps_list)
    est_seconds = estimate_time(solvers, steps_list)
    est_hours = est_seconds / 3600

    # ── Print plan ──
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           HOT-Step UBER BENCHMARK                            ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Solvers:    {', '.join(solvers):<49}║")
    print(f"║  Schedulers: {', '.join(schedulers):<49}║")
    print(f"║  Guidance:   {', '.join(guidance_modes):<49}║")
    print(f"║  Steps:      {args.steps:<49}║")
    print(f"║  Seeds:      {args.seeds:<49}║")
    print(f"╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Combos:     {total_combos:<8} ({len(solvers)}×{len(schedulers)}×{len(guidance_modes)}){' ':<29}║")
    print(f"║  Total runs: {total_runs:<8} ({total_combos} combos × {len(steps_list)} steps){' ':<18}║")
    print(f"║  Est. time:  {est_hours:.1f}h ({est_seconds/60:.0f} min){' ':<33}║")
    print(f"║  Output:     {str(base_dir):<49}║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    print()

    # Print per-solver time breakdown
    print("  Per-solver estimates:")
    for solver in solvers:
        nfe = NFE_MAP.get(solver, 1)
        solver_time = 0
        n_combos_per = len(schedulers) * len(guidance_modes)
        for steps in steps_list:
            solver_time += (5 + 0.59 * nfe * steps + 2) * n_combos_per
        print(f"    {solver:<20s} ({nfe} NFE) × {n_combos_per} combos × {len(steps_list)} steps = {solver_time/60:.0f} min")
    print()

    if args.dry_run:
        print("  DRY RUN — listing all combos:")
        print()
        combo_num = 0
        for solver in solvers:
            for scheduler in schedulers:
                for guidance in guidance_modes:
                    combo_num += 1
                    tag = f"{solver}__{scheduler}__{guidance}"
                    print(f"    [{combo_num:>3}/{total_combos}] {tag}")
        print()
        print(f"  Total: {total_combos} combos, {total_runs} generation runs")
        print("  Remove --dry-run to execute.")
        sys.exit(0)

    # ── Execute ──
    script_dir = Path(__file__).parent
    benchmark_script = script_dir / "steps_benchmark.py"
    if not benchmark_script.exists():
        print(f"ERROR: steps_benchmark.py not found at {benchmark_script}")
        sys.exit(1)

    base_dir.mkdir(parents=True, exist_ok=True)

    # Save uber config
    uber_config = {
        "params_json": str(params_path.resolve()),
        "solvers": solvers,
        "schedulers": schedulers,
        "guidance_modes": guidance_modes,
        "steps": steps_list,
        "seeds": args.seeds,
        "engine": args.engine,
        "format": args.format,
        "timestamp": timestamp,
        "estimated_hours": round(est_hours, 1),
        "total_combos": total_combos,
        "total_runs": total_runs,
        "generation_params": base_params,
    }
    with open(base_dir / "uber_config.json", "w", encoding="utf-8") as f:
        json.dump(uber_config, f, indent=2)

    combo_num = 0
    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for solver in solvers:
        for scheduler in schedulers:
            for guidance in guidance_modes:
                combo_num += 1
                tag = f"{solver}__{scheduler}__{guidance}"
                combo_dir = base_dir / tag

                # Resume support
                if args.resume and combo_dir.exists() and (combo_dir / "benchmark_results.csv").exists():
                    skipped += 1
                    print(f"  [{combo_num}/{total_combos}] SKIP (exists): {tag}")
                    continue

                print(f"\n{'='*70}")
                print(f"  [{combo_num}/{total_combos}] {tag}")
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_per_combo = elapsed / completed
                    remaining = (total_combos - combo_num) * avg_per_combo
                    print(f"  Elapsed: {elapsed/60:.0f}m | ETA: {remaining/60:.0f}m ({remaining/3600:.1f}h)")
                print(f"{'='*70}")

                # Build per-combo params JSON
                combo_params = deepcopy(base_params)
                combo_params["inferMethod"] = solver
                combo_params["scheduler"] = scheduler
                combo_params["guidanceMode"] = guidance

                # Write temp params file
                combo_params_path = base_dir / f"_temp_{tag}.json"
                with open(combo_params_path, "w", encoding="utf-8") as f:
                    json.dump(combo_params, f, indent=2)

                # Run steps_benchmark.py
                cmd = [
                    sys.executable,
                    str(benchmark_script),
                    str(combo_params_path),
                    "--steps", args.steps,
                    "--seeds", args.seeds,
                    "--output", str(combo_dir),
                    "--engine", args.engine,
                    "--format", args.format,
                    "--keep-loaded",
                ]

                # Run steps_benchmark.py with UTF-8 forced
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"

                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(script_dir),
                        env=env,
                        timeout=7200,  # 2h max per combo
                    )
                    if result.returncode == 0:
                        completed += 1
                    else:
                        failed += 1
                        print(f"  WARNING: {tag} exited with code {result.returncode}")
                except subprocess.TimeoutExpired:
                    failed += 1
                    print(f"  ERROR: {tag} timed out after 2 hours")
                except Exception as e:
                    failed += 1
                    print(f"  ERROR: {tag} failed: {e}")
                finally:
                    # Clean up temp params file
                    try:
                        combo_params_path.unlink()
                    except Exception:
                        pass

    # ── Summary ──
    total_elapsed = time.time() - start_time
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           UBER BENCHMARK COMPLETE                            ║")
    print(f"╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Completed:  {completed:<49}║")
    print(f"║  Failed:     {failed:<49}║")
    print(f"║  Skipped:    {skipped:<49}║")
    print(f"║  Total time: {total_elapsed/3600:.1f}h ({total_elapsed/60:.0f} min){' ':<33}║")
    print(f"║  Results in: {str(base_dir):<49}║")
    print(f"╚════════════════════════════════════════════════════════════════╝")

    # Generate aggregate summary CSV
    try:
        import csv
        summary_path = base_dir / "uber_summary.csv"
        fieldnames = [
            "solver", "scheduler", "guidance", "steps", "seed",
            "sibilance", "flatness", "peakiness", "crest", "noise_floor",
            "centroid", "thd", "bandwidth", "rms", "spectral_delta",
            "quality_score",
        ]

        with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for solver in solvers:
                for scheduler in schedulers:
                    for guidance in guidance_modes:
                        tag = f"{solver}__{scheduler}__{guidance}"
                        csv_path = base_dir / tag / "benchmark_results.csv"
                        if not csv_path.exists():
                            continue
                        with open(csv_path, "r", encoding="utf-8") as rf:
                            reader = csv.DictReader(rf)
                            for row in reader:
                                row["solver"] = solver
                                row["scheduler"] = scheduler
                                row["guidance"] = guidance
                                writer.writerow({k: row.get(k, "") for k in fieldnames})

        print(f"\n  Aggregate CSV: {summary_path}")
    except Exception as e:
        print(f"\n  WARNING: Failed to generate aggregate CSV: {e}")

    print("\n  Done! 🎵🔥")


if __name__ == "__main__":
    main()
