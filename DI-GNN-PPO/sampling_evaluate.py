"""
sampling_evaluate.py
====================

Sampling-based inference for DI-GNN-PPO.

Instead of giving the trained policy only one deterministic greedy rollout,
this evaluator samples multiple trajectories from the learned policy and
keeps the best schedule found.

For each test instance:

    Trained DI-GNN-PPO
            |
            +---- Sample 1 ----> makespan
            +---- Sample 2 ----> makespan
            +---- ...
            +---- Sample N ----> makespan
            |
            +---- Best makespan

This is inference-time exploration only.
The model is NOT retrained.

Examples
--------
    python sampling_evaluate.py --run-name di_gnn_j30 --sizes j30
    python sampling_evaluate.py --run-name di_gnn_j30 --sizes j30 --samples 50
    python sampling_evaluate.py --run-name di_gnn_j60 --sizes j60 --samples 100

Output
------
    runs/<run-name>/sampling_evaluation_results.csv
    runs/<run-name>/sampling_evaluation_per_instance.csv
    runs/<run-name>/sampling_evaluation_results.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RUNS_ROOT,
    PROBLEM_SIZES,
    Config,
)
from domain_features import StaticFeatureCache
from psplib_loader import RCPSPInstance, load_dataset
from rcpsp_environment import RCPSPEnv
from utils import (
    format_summary_row,
    get_device,
    load_checkpoint,
    set_seed,
    split_dataset,
    summarize,
    write_csv,
)


BASELINE_RULES = ("Random", "SPT", "LFT", "GRPW")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(
    checkpoint_path: Path,
    device,
):
    """
    Load the exact model architecture and configuration stored in the
    checkpoint.
    """

    from di_gnn_model import DIGNNActorCritic

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    if "config" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain 'config'."
        )

    if "model_state" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain 'model_state'."
        )

    cfg = Config.from_dict(
        checkpoint["config"]
    )

    model = DIGNNActorCritic(cfg).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, cfg, checkpoint


# ============================================================================
# POLICY ROLLOUTS
# ============================================================================

def greedy_rollout(
    model,
    env: RCPSPEnv,
    instance: RCPSPInstance,
) -> int:
    """
    One deterministic greedy rollout.

    At every decision the model selects its highest-probability action.
    """

    obs = env.reset(instance)
    done = False

    while not done:
        action, _, _ = model.act(
            obs,
            greedy=True,
        )

        obs, _, done, _ = env.step(action)

    return int(env.makespan)


def sampled_rollout(
    model,
    env: RCPSPEnv,
    instance: RCPSPInstance,
) -> int:
    """
    One stochastic rollout.

    The model samples from its learned action distribution instead of always
    selecting the highest-probability action.

    This allows different trajectories to be explored.
    """

    obs = env.reset(instance)
    done = False

    while not done:

        action, _, _ = model.act(
            obs,
            greedy=False,
        )

        obs, _, done, _ = env.step(action)

    return int(env.makespan)


def sampling_rollout(
    model,
    env: RCPSPEnv,
    instance: RCPSPInstance,
    num_samples: int,
) -> Dict[str, float]:
    """
    Run multiple stochastic trajectories for one instance.

    Returns:
        greedy:
            Deterministic greedy makespan.

        best:
            Best makespan among sampled trajectories.

        mean:
            Mean makespan across sampled trajectories.

        median:
            Median sampled makespan.

        std:
            Standard deviation of sampled makespans.
    """

    greedy = greedy_rollout(
        model,
        env,
        instance,
    )

    samples = []

    for _ in range(num_samples):
        makespan = sampled_rollout(
            model,
            env,
            instance,
        )

        samples.append(makespan)

    values = np.asarray(
        samples,
        dtype=np.float64,
    )

    return {
        "greedy": float(greedy),
        "best": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


# ============================================================================
# CLASSICAL BASELINES
# ============================================================================

def _priority(
    rule: str,
    st,
    cfg: Config,
) -> np.ndarray:

    if rule == "SPT":
        return -st.durations.astype(np.float64)

    if rule == "LFT":
        return -st.lf.astype(np.float64)

    if rule == "GRPW":

        if cfg.grpw_variant == "all":
            return st.grpw_all.astype(np.float64)

        return st.grpw_immediate.astype(np.float64)

    raise ValueError(
        f"Unknown priority rule: {rule}"
    )


def rule_rollout(
    env: RCPSPEnv,
    instance: RCPSPInstance,
    rule: str,
    rng: np.random.Generator | None = None,
) -> int:

    obs = env.reset(instance)

    if rule == "Random":

        if rng is None:
            raise ValueError(
                "Random rule requires an RNG."
            )

        priority = None

    else:

        priority = _priority(
            rule,
            env.static,
            env.cfg,
        )

    done = False

    while not done:

        candidates = np.flatnonzero(
            obs["mask"]
        )

        if len(candidates) == 0:
            raise RuntimeError(
                f"No valid action available for "
                f"instance {instance.name}."
            )

        if priority is None:

            action = int(
                rng.choice(candidates)
            )

        else:

            scores = priority[candidates]

            action = int(
                candidates[
                    np.argmax(scores)
                ]
            )

        obs, _, done, _ = env.step(action)

    return int(env.makespan)


# ============================================================================
# CHECKPOINT
# ============================================================================

def resolve_checkpoint(
    args: argparse.Namespace,
    size: str,
) -> Path:

    if args.checkpoint is not None:
        return args.checkpoint

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"di_gnn_{size}"

    return (
        args.runs_root
        / run_name
        / "best.pt"
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_size(
    size: str,
    args: argparse.Namespace,
    device,
):
    """
    Evaluate one problem size using sampled inference.
    """

    checkpoint_path = resolve_checkpoint(
        args,
        size,
    )

    print("\n" + "=" * 72)
    print(size.upper())
    print("=" * 72)

    print(
        f"Loading checkpoint:\n"
        f"  {checkpoint_path}"
    )

    model, cfg, checkpoint = load_model(
        checkpoint_path,
        device,
    )

    print("Checkpoint loaded successfully.")

    print(
        f"  update:       "
        f"{checkpoint.get('update', '?')}"
    )

    print(
        f"  best val:     "
        f"{checkpoint.get('validation_makespan', '?')}"
    )

    print(
        f"  features:     "
        f"{cfg.node_feature_dim()}"
    )

    print(
        f"  GNN blocks:   "
        f"{cfg.gin_layers}"
    )

    print(
        f"  hidden dim:   "
        f"{cfg.hidden_dim}"
    )

    print(
        f"  bidirectional:"
        f"{cfg.bidirectional_edges}"
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    set_seed(cfg.seed)

    instances = load_dataset(
        data_root=args.data_root,
        size=size,
    )

    train_set, val_set, test_set = split_dataset(
        instances,
        cfg,
    )

    print(
        f"Dataset: {len(instances)} instances"
    )

    print(
        f"Split: train={len(train_set)} "
        f"val={len(val_set)} "
        f"test={len(test_set)}"
    )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    env = RCPSPEnv(
        cfg,
        StaticFeatureCache(),
    )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    print(
        f"\nSampling {args.samples} trajectories "
        f"per test instance..."
    )

    greedy_results = []
    sampled_best_results = []
    sampled_mean_results = []

    per_instance_rows = []

    for index, instance in enumerate(test_set, start=1):

        result = sampling_rollout(
            model,
            env,
            instance,
            args.samples,
        )

        greedy = int(result["greedy"])
        best = int(result["best"])
        sampled_mean = result["mean"]

        greedy_results.append(greedy)
        sampled_best_results.append(best)
        sampled_mean_results.append(sampled_mean)

        improvement = greedy - best

        per_instance_rows.append({
            "problem_size": size,
            "instance": instance.name,
            "greedy_makespan": greedy,
            "sampled_best_makespan": best,
            "sampled_mean_makespan": round(
                sampled_mean,
                4,
            ),
            "sampled_median_makespan": round(
                result["median"],
                4,
            ),
            "sampled_std": round(
                result["std"],
                4,
            ),
            "improvement_vs_greedy": improvement,
            "samples": args.samples,
        })

        if index % 10 == 0 or index == len(test_set):

            print(
                f"  {index:3d}/{len(test_set)} "
                f"instances | "
                f"current best-of-{args.samples}: "
                f"{best}"
            )

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    rng = np.random.default_rng(
        cfg.seed
        if args.seed is None
        else args.seed
    )

    baseline_results = {}

    if not args.no_baselines:

        for rule in BASELINE_RULES:

            print(
                f"Running {rule}..."
            )

            baseline_results[rule] = [
                rule_rollout(
                    env,
                    instance,
                    rule,
                    rng,
                )
                for instance in test_set
            ]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    results = {
        "DI-GNN-PPO Greedy": greedy_results,
        f"DI-GNN-PPO Sampling-{args.samples}":
            sampled_best_results,
    }

    results.update(
        baseline_results
    )

    print("\nResults")
    print("-" * 72)

    summaries = {
        method: summarize(values)
        for method, values in results.items()
    }

    for method, summary in summaries.items():

        print(
            format_summary_row(
                method,
                summary,
            )
        )

    # ------------------------------------------------------------------
    # Sampling improvement
    # ------------------------------------------------------------------

    greedy_mean = summaries[
        "DI-GNN-PPO Greedy"
    ]["mean"]

    sampled_mean = summaries[
        f"DI-GNN-PPO Sampling-{args.samples}"
    ]["mean"]

    improvement = (
        greedy_mean
        - sampled_mean
    )

    print(
        "\nSampling improvement"
    )

    print("-" * 72)

    print(
        f"Greedy mean: "
        f"{greedy_mean:.3f}"
    )

    print(
        f"Best-of-{args.samples} mean: "
        f"{sampled_mean:.3f}"
    )

    print(
        f"Improvement: "
        f"{improvement:+.3f} makespan"
    )

    # ------------------------------------------------------------------
    # Compare with baselines
    # ------------------------------------------------------------------

    print(
        f"\nBest-of-{args.samples} comparison"
    )

    print("-" * 72)

    for rule in BASELINE_RULES:

        if rule not in summaries:
            continue

        gap = (
            sampled_mean
            - summaries[rule]["mean"]
        )

        status = (
            "BETTER"
            if gap < 0
            else "WORSE"
            if gap > 0
            else "TIED"
        )

        print(
            f"vs {rule:6s}: "
            f"{gap:+.3f} makespan | "
            f"{status}"
        )

    best_method = min(
        summaries,
        key=lambda method:
        summaries[method]["mean"],
    )

    print(
        f"\nBest mean makespan: "
        f"{best_method} "
        f"({summaries[best_method]['mean']:.3f})"
    )

    # ------------------------------------------------------------------
    # Add summaries to output
    # ------------------------------------------------------------------

    summary_rows = []

    for method, summary in summaries.items():

        summary_rows.append({
            "problem_size": size,
            "method": method,
            "num_test_instances": summary["n"],
            "mean_makespan": round(
                summary["mean"],
                4,
            ),
            "median_makespan": summary["median"],
            "std_makespan": round(
                summary["std"],
                4,
            ),
            "min_makespan": summary["min"],
            "max_makespan": summary["max"],
            "samples": (
                args.samples
                if "Sampling" in method
                else ""
            ),
            "seed": cfg.seed,
            "checkpoint": (
                str(checkpoint_path)
                if "DI-GNN-PPO" in method
                else ""
            ),
            "best_val_mean_makespan": (
                checkpoint.get(
                    "validation_makespan",
                    "",
                )
                if "DI-GNN-PPO" in method
                else ""
            ),
        })

    return (
        summary_rows,
        per_instance_rows,
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Sampling-based evaluation of "
            "DI-GNN-PPO."
        )
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Training run containing best.pt.",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["j30"],
        choices=PROBLEM_SIZES,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help=(
            "Number of stochastic trajectories "
            "per test instance."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--no-baselines",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error(
            "--samples must be >= 1"
        )

    if (
        args.checkpoint is not None
        and len(args.sizes) != 1
    ):
        parser.error(
            "--checkpoint requires exactly "
            "one --sizes entry."
        )

    if (
        args.checkpoint is not None
        and args.run_name is not None
    ):
        parser.error(
            "Use either --checkpoint or "
            "--run-name, not both."
        )

    return args


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    device = get_device(
        args.device
    )

    print("=" * 72)
    print(
        "DI-GNN-PPO SAMPLING EVALUATION"
    )
    print("=" * 72)

    print(
        f"Device: {device}"
    )

    print(
        f"Sizes: {', '.join(args.sizes)}"
    )

    print(
        f"Samples per instance: "
        f"{args.samples}"
    )

    if args.run_name is not None:

        print(
            f"Run: {args.run_name}"
        )

    all_summary_rows = []
    all_per_instance_rows = []

    for size in args.sizes:

        (
            summary_rows,
            per_instance_rows,
        ) = evaluate_size(
            size,
            args,
            device,
        )

        all_summary_rows.extend(
            summary_rows
        )

        all_per_instance_rows.extend(
            per_instance_rows
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    if args.run_name is not None:

        output_run_name = args.run_name

    elif args.checkpoint is not None:

        output_run_name = (
            args.checkpoint.parent.name
        )

    else:

        output_run_name = (
            f"di_gnn_{args.sizes[0]}"
            if len(args.sizes) == 1
            else "di_gnn_sampling_evaluation"
        )

    out_dir = (
        args.runs_root
        / output_run_name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------

    write_csv(
        out_dir
        / "sampling_evaluation_results.csv",
        all_summary_rows,
    )

    write_csv(
        out_dir
        / "sampling_evaluation_per_instance.csv",
        all_per_instance_rows,
    )

    # ------------------------------------------------------------------
    # TXT report
    # ------------------------------------------------------------------

    lines = []

    lines.append(
        "DI-GNN-PPO SAMPLING EVALUATION"
    )

    lines.append(
        f"Samples per instance: {args.samples}"
    )

    lines.append("")

    for row in all_summary_rows:

        lines.append(
            format_summary_row(
                row["method"],
                {
                    "n": row["num_test_instances"],
                    "mean": row["mean_makespan"],
                    "median": row["median_makespan"],
                    "std": row["std_makespan"],
                    "min": row["min_makespan"],
                    "max": row["max_makespan"],
                },
            )
        )

    (
        out_dir
        / "sampling_evaluation_results.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "SAMPLING EVALUATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Results saved to:\n"
        f"  {out_dir}"
    )


if __name__ == "__main__":
    main()
