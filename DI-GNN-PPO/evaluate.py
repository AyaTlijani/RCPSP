"""
evaluate.py
===========

Evaluate the best DI-GNN-PPO checkpoint on the held-out TEST split and compare
it with classical priority rules on the exact same instances and environment.

Methods:
    DI-GNN-PPO (greedy deterministic policy)
    Random
    SPT
    LFT
    GRPW

Outputs:
    runs/<run-name>/evaluation_results.csv
    runs/<run-name>/evaluation_per_instance.csv
    runs/<run-name>/evaluation_results.txt

Examples
--------
    python evaluate.py --run-name di_gnn_j30 --sizes j30
    python evaluate.py --run-name di_gnn_j60 --sizes j60
    python evaluate.py --run-name di_gnn_j90 --sizes j90

Or explicitly provide a checkpoint:

    python evaluate.py --checkpoint runs/di_gnn_j30/best.pt --sizes j30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RUNS_ROOT,
    PROBLEM_SIZES,
    Config,
)
from domain_features import StaticFeatureCache, StaticFeatures
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
# LEARNED POLICY
# ============================================================================

def greedy_rollout(
    model,
    env: RCPSPEnv,
    instance: RCPSPInstance,
) -> int:
    """
    Run one deterministic greedy episode.

    The model always selects the highest-probability valid action.
    Returns the final makespan.
    """
    obs = env.reset(instance)
    done = False

    while not done:
        action, _, _ = model.act(obs, greedy=True)
        obs, _, done, _ = env.step(action)

    return int(env.makespan)


def evaluate_policy(
    model,
    env: RCPSPEnv,
    instances: Sequence[RCPSPInstance],
) -> List[int]:
    """
    Evaluate the learned policy on every instance.

    The same environment object is reused, but reset(instance) completely
    initializes each episode.
    """
    was_training = model.training
    model.eval()

    try:
        return [
            greedy_rollout(model, env, instance)
            for instance in instances
        ]
    finally:
        model.train(was_training)


# ============================================================================
# CLASSICAL PRIORITY RULES
# ============================================================================

def _priority(
    rule: str,
    st: StaticFeatures,
    cfg: Config,
) -> np.ndarray:
    """
    Return priority scores.

    Higher score means the activity is preferred among currently
    startable activities.
    """

    if rule == "SPT":
        # Shortest processing time first.
        return -st.durations.astype(np.float64)

    if rule == "LFT":
        # Lowest latest-finish-time first.
        return -st.lf.astype(np.float64)

    if rule == "GRPW":
        if cfg.grpw_variant == "all":
            return st.grpw_all.astype(np.float64)

        return st.grpw_immediate.astype(np.float64)

    raise ValueError(f"Unknown priority rule: {rule}")


def rule_rollout(
    env: RCPSPEnv,
    instance: RCPSPInstance,
    rule: str,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Run one priority-rule episode.

    Random is stochastic.
    SPT/LFT/GRPW are deterministic.
    """

    obs = env.reset(instance)

    if rule == "Random":
        if rng is None:
            raise ValueError("Random rule requires an RNG")

        priority = None

    else:
        priority = _priority(
            rule,
            env.static,
            env.cfg,
        )

    done = False

    while not done:
        candidates = np.flatnonzero(obs["mask"])

        if len(candidates) == 0:
            raise RuntimeError(
                f"No valid action available for instance {instance.name} "
                f"while episode is not finished."
            )

        if priority is None:
            action = int(rng.choice(candidates))
        else:
            candidate_scores = priority[candidates]
            action = int(
                candidates[np.argmax(candidate_scores)]
            )

        obs, _, done, _ = env.step(action)

    return int(env.makespan)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(
    checkpoint_path: Path,
    device,
):
    """
    Load DI-GNN-PPO checkpoint and reconstruct its exact training config.

    The configuration stored inside the checkpoint is authoritative.
    """

    from di_gnn_model import DIGNNActorCritic

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    if "config" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain a 'config' entry."
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
# REPORTING
# ============================================================================

def print_comparison(
    results: Dict[str, List[int]],
) -> None:
    """
    Print summary statistics and comparison against DI-GNN-PPO.
    """

    summaries = {
        method: summarize(values)
        for method, values in results.items()
    }

    print("\nResults")
    print("-" * 72)

    for method, summary in summaries.items():
        print(
            format_summary_row(
                method,
                summary,
            )
        )

    if "DI-GNN-PPO" not in summaries:
        print(
            "\nWARNING: DI-GNN-PPO was not evaluated."
        )
        return

    di_mean = summaries["DI-GNN-PPO"]["mean"]

    print("\nDI-GNN-PPO comparison")
    print("-" * 72)

    for method in BASELINE_RULES:
        if method not in summaries:
            continue

        gap = (
            di_mean
            - summaries[method]["mean"]
        )

        if gap < 0:
            status = "BETTER"
        elif gap > 0:
            status = "WORSE"
        else:
            status = "TIED"

        print(
            f"vs {method:6s}: "
            f"{gap:+.3f} makespan | {status}"
        )

    best = min(
        summaries,
        key=lambda method: summaries[method]["mean"],
    )

    print(
        f"\nBest mean makespan: "
        f"{best} ({summaries[best]['mean']:.3f})"
    )

    if best == "DI-GNN-PPO":
        print(
            "Conclusion: DI-GNN-PPO has the best mean test makespan."
        )
    else:
        print(
            f"Conclusion: {best} still beats "
            f"DI-GNN-PPO on mean test makespan."
        )


# ============================================================================
# CHECKPOINT RESOLUTION
# ============================================================================

def resolve_checkpoint(
    args: argparse.Namespace,
    size: str,
) -> Path:
    """
    Resolve the checkpoint for a particular problem size.

    Priority:
        1. Explicit --checkpoint
        2. runs/<run-name>/best.pt

    If no run name is supplied, use the same naming convention as train.py:
        di_gnn_j30
        di_gnn_j60
        di_gnn_j90
        ...
    """

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
) -> tuple[
    List[Dict],
    List[Dict],
    List[str],
]:
    """
    Evaluate one problem size.
    """

    checkpoint_path = resolve_checkpoint(
        args,
        size,
    )

    print("\n" + "=" * 72)
    print(size.upper())
    print("=" * 72)

    # ----------------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------------

    model = None
    checkpoint = {}

    if checkpoint_path.is_file():

        print(
            f"Loading checkpoint:\n"
            f"  {checkpoint_path}"
        )

        model, cfg, checkpoint = load_model(
            checkpoint_path,
            device,
        )

        print(
            f"Checkpoint loaded successfully."
        )

        print(
            f"  update:       {checkpoint.get('update', '?')}"
        )

        print(
            f"  best val:     "
            f"{checkpoint.get('validation_makespan', '?')}"
        )

        print(
            f"  features:     {cfg.node_feature_dim()}"
        )

        print(
            f"  GNN blocks:   {cfg.gin_layers}"
        )

        print(
            f"  hidden dim:   {cfg.hidden_dim}"
        )

        print(
            f"  bidirectional:{cfg.bidirectional_edges}"
        )

        print(
            f"  seed:         {cfg.seed}"
        )

    else:
        if args.no_model:
            cfg = Config()
            print(
                "DI-GNN-PPO evaluation disabled by --no-model."
            )
        else:
            raise FileNotFoundError(
                "\n"
                "DI-GNN-PPO checkpoint not found.\n\n"
                f"Expected:\n"
                f"  {checkpoint_path}\n\n"
                "Train the model first or provide the correct checkpoint with:\n"
                "  --checkpoint <path>\n"
            )

    # ----------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------

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

    if len(test_set) == 0:
        raise RuntimeError(
            f"Test split for {size} is empty."
        )

    # ----------------------------------------------------------------------
    # Environment
    # ----------------------------------------------------------------------

    env = RCPSPEnv(
        cfg,
        StaticFeatureCache(),
    )

    results: Dict[str, List[int]] = {}

    # ----------------------------------------------------------------------
    # DI-GNN-PPO
    # ----------------------------------------------------------------------

    if model is not None:

        print(
            "\nRunning DI-GNN-PPO..."
        )

        results["DI-GNN-PPO"] = evaluate_policy(
            model,
            env,
            test_set,
        )

    # ----------------------------------------------------------------------
    # Classical baselines
    # ----------------------------------------------------------------------

    if not args.no_baselines:

        rng = np.random.default_rng(
            cfg.seed
            if args.seed is None
            else args.seed
        )

        for rule in BASELINE_RULES:

            print(
                f"Running {rule}..."
            )

            results[rule] = [
                rule_rollout(
                    env,
                    instance,
                    rule,
                    rng,
                )
                for instance in test_set
            ]

    if not results:
        raise RuntimeError(
            "Nothing was evaluated."
        )

    # ----------------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------------

    print_comparison(
        results
    )

    # ----------------------------------------------------------------------
    # Prepare CSV/TXT output
    # ----------------------------------------------------------------------

    summary_rows: List[Dict] = []
    per_instance_rows: List[Dict] = []
    text_lines: List[str] = []

    text_lines.append(
        f"{size.upper()} | "
        f"test={len(test_set)} | "
        f"checkpoint={checkpoint_path}"
    )

    for method, makespans in results.items():

        summary = summarize(
            makespans
        )

        text_lines.append(
            format_summary_row(
                method,
                summary,
            )
        )

        is_model = (
            method == "DI-GNN-PPO"
        )

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
            "seed": cfg.seed,
            "checkpoint": (
                str(checkpoint_path)
                if is_model
                else ""
            ),
            "best_val_mean_makespan": (
                checkpoint.get(
                    "validation_makespan",
                    "",
                )
                if is_model
                else ""
            ),
        })

        for instance, makespan in zip(
            test_set,
            makespans,
        ):
            per_instance_rows.append({
                "problem_size": size,
                "method": method,
                "instance": instance.name,
                "makespan": makespan,
            })

    text_lines.append("")

    return (
        summary_rows,
        per_instance_rows,
        text_lines,
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DI-GNN-PPO and classical "
            "RCPSP baselines on the held-out test split."
        )
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Training run containing best.pt. "
            "If omitted, uses di_gnn_<size>."
        ),
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(PROBLEM_SIZES),
        choices=PROBLEM_SIZES,
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
        help=(
            "Explicit checkpoint. "
            "Only valid when evaluating one problem size."
        ),
    )

    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Evaluate only DI-GNN-PPO.",
    )

    parser.add_argument(
        "--no-model",
        action="store_true",
        help=(
            "Skip DI-GNN-PPO and evaluate only classical baselines. "
            "Normally unnecessary."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for the Random baseline. "
            "Defaults to the training configuration seed."
        ),
    )

    args = parser.parse_args()

    if (
        args.checkpoint is not None
        and len(args.sizes) != 1
    ):
        parser.error(
            "--checkpoint requires exactly one --sizes entry."
        )

    if (
        args.checkpoint is not None
        and args.run_name is not None
    ):
        parser.error(
            "Use either --checkpoint or --run-name, not both."
        )

    return args


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    args = parse_args()

    device = get_device(
        args.device
    )

    print("=" * 72)
    print("DI-GNN-PPO EVALUATION")
    print("=" * 72)

    print(
        f"Device: {device}"
    )

    print(
        f"Sizes: {', '.join(args.sizes)}"
    )

    if args.run_name is not None:
        print(
            f"Run: {args.run_name}"
        )

    # ----------------------------------------------------------------------
    # Evaluate each requested problem size
    # ----------------------------------------------------------------------

    all_summary_rows: List[Dict] = []
    all_per_instance_rows: List[Dict] = []
    all_text_lines: List[str] = []

    for size in args.sizes:

        (
            summary_rows,
            per_instance_rows,
            text_lines,
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

        all_text_lines.extend(
            text_lines
        )

    # ----------------------------------------------------------------------
    # Output directory
    # ----------------------------------------------------------------------

    if args.run_name is not None:
        output_run_name = args.run_name

    elif args.checkpoint is not None:
        output_run_name = (
            args.checkpoint.parent.name
        )

    else:
        # For multiple sizes, use the common root.
        # For a single size, this becomes di_gnn_j30 etc.
        output_run_name = (
            f"di_gnn_{args.sizes[0]}"
            if len(args.sizes) == 1
            else "di_gnn_evaluation"
        )

    out_dir = (
        args.runs_root
        / output_run_name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------------

    write_csv(
        out_dir / "evaluation_results.csv",
        all_summary_rows,
    )

    write_csv(
        out_dir / "evaluation_per_instance.csv",
        all_per_instance_rows,
    )

    (
        out_dir / "evaluation_results.txt"
    ).write_text(
        "\n".join(all_text_lines) + "\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # Final message
    # ----------------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Results overwritten in:\n"
        f"  {out_dir}"
    )


if __name__ == "__main__":
    main()

