from pathlib import Path
import csv
import random
import sys

import numpy as np
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

from loader import RCPSPLoader
from environment import RCPSPEnvironment

from mlp_ppo.features import RCPSPFeatureExtractor

from mlp_ppo.agent import RCPSPMLPPPOAgent

from mlp_ppo.config import (
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    HIDDEN_DIM,
    MLP_LAYERS,
    LEARNING_RATE,
    CLIP_EPSILON,
    ENTROPY_COEF,
    VALUE_COEF,
    GAMMA,
    GAE_LAMBDA,
    MAX_GRAD_NORM,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM BASELINE
# ============================================================

def evaluate_random(project):

    env = RCPSPEnvironment(project)

    env.reset()

    while not env.is_done():

        available = env.get_available_actions()

        if not available:

            if not env.running_activities:
                raise RuntimeError(
                    "Random policy cannot make progress."
                )

            env.advance_time()
            continue

        activity_id = random.choice(available)

        env.step(activity_id)

    return env.get_makespan()


# ============================================================
# SPT BASELINE
# ============================================================

def evaluate_spt(project):

    env = RCPSPEnvironment(project)

    env.reset()

    while not env.is_done():

        available = env.get_available_actions()

        if not available:

            if not env.running_activities:
                raise RuntimeError(
                    "SPT policy cannot make progress."
                )

            env.advance_time()
            continue

        activity_id = min(
            available,
            key=lambda aid:
                env.real_activities[aid].duration,
        )

        env.step(activity_id)

    return env.get_makespan()


# ============================================================
# MLP-PPO
# ============================================================

def evaluate_mlp(
    project,
    feature_extractor,
    agent,
):

    env = RCPSPEnvironment(project)

    env.reset()

    while not env.is_done():

        available = env.get_available_actions()

        if not available:

            if not env.running_activities:
                raise RuntimeError(
                    "MLP-PPO cannot make progress."
                )

            env.advance_time()
            continue

        candidate_features = (
            feature_extractor.get_candidate_features(env)
        )

        global_features = (
            feature_extractor.get_global_features(env)
        )

        action_index, _, _ = agent.select_action(
            candidate_features,
            global_features,
            deterministic=True,
        )

        activity_id = available[action_index]

        env.step(activity_id)

    return env.get_makespan()


# ============================================================
# METRICS
# ============================================================

def mean(values):

    return float(np.mean(values))


def improvement(
    baseline,
    method,
):

    return (
        (baseline - method)
        / baseline
        * 100.0
    )


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate_dataset(dataset_name):

    print()
    print("=" * 70)
    print(f"EVALUATION: {dataset_name.upper()}")
    print("=" * 70)

    dataset_dir = DATA_DIR / dataset_name

    loader = RCPSPLoader(
        folder=str(dataset_dir),
        split=(
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
        ),
        seed=SEED,
    )

    train_projects = loader.train_set
    test_projects = loader.test_set

    print(
        f"Test instances: {len(test_projects)}"
    )

    # --------------------------------------------------------
    # Feature extractor
    # --------------------------------------------------------

    feature_extractor = RCPSPFeatureExtractor(
        train_projects[0]
    )

    feature_dim = (
        feature_extractor.get_feature_dim()
    )

    global_dim = (
        feature_extractor.get_global_dim()
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = RCPSPMLPPPOAgent(
        feature_dim=feature_dim,
        global_dim=global_dim,
        hidden_dim=HIDDEN_DIM,
        layers=MLP_LAYERS,
        learning_rate=LEARNING_RATE,
        clip_epsilon=CLIP_EPSILON,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        device=DEVICE,
    )

    model_path = (
        MODEL_DIR
        / f"mlp_ppo_{dataset_name}_best.pth"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"MLP-PPO checkpoint not found: {model_path}"
        )

    agent.load(str(model_path))

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    random_results = []
    spt_results = []
    mlp_results = []

    rows = []

    for index, project in enumerate(
        test_projects,
        start=1,
    ):

        random_makespan = evaluate_random(project)

        spt_makespan = evaluate_spt(project)

        mlp_makespan = evaluate_mlp(
            project,
            feature_extractor,
            agent,
        )

        random_results.append(random_makespan)
        spt_results.append(spt_makespan)
        mlp_results.append(mlp_makespan)

        rows.append(
            {
                "instance": project.project_id,
                "random": random_makespan,
                "spt": spt_makespan,
                "mlp_ppo": mlp_makespan,
            }
        )

        print(
            f"[{index:02d}/{len(test_projects)}] "
            f"{project.project_id} | "
            f"Random {random_makespan:.2f} | "
            f"SPT {spt_makespan:.2f} | "
            f"MLP-PPO {mlp_makespan:.2f}"
        )

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    random_mean = mean(random_results)
    spt_mean = mean(spt_results)
    mlp_mean = mean(mlp_results)

    # --------------------------------------------------------
    # Wins
    # --------------------------------------------------------

    mlp_better_random = sum(
        mlp < random
        for mlp, random in zip(
            mlp_results,
            random_results,
        )
    )

    mlp_better_spt = sum(
        mlp < spt
        for mlp, spt in zip(
            mlp_results,
            spt_results,
        )
    )

    mlp_best_both = sum(
        mlp < random and mlp < spt
        for mlp, random, spt in zip(
            mlp_results,
            random_results,
            spt_results,
        )
    )

    # --------------------------------------------------------
    # Reductions
    # --------------------------------------------------------

    reduction_random = improvement(
        random_mean,
        mlp_mean,
    )

    reduction_spt = improvement(
        spt_mean,
        mlp_mean,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(f"{dataset_name.upper()} SUMMARY")
    print("-" * 70)

    print(f"Random mean: {random_mean:.2f}")
    print(f"SPT mean: {spt_mean:.2f}")
    print(f"MLP-PPO mean: {mlp_mean:.2f}")

    print()

    print(
        f"MLP-PPO reduction vs Random: "
        f"{reduction_random:.2f}%"
    )

    print(
        f"MLP-PPO reduction vs SPT: "
        f"{reduction_spt:.2f}%"
    )

    print()

    print(
        f"MLP-PPO better than Random: "
        f"{mlp_better_random}/{len(test_projects)}"
    )

    print(
        f"MLP-PPO better than SPT: "
        f"{mlp_better_spt}/{len(test_projects)}"
    )

    print(
        f"MLP-PPO better than both: "
        f"{mlp_best_both}/{len(test_projects)}"
    )

    # --------------------------------------------------------
    # Save detailed per-instance CSV
    # --------------------------------------------------------

    result_path = (
        RESULTS_DIR
        / f"evaluation_{dataset_name}.csv"
    )

    with open(
        result_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "instance",
                "random",
                "spt",
                "mlp_ppo",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved: {result_path}")

    # --------------------------------------------------------
    # Return summary
    # --------------------------------------------------------

    return {
        "dataset": dataset_name.upper(),
        "random_mean": random_mean,
        "spt_mean": spt_mean,
        "mlp_mean": mlp_mean,
        "reduction_random": reduction_random,
        "reduction_spt": reduction_spt,
        "better_random": mlp_better_random,
        "better_spt": mlp_better_spt,
        "better_both": mlp_best_both,
        "test_count": len(test_projects),
    }


# ============================================================
# SAVE FINAL TEXT SUMMARY
# ============================================================

def save_final_results(all_results):

    result_path = (
        RESULTS_DIR / "final_results.txt"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "RCPSP MLP-PPO EVALUATION RESULTS\n"
        )

        file.write(
            "================================\n\n"
        )

        file.write(
            "Evaluation protocol:\n"
        )

        file.write(
            "- Test set: 48 held-out instances per dataset\n"
        )

        file.write(
            "- Policies: Random, SPT, MLP-PPO\n"
        )

        file.write(
            "- MLP-PPO uses the best validation checkpoint\n"
        )

        file.write(
            "- Lower makespan is better\n\n"
        )

        file.write(
            "================================\n\n"
        )

        for result in all_results:

            file.write(
                f"{result['dataset']}\n"
            )

            file.write(
                "-" * 30 + "\n"
            )

            file.write(
                f"Test instances: "
                f"{result['test_count']}\n"
            )

            file.write(
                f"Random mean makespan: "
                f"{result['random_mean']:.2f}\n"
            )

            file.write(
                f"SPT mean makespan: "
                f"{result['spt_mean']:.2f}\n"
            )

            file.write(
                f"MLP-PPO mean makespan: "
                f"{result['mlp_mean']:.2f}\n"
            )

            file.write("\n")

            file.write(
                f"MLP-PPO reduction vs Random: "
                f"{result['reduction_random']:.2f}%\n"
            )

            file.write(
                f"MLP-PPO reduction vs SPT: "
                f"{result['reduction_spt']:.2f}%\n"
            )

            file.write("\n")

            file.write(
                f"MLP-PPO better than Random: "
                f"{result['better_random']}/"
                f"{result['test_count']}\n"
            )

            file.write(
                f"MLP-PPO better than SPT: "
                f"{result['better_spt']}/"
                f"{result['test_count']}\n"
            )

            file.write(
                f"MLP-PPO better than both: "
                f"{result['better_both']}/"
                f"{result['test_count']}\n"
            )

            file.write("\n\n")

        # ----------------------------------------------------
        # Compact comparison table
        # ----------------------------------------------------

        file.write(
            "================================\n"
        )

        file.write(
            "COMPACT COMPARISON\n"
        )

        file.write(
            "================================\n\n"
        )

        file.write(
            "Dataset | Random | SPT | MLP-PPO | "
            "vs Random | vs SPT\n"
        )

        file.write(
            "-" * 75 + "\n"
        )

        for result in all_results:

            file.write(
                f"{result['dataset']:7} | "
                f"{result['random_mean']:6.2f} | "
                f"{result['spt_mean']:5.2f} | "
                f"{result['mlp_mean']:7.2f} | "
                f"{result['reduction_random']:9.2f}% | "
                f"{result['reduction_spt']:6.2f}%\n"
            )

    print()
    print("=" * 70)
    print("FINAL RESULTS FILE")
    print("=" * 70)
    print(f"Saved: {result_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("RCPSP MLP-PPO EVALUATION")
    print("=" * 70)

    print(f"Data root: {DATA_DIR}")
    print(f"Device: {DEVICE}")

    all_results = []

    for dataset_name in (
        "j30",
        "j60",
        "j90",
    ):

        result = evaluate_dataset(
            dataset_name
        )

        all_results.append(result)

    save_final_results(
        all_results
    )

    print()
    print("=" * 70)
    print("ALL EVALUATIONS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

