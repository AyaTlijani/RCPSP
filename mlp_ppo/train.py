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
# PARENT PROJECT IMPORTS
# ============================================================

from loader import RCPSPLoader
from environment import RCPSPEnvironment


# ============================================================
# MLP-PPO IMPORTS
# ============================================================

from mlp_ppo.features import RCPSPFeatureExtractor

from mlp_ppo.agent import (
    RCPSPMLPPPOAgent,
    Transition,
)

from mlp_ppo.config import (
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    TOTAL_EPISODES,
    EPISODES_PER_UPDATE,
    UPDATE_EPOCHS,
    LEARNING_RATE,
    GAMMA,
    GAE_LAMBDA,
    CLIP_EPSILON,
    ENTROPY_COEF,
    VALUE_COEF,
    MAX_GRAD_NORM,
    HIDDEN_DIM,
    MLP_LAYERS,
    REWARD_SCALE,
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
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# RUN ONE EPISODE
# ============================================================

def run_episode(
    project,
    feature_extractor,
    agent,
):

    env = RCPSPEnvironment(project)

    env.reset()

    transitions = []

    total_reward = 0.0

    while not env.is_done():

        available_actions = (
            env.get_available_actions()
        )

        # ----------------------------------------------------
        # No decision possible:
        # advance to next event.
        # ----------------------------------------------------

        if not available_actions:

            if not env.running_activities:

                raise RuntimeError(
                    "No available activities and "
                    "no running activities."
                )

            env.advance_time()

            continue

        # ----------------------------------------------------
        # Build candidate features.
        # ----------------------------------------------------

        candidate_features = (
            feature_extractor
            .get_candidate_features(env)
        )

        global_features = (
            feature_extractor
            .get_global_features(env)
        )

        # ----------------------------------------------------
        # PPO chooses one eligible activity.
        # ----------------------------------------------------

        action_index, log_prob, value = (
            agent.select_action(
                candidate_features,
                global_features,
                deterministic=False,
            )
        )

        activity_id = (
            available_actions[action_index]
        )

        previous_time = env.current_time

        # ----------------------------------------------------
        # Start selected activity.
        # ----------------------------------------------------

        env.step(activity_id)

        # ----------------------------------------------------
        # Advance until another decision point.
        # ----------------------------------------------------

        while (
            not env.is_done()
            and not env.get_available_actions()
        ):

            env.advance_time()

        elapsed = (
            env.current_time
            - previous_time
        )

        reward = (
            -float(elapsed)
            / REWARD_SCALE
        )

        total_reward += reward

        # ----------------------------------------------------
        # Store PPO transition.
        # ----------------------------------------------------

        transitions.append(
            Transition(
                features=candidate_features,
                global_features=global_features,
                action_index=action_index,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=env.is_done(),
            )
        )

    makespan = env.get_makespan()

    return (
        makespan,
        total_reward,
        transitions,
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(
    projects,
    feature_extractor,
    agent,
):

    if not projects:
        return float("inf")

    makespans = []

    for project in projects:

        env = RCPSPEnvironment(project)

        env.reset()

        while not env.is_done():

            available_actions = (
                env.get_available_actions()
            )

            if not available_actions:

                if not env.running_activities:

                    raise RuntimeError(
                        "Validation environment "
                        "cannot make progress."
                    )

                env.advance_time()

                continue

            candidate_features = (
                feature_extractor
                .get_candidate_features(env)
            )

            global_features = (
                feature_extractor
                .get_global_features(env)
            )

            action_index, _, _ = (
                agent.select_action(
                    candidate_features,
                    global_features,
                    deterministic=True,
                )
            )

            activity_id = (
                available_actions[action_index]
            )

            env.step(activity_id)

        makespans.append(
            env.get_makespan()
        )

    return float(
        np.mean(makespans)
    )


# ============================================================
# TRAIN ONE DATASET
# ============================================================

def train_dataset(
    dataset_name,
):

    print()
    print("=" * 60)
    print(
        f"MLP-PPO TRAINING: "
        f"{dataset_name.upper()}"
    )
    print("=" * 60)

    dataset_dir = (
        DATA_DIR / dataset_name
    )

    print(
        f"Dataset: {dataset_dir}"
    )

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{dataset_dir}"
        )

    files = sorted(
        dataset_dir.glob("*.sm")
    )

    if not files:

        raise FileNotFoundError(
            f"No .sm files found in "
            f"{dataset_dir}"
        )

    print(
        f"Found {len(files)} .sm files"
    )

    # --------------------------------------------------------
    # Split THIS dataset only.
    # --------------------------------------------------------

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
    val_projects = loader.val_set
    test_projects = loader.test_set

    print(
        f"Train: {len(train_projects)}"
    )

    print(
        f"Validation: {len(val_projects)}"
    )

    print(
        f"Test: {len(test_projects)}"
    )

    # --------------------------------------------------------
    # Feature extractor.
    # --------------------------------------------------------

    reference_project = (
        train_projects[0]
    )

    feature_extractor = (
        RCPSPFeatureExtractor(
            reference_project
        )
    )

    feature_dim = (
        feature_extractor
        .get_feature_dim()
    )

    global_dim = (
        feature_extractor
        .get_global_dim()
    )

    print(
        f"Activity feature dimension: "
        f"{feature_dim}"
    )

    print(
        f"Global feature dimension: "
        f"{global_dim}"
    )

    # --------------------------------------------------------
    # Agent.
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

    # --------------------------------------------------------
    # Paths.
    # --------------------------------------------------------

    best_model_path = (
        MODEL_DIR
        / f"mlp_ppo_{dataset_name}_best.pth"
    )

    final_model_path = (
        MODEL_DIR
        / f"mlp_ppo_{dataset_name}_final.pth"
    )

    csv_path = (
        RESULTS_DIR
        / f"training_{dataset_name}.csv"
    )

    # --------------------------------------------------------
    # Training.
    # --------------------------------------------------------

    best_val = float("inf")

    collected_transitions = []

    rows = []

    episode_makespans = []

    for episode in range(
        1,
        TOTAL_EPISODES + 1,
    ):

        project = random.choice(
            train_projects
        )

        (
            makespan,
            reward,
            transitions,
        ) = run_episode(
            project,
            feature_extractor,
            agent,
        )

        collected_transitions.extend(
            transitions
        )

        episode_makespans.append(
            makespan
        )

        # ----------------------------------------------------
        # PPO update every N episodes.
        # ----------------------------------------------------

        if (
            episode
            % EPISODES_PER_UPDATE
            == 0
        ):

            agent.update(
                collected_transitions,
                update_epochs=UPDATE_EPOCHS,
            )

            collected_transitions = []

        recent = episode_makespans[
            -20:
        ]

        recent_mean = float(
            np.mean(recent)
        )

        rows.append(
            {
                "episode": episode,
                "instance": project.project_id,
                "makespan": makespan,
                "reward": reward,
                "recent_mean_makespan":
                    recent_mean,
            }
        )

        # ----------------------------------------------------
        # Console output.
        # ----------------------------------------------------

        if (
            episode == 1
            or episode % 10 == 0
        ):

            print(
                f"[{dataset_name}] "
                f"Episode "
                f"{episode:4d}/"
                f"{TOTAL_EPISODES} | "
                f"Makespan "
                f"{makespan:.2f} | "
                f"Recent mean "
                f"{recent_mean:.2f}"
            )

        # ----------------------------------------------------
        # Validation.
        # ----------------------------------------------------

        if (
            episode % 25 == 0
            or episode == TOTAL_EPISODES
        ):

            val_makespan = (
                evaluate_validation(
                    val_projects,
                    feature_extractor,
                    agent,
                )
            )

            print(
                f"[{dataset_name}] "
                f"Validation "
                f"episode {episode}: "
                f"{val_makespan:.2f}"
            )

            if (
                val_makespan
                < best_val
            ):

                best_val = (
                    val_makespan
                )

                agent.save(
                    str(best_model_path)
                )

                print(
                    f"Saved best model: "
                    f"{best_model_path}"
                )

    # --------------------------------------------------------
    # Final model.
    # --------------------------------------------------------

    agent.save(
        str(final_model_path)
    )

    # --------------------------------------------------------
    # CSV.
    # --------------------------------------------------------

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "instance",
                "makespan",
                "reward",
                "recent_mean_makespan",
            ],
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print(
        f"{dataset_name.upper()} training finished."
    )

    print(
        f"Best validation makespan: "
        f"{best_val:.2f}"
    )

    print(
        f"Best model: "
        f"{best_model_path}"
    )

    print(
        f"Final model: "
        f"{final_model_path}"
    )

    print(
        f"Training log: "
        f"{csv_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RCPSP MLP-PPO TRAINING")
    print("=" * 60)

    print(
        f"Data root: {DATA_DIR}"
    )

    print(
        f"Device: {DEVICE}"
    )

    set_seed(SEED)

    # Each project size is trained independently.
    for dataset_name in (
        "j30",
        "j60",
        "j90",
    ):

        train_dataset(
            dataset_name
        )

    print()
    print("=" * 60)
    print("ALL MLP-PPO TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()