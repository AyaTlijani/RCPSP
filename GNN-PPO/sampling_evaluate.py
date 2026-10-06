import os
import random
from pathlib import Path

import numpy as np
import torch

from loader import RCPSPLoader
from graph_representation import RCPSPGraphBuilder
from environment import RCPSPEnvironment
from agent import RCPSPPPOAgent


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints"
OUTPUT_FILE = Path(__file__).resolve().parent / "sampling_evaluation_results.txt"

DATASETS = ["j30", "j60", "j90"]

SEED = 1
NUM_SAMPLES = 20
SAMPLE_SIZE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def build_graph(project, resource_ids, state, device):
    graph = RCPSPGraphBuilder(
        project,
        resource_ids,
        state
    ).build_graph()

    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)

    return graph


def get_r_free(state, resource_ids, device):
    return torch.tensor(
        [state["r_free"][r] for r in resource_ids],
        dtype=torch.float32,
        device=device,
    )


def run_until_done(env, select_action):
    """Run environment until completion using the given action selector."""

    env.reset()

    while True:
        state = env.get_state()
        available = state["available_actions"]

        if not available:
            if env.is_done():
                break

            env.advance_time()
            continue

        action = select_action(state, available)
        env.step(action)

    return env.get_makespan()


# ============================================================
# BASELINES
# ============================================================

def run_random(project):
    def select(state, available):
        return random.choice(available)

    return run_until_done(
        RCPSPEnvironment(project),
        select
    )


def run_spt(project):
    activities = {
        a.id: a
        for a in project.real_activities
    }

    def select(state, available):
        return min(
            available,
            key=lambda aid: activities[aid].duration
        )

    return run_until_done(
        RCPSPEnvironment(project),
        select
    )


# ============================================================
# PPO GREEDY
# ============================================================

def run_ppo_greedy(agent, project, resource_ids, device):

    agent.eval()

    def select(state, available):

        graph = build_graph(
            project,
            resource_ids,
            state,
            device
        )

        r_free = get_r_free(
            state,
            resource_ids,
            device
        )

        action, _, _ = agent.select_action(
            graph,
            r_free,
            available,
            greedy=True
        )

        return action

    with torch.no_grad():
        return run_until_done(
            RCPSPEnvironment(project),
            select
        )


# ============================================================
# PPO STOCHASTIC
# ============================================================

def run_ppo_sampled(agent, project, resource_ids, device):

    agent.eval()

    def select(state, available):

        graph = build_graph(
            project,
            resource_ids,
            state,
            device
        )

        r_free = get_r_free(
            state,
            resource_ids,
            device
        )

        # IMPORTANT:
        # greedy=False makes the PPO policy sample from
        # its action distribution instead of taking argmax.
        action, _, _ = agent.select_action(
            graph,
            r_free,
            available,
            greedy=False
        )

        return action

    with torch.no_grad():
        return run_until_done(
            RCPSPEnvironment(project),
            select
        )


# ============================================================
# SAMPLING EVALUATION
# ============================================================

def run_sampling(
    agent,
    project,
    resource_ids,
    device,
    num_samples
):
    """
    Run multiple stochastic PPO trajectories for one instance.

    Returns:
        greedy_makespan
        best_sampled_makespan
        mean_sampled_makespan
        median_sampled_makespan
        std_sampled_makespan
        all_sampled_makespans
    """

    # First get the normal deterministic PPO result.
    greedy_makespan = run_ppo_greedy(
        agent,
        project,
        resource_ids,
        device
    )

    sampled_makespans = []

    for _ in range(num_samples):
        makespan = run_ppo_sampled(
            agent,
            project,
            resource_ids,
            device
        )

        sampled_makespans.append(makespan)

    sampled_makespans = np.array(
        sampled_makespans,
        dtype=float
    )

    return {
        "greedy": float(greedy_makespan),
        "best": float(np.min(sampled_makespans)),
        "mean": float(np.mean(sampled_makespans)),
        "median": float(np.median(sampled_makespans)),
        "std": float(np.std(sampled_makespans)),
        "samples": sampled_makespans,
    }


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate_dataset(dataset: str, f):

    data_folder = DATA_ROOT / dataset

    model_path = (
        CHECKPOINT_ROOT
        / dataset
        / f"agent_{dataset}_final.pt"
    )

    if not model_path.exists():

        f.write("=" * 70 + "\n")
        f.write(f"{dataset.upper()} - SKIPPED\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model not found: {model_path}\n\n")

        print(f"{dataset.upper()}: checkpoint not found")
        return None

    print()
    print("=" * 70)
    print(f"{dataset.upper()}")
    print("=" * 70)

    print(f"Checkpoint: {model_path}")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    loader = RCPSPLoader(
        str(data_folder),
        seed=SEED
    )

    test_set = loader.test_set

    resource_ids = RCPSPLoader.resource_ids(
        test_set[0]
    )

    resource_dim = len(resource_ids)

    node_feature_dim = (
        3
        + 1
        + 1
        + 2
        + resource_dim
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    agent = RCPSPPPOAgent(
        node_feature_dim=node_feature_dim,
        resource_dim=resource_dim,
        hidden_dim=32,
    ).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    agent.load_state_dict(checkpoint)
    agent.eval()

    print(f"Test instances: {len(test_set)}")
    print(f"Samples per instance: {NUM_SAMPLES}")
    print()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    random_ms = []
    spt_ms = []

    greedy_ms = []
    sampled_best_ms = []
    sampled_mean_ms = []

    per_instance = []

    total = len(test_set)

    for idx, project in enumerate(test_set):

        # Classical baselines
        random_value = run_random(project)
        spt_value = run_spt(project)

        # PPO sampling
        sampling_result = run_sampling(
            agent,
            project,
            resource_ids,
            DEVICE,
            NUM_SAMPLES
        )

        random_ms.append(random_value)
        spt_ms.append(spt_value)

        greedy_ms.append(
            sampling_result["greedy"]
        )

        sampled_best_ms.append(
            sampling_result["best"]
        )

        sampled_mean_ms.append(
            sampling_result["mean"]
        )

        per_instance.append({
            "instance": idx + 1,
            "random": random_value,
            "spt": spt_value,
            "greedy": sampling_result["greedy"],
            "sampling_best": sampling_result["best"],
            "sampling_mean": sampling_result["mean"],
            "sampling_median": sampling_result["median"],
            "sampling_std": sampling_result["std"],
        })

        if (idx + 1) % 10 == 0 or idx + 1 == total:

            current_best_mean = np.mean(
                sampled_best_ms
            )

            print(
                f"{idx + 1:>3}/{total} instances | "
                f"current best-of-{NUM_SAMPLES}: "
                f"{current_best_mean:.2f}"
            )

    # --------------------------------------------------------
    # ARRAYS
    # --------------------------------------------------------

    random_ms = np.array(random_ms, dtype=float)
    spt_ms = np.array(spt_ms, dtype=float)

    greedy_ms = np.array(greedy_ms, dtype=float)
    sampled_best_ms = np.array(
        sampled_best_ms,
        dtype=float
    )

    sampled_mean_ms = np.array(
        sampled_mean_ms,
        dtype=float
    )

    total = len(test_set)

    # --------------------------------------------------------
    # MEANS
    # --------------------------------------------------------

    random_mean = random_ms.mean()
    spt_mean = spt_ms.mean()

    greedy_mean = greedy_ms.mean()
    sampling_best_mean = sampled_best_ms.mean()
    sampling_mean = sampled_mean_ms.mean()

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    greedy_to_sampling = (
        greedy_mean - sampling_best_mean
    )

    greedy_to_sampling_pct = (
        greedy_to_sampling
        / greedy_mean
        * 100
    )

    sampling_vs_random = (
        random_mean - sampling_best_mean
    ) / random_mean * 100

    sampling_vs_spt = (
        spt_mean - sampling_best_mean
    ) / spt_mean * 100

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("Results")
    print("-" * 70)

    print(
        f"Random                 "
        f"mean={random_mean:8.2f}"
    )

    print(
        f"SPT                    "
        f"mean={spt_mean:8.2f}"
    )

    print(
        f"Old GNN-PPO Greedy     "
        f"mean={greedy_mean:8.2f}"
    )

    print(
        f"GNN-PPO Sampling-{NUM_SAMPLES:<3}"
        f"mean={sampling_best_mean:8.2f}"
    )

    print()
    print("Sampling improvement")
    print("-" * 70)

    print(
        f"Greedy mean:       {greedy_mean:.3f}"
    )

    print(
        f"Best-of-{NUM_SAMPLES} mean: "
        f"{sampling_best_mean:.3f}"
    )

    print(
        f"Improvement:       "
        f"{greedy_to_sampling:+.3f} makespan "
        f"({greedy_to_sampling_pct:+.2f}%)"
    )

    print()
    print(
        f"Sampling vs Random: "
        f"{sampling_vs_random:+.2f}%"
    )

    print(
        f"Sampling vs SPT:    "
        f"{sampling_vs_spt:+.2f}%"
    )

    # --------------------------------------------------------
    # FILE OUTPUT
    # --------------------------------------------------------

    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write(f"                    {dataset.upper()}\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Test instances: {total}\n")
    f.write(
        f"Samples per instance: {NUM_SAMPLES}\n\n"
    )

    f.write("AVERAGE MAKESPAN\n")
    f.write("-" * 45 + "\n")

    f.write(
        f"Random baseline       : "
        f"{random_mean:.2f}\n"
    )

    f.write(
        f"SPT baseline          : "
        f"{spt_mean:.2f}\n"
    )

    f.write(
        f"GNN-PPO greedy        : "
        f"{greedy_mean:.2f}\n"
    )

    f.write(
        f"GNN-PPO sampling-{NUM_SAMPLES:<3}: "
        f"{sampling_best_mean:.2f}\n\n"
    )

    f.write("SAMPLING IMPROVEMENT\n")
    f.write("-" * 45 + "\n")

    f.write(
        f"Greedy mean           : "
        f"{greedy_mean:.3f}\n"
    )

    f.write(
        f"Best-of-{NUM_SAMPLES} mean      : "
        f"{sampling_best_mean:.3f}\n"
    )

    f.write(
        f"Improvement            : "
        f"{greedy_to_sampling:+.3f}\n"
    )

    f.write(
        f"Improvement percentage  : "
        f"{greedy_to_sampling_pct:+.2f}%\n\n"
    )

    f.write("BASELINE COMPARISON\n")
    f.write("-" * 45 + "\n")

    f.write(
        f"vs Random: "
        f"{sampling_vs_random:+.2f}%\n"
    )

    f.write(
        f"vs SPT   : "
        f"{sampling_vs_spt:+.2f}%\n\n"
    )

    # --------------------------------------------------------
    # EXAMPLE INSTANCES
    # --------------------------------------------------------

    sample_indices = random.sample(
        range(total),
        min(SAMPLE_SIZE, total)
    )

    f.write("EXAMPLE TEST INSTANCES\n")
    f.write("-" * 80 + "\n")

    f.write(
        f"{'Instance':<12}"
        f"{'Random':>10}"
        f"{'SPT':>10}"
        f"{'Greedy':>10}"
        f"{'Sampled':>10}"
        f"{'SampleMean':>12}\n"
    )

    f.write("-" * 80 + "\n")

    for i in sample_indices:

        row = per_instance[i]

        f.write(
            f"{row['instance']:<12}"
            f"{row['random']:>10.1f}"
            f"{row['spt']:>10.1f}"
            f"{row['greedy']:>10.1f}"
            f"{row['sampling_best']:>10.1f}"
            f"{row['sampling_mean']:>12.1f}\n"
        )

    f.write("\n")

    # --------------------------------------------------------
    # RETURN SUMMARY
    # --------------------------------------------------------

    return {
        "dataset": dataset,
        "random": random_mean,
        "spt": spt_mean,
        "greedy": greedy_mean,
        "sampling": sampling_best_mean,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    dataset_results = []

    print("=" * 70)
    print("GNN-PPO SAMPLING EVALUATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Samples per instance: {NUM_SAMPLES}")
    print(f"Seed: {SEED}")
    print()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 70 + "\n")
        f.write("           GNN-PPO SAMPLING EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Device: {DEVICE}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(
            f"Samples per instance: "
            f"{NUM_SAMPLES}\n\n"
        )

        for dataset in DATASETS:

            result = evaluate_dataset(
                dataset,
                f
            )

            if result is not None:
                dataset_results.append(result)

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("                    FINAL SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"{'Dataset':<12}"
            f"{'Random':>12}"
            f"{'SPT':>12}"
            f"{'Greedy':>12}"
            f"{'Sampling':>12}\n"
        )

        f.write("-" * 60 + "\n")

        for result in dataset_results:

            f.write(
                f"{result['dataset'].upper():<12}"
                f"{result['random']:>12.2f}"
                f"{result['spt']:>12.2f}"
                f"{result['greedy']:>12.2f}"
                f"{result['sampling']:>12.2f}\n"
            )

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("Evaluation complete.\n")
        f.write("=" * 70 + "\n")

    print()
    print("=" * 70)
    print("SAMPLING EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()