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

DATASETS = ["j30", "j60", "j90"]
SEED = 1
SAMPLE_SIZE = 6
OUTPUT_FILE = "evaluation_results.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def build_graph(project, resource_ids, state, device):
    graph = RCPSPGraphBuilder(project, resource_ids, state).build_graph()
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
# BASELINES & PPO
# ============================================================

def run_random(project):
    def select(state, available):
        return random.choice(available)

    return run_until_done(RCPSPEnvironment(project), select)


def run_spt(project):
    activities = {a.id: a for a in project.real_activities}

    def select(state, available):
        return min(available, key=lambda aid: activities[aid].duration)

    return run_until_done(RCPSPEnvironment(project), select)


def run_ppo(agent, project, resource_ids, device):
    agent.eval()

    def select(state, available):
        graph = build_graph(project, resource_ids, state, device)
        r_free = get_r_free(state, resource_ids, device)
        action, _, _ = agent.select_action(graph, r_free, available, greedy=True)
        return action

    with torch.no_grad():
        return run_until_done(RCPSPEnvironment(project), select)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_dataset(dataset: str, f) -> dict | None:
    data_folder = Path("data") / dataset
    model_path = Path("checkpoints") / dataset / f"agent_{dataset}_final.pt"

    if not model_path.exists():
        f.write("=" * 60 + "\n")
        f.write(f"{dataset.upper()} - SKIPPED\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model not found: {model_path}\n\n")
        return None

    loader = RCPSPLoader(str(data_folder), seed=SEED)
    test_set = loader.test_set
    resource_ids = RCPSPLoader.resource_ids(test_set[0])
    resource_dim = len(resource_ids)
    node_feature_dim = 3 + 1 + 1 + 2 + resource_dim

    agent = RCPSPPPOAgent(
        node_feature_dim=node_feature_dim,
        resource_dim=resource_dim,
        hidden_dim=32,
    ).to(DEVICE)
    agent.load_state_dict(torch.load(model_path, map_location=DEVICE))

    random_ms, spt_ms, ppo_ms = [], [], []
    for project in test_set:
        random_ms.append(run_random(project))
        spt_ms.append(run_spt(project))
        ppo_ms.append(run_ppo(agent, project, resource_ids, DEVICE))

    random_ms = np.array(random_ms)
    spt_ms = np.array(spt_ms)
    ppo_ms = np.array(ppo_ms)
    total = len(test_set)

    random_mean = random_ms.mean()
    spt_mean = spt_ms.mean()
    ppo_mean = ppo_ms.mean()

    ppo_vs_random = (random_mean - ppo_mean) / random_mean * 100
    ppo_vs_spt = (spt_mean - ppo_mean) / spt_mean * 100

    better_random = np.sum(ppo_ms < random_ms)
    better_spt = np.sum(ppo_ms < spt_ms)
    best = np.sum((ppo_ms < random_ms) & (ppo_ms < spt_ms))

    # Write results
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write(f"                    {dataset.upper()}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test instances: {total}\n\n")

    f.write("AVERAGE MAKESPAN\n")
    f.write("-" * 40 + "\n")
    f.write(f"Random baseline : {random_mean:.2f}\n")
    f.write(f"SPT baseline    : {spt_mean:.2f}\n")
    f.write(f"PPO model       : {ppo_mean:.2f}\n\n")

    f.write("PPO IMPROVEMENT\n")
    f.write("-" * 40 + "\n")
    f.write(f"Compared with Random : {ppo_vs_random:.2f}%\n")
    f.write(f"Compared with SPT    : {ppo_vs_spt:.2f}%\n\n")

    f.write("PPO PERFORMANCE\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"PPO better than Random : {better_random}/{total} "
        f"({better_random / total * 100:.1f}%)\n"
    )
    f.write(
        f"PPO better than SPT    : {better_spt}/{total} "
        f"({better_spt / total * 100:.1f}%)\n"
    )
    f.write(
        f"PPO best against both  : {best}/{total} "
        f"({best / total * 100:.1f}%)\n\n"
    )

    # Sample instances
    sample_indices = random.sample(range(total), min(SAMPLE_SIZE, total))
    f.write("EXAMPLE TEST INSTANCES\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Instance':<12}{'Random':>12}{'SPT':>12}{'PPO':>12}\n")
    f.write("-" * 48 + "\n")
    for i in sample_indices:
        f.write(
            f"{i + 1:<12}"
            f"{random_ms[i]:>12.1f}"
            f"{spt_ms[i]:>12.1f}"
            f"{ppo_ms[i]:>12.1f}\n"
        )
    f.write("\n")

    return {
        "dataset": dataset,
        "random": random_mean,
        "spt": spt_mean,
        "ppo": ppo_mean,
    }


def main():
    dataset_results = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("           RCPSP MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Seed: {SEED}\n")
        f.write("Methods: Random | SPT | PPO\n\n")

        for dataset in DATASETS:
            result = evaluate_dataset(dataset, f)
            if result is not None:
                dataset_results.append(result)

        # Final summary
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("                    FINAL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Dataset':<12}{'Random':>12}{'SPT':>12}{'PPO':>12}\n")
        f.write("-" * 48 + "\n")
        for result in dataset_results:
            f.write(
                f"{result['dataset'].upper():<12}"
                f"{result['random']:>12.2f}"
                f"{result['spt']:>12.2f}"
                f"{result['ppo']:>12.2f}\n"
            )
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("Evaluation complete.\n")
        f.write("=" * 60 + "\n")

    print()
    print("=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()