import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from loader import RCPSPLoader
from graph_representation import RCPSPGraphBuilder
from environment import RCPSPEnvironment
from agent import RCPSPPPOAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints"

# only train + test, no validation split
DATASETS = ["j30", "j60", "j90"]

SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# speed / quality balance
MAX_TRAIN_INSTANCES = 120
TOTAL_EPISODES = 400
EPISODES_PER_UPDATE = 8
UPDATE_EPOCHS = 2

LEARNING_RATE = 3e-4
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.025
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
REWARD_SCALE = 10.0

PRINT_EVERY = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def build_graph(project, resource_ids, state, device):
    graph = RCPSPGraphBuilder(project, resource_ids, state).build_graph()
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    return graph


def get_r_free(state, resource_ids, device):
    return torch.tensor(
        [state["r_free"][r] for r in resource_ids],
        dtype=torch.float32,
        device=device
    )


def collect_episode(agent, project, resource_ids, device):
    env = RCPSPEnvironment(project)
    env.reset()

    transitions = []
    last_time = 0.0

    while True:
        state = env.get_state()
        available = state["available_actions"]

        if not available:
            if env.is_done():
                break
            env.advance_time()
            continue

        graph = build_graph(project, resource_ids, state, device)
        r_free = get_r_free(state, resource_ids, device)

        action, log_prob, value = agent.select_action(
            graph, r_free, available, greedy=False
        )

        env.step(action)
        new_time = env.current_time
        reward = -(new_time - last_time) / REWARD_SCALE
        last_time = new_time

        transitions.append({
            "graph": graph,
            "r_free": r_free,
            "available_actions": available,
            "action": action,
            "log_prob": log_prob.detach(),
            "value": value.detach(),
            "reward": reward,
        })

    return transitions, env.get_makespan()


def compute_gae(transitions):
    rewards = [t["reward"] for t in transitions]
    values = [t["value"].item() for t in transitions]

    advantages = [0.0] * len(transitions)
    last_gae = 0.0

    for t in reversed(range(len(transitions))):
        next_value = values[t + 1] if t + 1 < len(transitions) else 0.0
        delta = rewards[t] + GAMMA * next_value - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        advantages[t] = last_gae

    returns = [advantages[t] + values[t] for t in range(len(transitions))]
    return advantages, returns


def ppo_update(agent, optimizer, transitions, advantages, returns, device):
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    old_log_probs = torch.stack([t["log_prob"] for t in transitions]).to(device)

    # normalize advantages, fall back to just centering if std is tiny
    if advantages.numel() > 1:
        std = advantages.std()
        if std > 1e-5:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        else:
            advantages = advantages - advantages.mean()

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(UPDATE_EPOCHS):
        new_log_probs = []
        new_values = []
        entropies = []

        for tr in transitions:
            log_prob, entropy, value = agent.evaluate_action(
                tr["graph"], tr["r_free"], tr["available_actions"], tr["action"]
            )
            new_log_probs.append(log_prob)
            new_values.append(value.squeeze())
            entropies.append(entropy)

        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values)
        entropy = torch.stack(entropies).mean()

        ratio = (new_log_probs - old_log_probs).exp()

        surr1 = -advantages * ratio
        surr2 = -advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
        policy_loss = torch.max(surr1, surr2).mean()

        value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        loss = policy_loss - ENT_COEF * entropy + VF_COEF * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        last_policy_loss = policy_loss.item()
        last_value_loss = value_loss.item()
        last_entropy = entropy.item()

    return last_policy_loss, last_value_loss, last_entropy


def evaluate_random(dataset):
    makespans = []
    for project in dataset:
        env = RCPSPEnvironment(project)
        env.reset()
        while True:
            state = env.get_state()
            available = state["available_actions"]
            if not available:
                if env.is_done():
                    break
                env.advance_time()
                continue
            env.step(random.choice(available))
        makespans.append(env.get_makespan())
    return float(np.mean(makespans))


def evaluate_spt(dataset):
    makespans = []
    for project in dataset:
        activities = {a.id: a for a in project.real_activities}
        env = RCPSPEnvironment(project)
        env.reset()
        while True:
            state = env.get_state()
            available = state["available_actions"]
            if not available:
                if env.is_done():
                    break
                env.advance_time()
                continue
            action = min(available, key=lambda aid: activities[aid].duration)
            env.step(action)
        makespans.append(env.get_makespan())
    return float(np.mean(makespans))


def evaluate_greedy(agent, dataset, resource_ids, device):
    makespans = []
    agent.eval()
    with torch.no_grad():
        for project in dataset:
            env = RCPSPEnvironment(project)
            env.reset()
            while True:
                state = env.get_state()
                available = state["available_actions"]
                if not available:
                    if env.is_done():
                        break
                    env.advance_time()
                    continue
                graph = build_graph(project, resource_ids, state, device)
                r_free = get_r_free(state, resource_ids, device)
                action, _, _ = agent.select_action(
                    graph, r_free, available, greedy=True
                )
                env.step(action)
            makespans.append(env.get_makespan())
    agent.train()
    return float(np.mean(makespans))


results = {}

for DATASET in DATASETS:
    print("\n" + "=" * 70)
    print(f"  STARTING DATASET: {DATASET.upper()}")
    print("=" * 70)

    DATA_FOLDER = DATA_ROOT / DATASET
    CHECKPOINT_DIR = CHECKPOINT_ROOT / DATASET
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    loader = RCPSPLoader(DATA_FOLDER, seed=SEED)

    # merge original train + val into one bigger training set
    train_set = loader.train_set + loader.val_set
    test_set = loader.test_set

    if len(train_set) > MAX_TRAIN_INSTANCES:
        print(f"[INFO] Reducing train set: {len(train_set)} to {MAX_TRAIN_INSTANCES}")
        train_set = train_set[:MAX_TRAIN_INSTANCES]

    # keep loader.next_instance working off the trimmed set
    loader.train_set = train_set

    resource_ids = RCPSPLoader.resource_ids(train_set[0])
    RESOURCE_DIM = len(resource_ids)
    NODE_FEATURE_DIM = 3 + 1 + 1 + 2 + RESOURCE_DIM

    agent = RCPSPPPOAgent(
        node_feature_dim=NODE_FEATURE_DIM,
        resource_dim=RESOURCE_DIM,
        hidden_dim=32
    ).to(DEVICE)

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    print(f"Train instances : {len(train_set)}")
    print(f"Test instances  : {len(test_set)}")
    print(f"Resources       : {RESOURCE_DIM}")
    print(f"Node features   : {NODE_FEATURE_DIM}")
    print(f"Device          : {DEVICE}")
    print()

    agent.train()
    recent_makespans = deque(maxlen=40)
    start_time = time.time()
    episodes_done = 0
    num_updates = TOTAL_EPISODES // EPISODES_PER_UPDATE

    for update in range(1, num_updates + 1):
        batch_transitions = []
        batch_advantages = []
        batch_returns = []

        for _ in range(EPISODES_PER_UPDATE):
            project = loader.next_instance()
            transitions, makespan = collect_episode(
                agent, project, resource_ids, DEVICE
            )
            episodes_done += 1
            recent_makespans.append(makespan)

            if len(transitions) < 2:
                continue

            advantages, returns = compute_gae(transitions)
            batch_transitions.extend(transitions)
            batch_advantages.extend(advantages)
            batch_returns.extend(returns)

        if len(batch_transitions) < 2:
            continue

        policy_loss, value_loss, entropy = ppo_update(
            agent, optimizer, batch_transitions,
            batch_advantages, batch_returns, DEVICE
        )

        if update % PRINT_EVERY == 0 or update == num_updates:
            elapsed = time.time() - start_time
            avg_makespan = np.mean(recent_makespans)
            print(
                f"[{DATASET.upper()}] "
                f"Update {update:3d}/{num_updates} | "
                f"Episodes {episodes_done:4d}/{TOTAL_EPISODES} | "
                f"Avg makespan {avg_makespan:6.2f} | "
                f"Policy {policy_loss:+.4f} | "
                f"Value {value_loss:.4f} | "
                f"Entropy {entropy:.4f} | "
                f"Time {elapsed:.0f}s"
            )

    print(f"\n[{DATASET.upper()}] Evaluating on TEST set...")

    random_test = evaluate_random(test_set)
    spt_test = evaluate_spt(test_set)
    ppo_test = evaluate_greedy(agent, test_set, resource_ids, DEVICE)

    print("-" * 50)
    print(f"[{DATASET.upper()}] TEST RESULTS")
    print(f"  Random : {random_test:.2f}")
    print(f"  SPT    : {spt_test:.2f}")
    print(f"  PPO    : {ppo_test:.2f}")
    print("-" * 50)

    # only keep the final model, no intermediate checkpoints
    final_path = CHECKPOINT_DIR / f"agent_{DATASET}_final.pt"
    torch.save(agent.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    results[DATASET] = {
        "random": random_test,
        "spt": spt_test,
        "ppo": ppo_test,
    }


print("\n" + "=" * 70)
print("  FINAL SUMMARY (TEST SET)")
print("=" * 70)
for ds in DATASETS:
    r = results[ds]
    print(f"{ds.upper():<6} | Random: {r['random']:6.2f} | "
          f"SPT: {r['spt']:6.2f} | PPO: {r['ppo']:6.2f}")
print("=" * 70)
print("DONE")