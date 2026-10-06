"""
train.py
========
Training entry point for DI-GNN-PPO on PSPLIB RCPSP instances.

    PSPLIB -> RCPSPInstance -> RCPSPEnv -> 27-d node features
           -> multi-hop GNN actor-critic -> PPO

One command trains every requested problem size (default: j30, j60, j90),
one after another.

Speed: rollouts and validation score one tiny graph per step, which is faster on
CPU than on GPU (launch/sync overhead). So the model lives on CPU while collecting
and is moved to the GPU only for the PPO update. The algorithm is unchanged.

The train/val/test split is utils.split_dataset, the same function evaluate.py uses,
so the test split is never seen in training.

Usage
-----
    python train.py                  # j30, j60, j90 one after another
    python train.py --sizes 60 90    # only some sizes
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_DATA_ROOT, DEFAULT_RUNS_ROOT, Config
from di_gnn_model import DIGNNActorCritic
from domain_features import StaticFeatureCache
from evaluate import evaluate_policy
from ppo import PPO, RolloutBuffer
from psplib_loader import load_dataset
from rcpsp_environment import RCPSPEnv
from utils import get_device, save_checkpoint, set_seed, split_dataset, write_json


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.seed = args.seed

    for name in ("num_episodes", "episodes_per_update", "ppo_epochs",
                 "val_every_updates", "grpw_variant"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    # feature / graph ablations
    if args.no_demand_ratio:
        cfg.use_demand_ratio = False
    if args.no_criticality:
        cfg.use_criticality = False
    if args.no_structural:
        cfg.use_structural = False
    if args.no_resource_pressure:
        cfg.use_resource_pressure = False
    if args.no_bidirectional:
        cfg.bidirectional_edges = False

    cfg.validate()
    return cfg


def get_run_name(args: argparse.Namespace, size: int) -> str:
    return args.run_name or f"di_gnn_j{size}"


def collect_episode(model: DIGNNActorCritic, env: RCPSPEnv, instance,
                    buffer: RolloutBuffer) -> int:
    """Run one stochastic episode, store its transitions, return the makespan."""
    obs = env.reset(instance)
    done = False
    while not done:
        action, log_prob, value = model.act(obs, greedy=False)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, log_prob, value, reward, done)
        obs = next_obs
    return env.makespan


def run(args: argparse.Namespace, size: int) -> None:
    cfg = build_config(args)
    size_name = f"j{size}"
    run_name = get_run_name(args, size)
    run_dir = Path(args.runs_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    update_device = get_device(args.device)
    if update_device.type == "cuda":
        torch.set_num_threads(1)          # tiny graphs: 1 CPU thread is fastest for rollouts

    print("=" * 70)
    print(f"DI-GNN-PPO | {size_name.upper()} | PPO update on {update_device}, "
          f"rollouts on cpu | run={run_dir}")
    print(f"Node feature dimension: {cfg.node_feature_dim()}")
    print("=" * 70)

    instances = load_dataset(data_root=args.data_root, size=size_name)
    train_set, val_set, test_set = split_dataset(instances, cfg)
    print(f"Loaded {len(instances)} instances | "
          f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    write_json(run_dir / "config.json", {
        "size": size_name,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "config": cfg.to_dict(),
    })

    env = RCPSPEnv(cfg, StaticFeatureCache())      # one env; cache is keyed per instance
    model = DIGNNActorCritic(cfg)                  # lives on CPU except during PPO updates
    ppo = PPO(model, cfg, update_device)
    rng = np.random.default_rng(cfg.seed)
    buffer = RolloutBuffer()

    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    def validate() -> float:
        return float(np.mean(evaluate_policy(model, env, val_set)))

    def save(name: str, update: int, val: float) -> None:
        save_checkpoint(run_dir / name, model, cfg, update=update, validation_makespan=val)

    best_val = validate()
    best_update = 0
    save("best.pt", 0, best_val)
    print(f"Update 0 | validation makespan = {best_val:.3f}  -> saved best checkpoint")

    start_time = time.time()
    update = 0
    episode_count = 0

    while episode_count < cfg.num_episodes:
        buffer.clear()
        makespans = []

        for _ in range(cfg.episodes_per_update):
            instance = train_set[int(rng.integers(len(train_set)))]
            makespans.append(collect_episode(model, env, instance, buffer))
            episode_count += 1

        update += 1
        model.to(update_device)
        stats = ppo.update(buffer, rng)
        model.cpu()

        print(f"Update {update:4d} | Episodes {episode_count:5d}/{cfg.num_episodes} | "
              f"Makespan {np.mean(makespans):7.2f} | "
              f"Policy {stats['policy_loss']:8.4f} | Value {stats['value_loss']:7.4f} | "
              f"Entropy {stats['entropy']:6.4f} | KL {stats['approx_kl']:8.5f} | "
              f"Clip {stats['clip_frac']:5.3f} | EV {stats['explained_variance']:6.3f} | "
              f"{(time.time() - start_time) / 60:6.1f} min")

        if update % cfg.val_every_updates == 0 or episode_count >= cfg.num_episodes:
            val = validate()
            print(f"  Validation: {val:.3f}")
            if val < best_val:
                best_val, best_update = val, update
                save("best.pt", update, val)
                print("  -> saved best checkpoint")

    save("last.pt", update, best_val)

    print("\n" + "=" * 70)
    print(f"Training finished | {size_name.upper()} | "
          f"{(time.time() - start_time) / 60:.1f} min")
    print(f"Best validation: {best_val:.3f} (update {best_update})")
    print(f"Checkpoint: {run_dir / 'best.pt'}")
    print("=" * 70)
    print(f"\nNext: python evaluate.py --run-name {run_name} --sizes {size_name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DI-GNN-PPO on PSPLIB RCPSP instances.")
    p.add_argument("--sizes", type=int, nargs="+", choices=[30, 60, 90, 120],
                   default=[30, 60, 90])
    p.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--runs-root", type=str, default=str(DEFAULT_RUNS_ROOT))
    p.add_argument("--run-name", type=str, default=None,
                   help="only valid with a single size (default: di_gnn_j<size>)")
    p.add_argument("--device", type=str, default="auto", help="device for PPO updates")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--episodes-per-update", type=int, default=None)
    p.add_argument("--ppo-epochs", type=int, default=None)
    p.add_argument("--val-every-updates", type=int, default=None)
    p.add_argument("--grpw-variant", type=str, choices=["immediate", "all"], default=None)
    p.add_argument("--no-demand-ratio", action="store_true")
    p.add_argument("--no-criticality", action="store_true")
    p.add_argument("--no-structural", action="store_true")
    p.add_argument("--no-resource-pressure", action="store_true")
    p.add_argument("--no-bidirectional", action="store_true")
    args = p.parse_args()

    args.sizes = list(dict.fromkeys(args.sizes))          # drop duplicates, keep order
    if args.run_name is not None and len(args.sizes) != 1:
        p.error("--run-name requires exactly one size")
    return args


def main() -> None:
    args = parse_args()
    for size in args.sizes:
        run(args, size)

    if len(args.sizes) > 1:
        print("\nAll sizes finished. Evaluate with:")
        for size in args.sizes:
            print(f"  python evaluate.py --run-name {get_run_name(args, size)} --sizes j{size}")


if __name__ == "__main__":
    main()