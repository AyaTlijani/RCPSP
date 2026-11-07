# RCPSP with Reinforcement Learning

**Learning to schedule projects under resource limits with graph neural networks and PPO.**

The Resource-Constrained Project Scheduling Problem (RCPSP) asks for an order of activities that respects precedence constraints and limited resources while finishing the project as early as possible (minimum makespan). It is NP-hard, and in practice it is solved with priority rules or heavy search.

This repository trains a scheduling policy with reinforcement learning: at each step the agent picks the next activity to start, and the episode return is the negative makespan. Three agents are compared on the standard PSPLIB benchmark.

## Agents

| Folder | Agent | Idea |
|---|---|---|
| [`mlp_ppo/`](mlp_ppo) | MLP-PPO | Hand-made features per activity, a multilayer perceptron policy |
| [`GNN-PPO/`](GNN-PPO) | GNN-PPO | The project is a graph (activities = nodes, precedences = edges); a graph neural network scores the eligible activities |
| [`DI-GNN-PPO/`](DI-GNN-PPO) | DI-GNN-PPO | Domain-informed GNN: 27 node features (dynamic state, resource demand, critical-path measures, structural features, live resource pressure) and a multi-hop actor-critic |

All three are trained with PPO and compared with two baselines: **Random** (a random eligible activity) and **SPT** (shortest processing time first).

## Results

Mean makespan on 48 held-out PSPLIB instances per size (lower is better). The MLP-PPO row comes from its own evaluation run; its Random and SPT baselines differ from the GNN-PPO run by at most 0.15.

**MLP-PPO and GNN-PPO** (same test sets):

| Size | Random | SPT | MLP-PPO | GNN-PPO greedy | GNN-PPO best of 20 samples |
|---|---|---|---|---|---|
| J30 | 62.44 | 62.85 | 61.81 | 60.81 | **59.69** |
| J60 | 87.27 | 86.77 | 85.83 | 82.75 | **81.90** |
| J90 | 100.02 | 102.04 | 99.21 | 96.90 | **96.23** |

GNN-PPO improves on both baselines at every size: by 3.8–6.2% over Random and 5.0–5.7% over SPT with sampling. The graph representation clearly beats the MLP, which stays within about 3% of the baselines.

**DI-GNN-PPO** (J30, its own evaluation split):

| Method | Mean makespan |
|---|---|
| Random | 64.12 |
| SPT | 63.90 |
| DI-GNN-PPO greedy | 61.29 |
| DI-GNN-PPO best of 20 samples | **60.08** |

DI-GNN-PPO is evaluated on a different test split from the two other agents, so its numbers are not directly comparable with the table above. J60 and J90 models are trained (`DI-GNN-PPO/runs/`), but their evaluation is not in the repository yet.

Full per-instance results are saved next to each agent: `GNN-PPO/*evaluation_results.txt`, `mlp_ppo/results/`, `DI-GNN-PPO/runs/`.

## Data

[PSPLIB](https://www.om-db.wi.tum.de/psplib/) single-mode instances J30, J60 and J90 (480 instances each), in [`data/`](data).

## Run it

```bash
pip install -r requirements.txt

# DI-GNN-PPO: trains J30, J60 and J90 one after another
cd DI-GNN-PPO
python train.py --sizes 30
python evaluate.py
python sampling_evaluate.py
```

`DI-GNN-PPO/train.py` has switches to remove each feature group (`--no-criticality`, `--no-structural`, `--no-resource-pressure`, ...) for ablation studies. The configuration is saved with every checkpoint, so evaluation rebuilds exactly the trained model.

## Stack

Python · PyTorch · PyTorch Geometric · NumPy · PPO
