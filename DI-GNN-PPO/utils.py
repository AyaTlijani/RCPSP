"""
utils.py
========
Small helpers only: seeding, deterministic data split, checkpoint I/O, logging and
result formatting.  No scheduling logic lives here.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from config import Config
from psplib_loader import RCPSPInstance, natural_key


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------
# deterministic train / val / test split
# --------------------------------------------------------------------------
def split_dataset(instances: Sequence[RCPSPInstance], cfg: Config
                  ) -> Tuple[List[RCPSPInstance], List[RCPSPInstance], List[RCPSPInstance]]:
    """
    Sort by name (natural order), shuffle with RandomState(cfg.seed), split 80/10/10.
    Same seed + same files  =>  identical split for every experiment and problem size.
    For 480 instances: 384 / 48 / 48.
    """
    items = sorted(instances, key=lambda inst: natural_key(inst.name))
    n = len(items)
    perm = np.random.RandomState(cfg.seed).permutation(n)
    n_train = int(round(n * cfg.train_ratio))
    n_val = int(round(n * cfg.val_ratio))
    train = [items[i] for i in perm[:n_train]]
    val = [items[i] for i in perm[n_train:n_train + n_val]]
    test = [items[i] for i in perm[n_train + n_val:]]
    if set(i.name for i in train) & set(i.name for i in val) \
            or set(i.name for i in train) & set(i.name for i in test) \
            or set(i.name for i in val) & set(i.name for i in test):
        raise AssertionError("train/val/test splits overlap")
    return train, val, test


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, model: torch.nn.Module, cfg: Config, **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": cfg.to_dict()}
    payload.update(extra)                       # keep extras plain python types
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(Path(path), map_location=device)


# --------------------------------------------------------------------------
# logging / files
# --------------------------------------------------------------------------
def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# result summaries
# --------------------------------------------------------------------------
def summarize(makespans: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(makespans, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def format_summary_row(label: str, s: Dict[str, float]) -> str:
    return (f"{label:<22} n={s['n']:<3d} mean={s['mean']:8.2f}  median={s['median']:8.2f}  "
            f"std={s['std']:7.2f}  min={s['min']:7.1f}  max={s['max']:7.1f}")