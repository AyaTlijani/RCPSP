"""
psplib_loader.py
================
Loads PSPLIB single-mode RCPSP instances (J30 / J60 / J90 ... `.sm` files)
into a plain `RCPSPInstance` object.

Responsibilities (and ONLY these):
  * parse a `.sm` file;
  * keep the dummy source (index 0), the real activities (1..N-2) and the
    dummy sink (index N-1);
  * validate the instance (indices, single mode, DAG, demand <= capacity).

It deliberately computes NO slack / LF / GRPW / dynamic resources: those live in
`domain_features.py` / `rcpsp_environment.py`.

The parser is self-contained (it does not need the `psplib` package) so it
cannot silently depend on a library version.  File layout it relies on:

    jobs (incl. supersource/sink ):  32
      - renewable                 :  4   R
    PRECEDENCE RELATIONS:
    jobnr.    #modes  #successors   successors
       1        1          3           2   3   4
    REQUESTS/DURATIONS:
    jobnr. mode duration  R 1  R 2  R 3  R 4
    ------------------------------------------
      1      1     0       0    0    0    0
    RESOURCEAVAILABILITY:
      R 1  R 2  R 3  R 4
       12   13    4   12
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np


@dataclass(eq=False)
class RCPSPInstance:
    """One RCPSP instance.  Index 0 = dummy source, N-1 = dummy sink."""
    name: str
    num_activities: int                 # N, INCLUDING the two dummy activities
    num_resources: int                  # K
    durations: np.ndarray               # (N,)   int64
    resource_demands: np.ndarray        # (N, K) int64
    resource_capacities: np.ndarray     # (K,)   int64
    predecessors: List[List[int]]       # predecessors[i] = sorted list of activity indices
    successors: List[List[int]]         # successors[i]   = sorted list of activity indices

    @property
    def source(self) -> int:
        return 0

    @property
    def sink(self) -> int:
        return self.num_activities - 1

    @property
    def num_real(self) -> int:
        """Number of real (non-dummy) activities, e.g. 30 for J30."""
        return self.num_activities - 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def natural_key(text: str):
    """Sort key so that j301_2 < j301_10 < j302_1 (deterministic instance order)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def topological_order(successors: Sequence[Sequence[int]]) -> List[int]:
    """Kahn's algorithm.  Raises ValueError if the precedence graph has a cycle."""
    n = len(successors)
    indeg = [0] * n
    for i in range(n):
        for j in successors[i]:
            indeg[j] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while stack:
        i = stack.pop()
        order.append(i)
        for j in successors[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    if len(order) != n:
        raise ValueError("precedence graph contains a cycle")
    return order


def _int_row(line: str):
    """Return the list of ints in `line`, or None if the line has any non-integer token."""
    toks = line.split()
    if not toks:
        return None
    try:
        return [int(t) for t in toks]
    except ValueError:
        return None


def _find_line(lines: List[str], marker: str, path: Path) -> int:
    """Index of the first line containing `marker`, ignoring case and all whitespace
    (so 'RESOURCEAVAILABILITY' also matches 'RESOURCE AVAILABILITY')."""
    target = "".join(marker.upper().split())
    for i, line in enumerate(lines):
        if target in "".join(line.upper().split()):
            return i
    raise ValueError(f"{path}: section '{marker}' not found")


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def load_instance(path: Union[str, Path]) -> RCPSPInstance:
    """Parse one PSPLIB `.sm` file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    m_jobs = re.search(r"jobs\s*\(incl\.[^)]*\)\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    m_res = re.search(r"-\s*renewable\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    if m_jobs is None or m_res is None:
        raise ValueError(f"{path}: could not read #jobs / #renewable resources from header")
    n = int(m_jobs.group(1))
    k = int(m_res.group(1))

    # ---- precedence relations (token stream; robust to wrapped lines) -----
    start = _find_line(lines, "PRECEDENCE RELATIONS", path) + 1
    ints: List[int] = []
    for line in lines[start:]:
        if line.lstrip().startswith("*"):
            break
        row = _int_row(line)
        if row is not None:           # skips the textual column-header line
            ints.extend(row)
    successors: List[List[int]] = [[] for _ in range(n)]
    pos = 0
    for expected_job in range(1, n + 1):
        if pos + 3 > len(ints):
            raise ValueError(f"{path}: precedence section is truncated")
        job, modes, n_succ = ints[pos], ints[pos + 1], ints[pos + 2]
        if job != expected_job:
            raise ValueError(f"{path}: expected job {expected_job} in precedence section, got {job}")
        if modes != 1:
            raise ValueError(f"{path}: job {job} has {modes} modes; only single-mode RCPSP is supported")
        succ = ints[pos + 3: pos + 3 + n_succ]
        if len(succ) != n_succ:
            raise ValueError(f"{path}: truncated successor list for job {job}")
        successors[job - 1] = sorted(s - 1 for s in succ)   # 1-based -> 0-based
        pos += 3 + n_succ

    # ---- requests / durations --------------------------------------------
    start = _find_line(lines, "REQUESTS/DURATIONS", path) + 1
    durations = np.zeros(n, dtype=np.int64)
    demands = np.zeros((n, k), dtype=np.int64)
    seen = 0
    req_end = start
    for off, line in enumerate(lines[start:]):
        if seen == n:
            break
        if line.lstrip().startswith("*") and seen > 0:
            break
        row = _int_row(line)
        if row is None:               # header / dashed separator
            continue
        if len(row) < 3 + k:
            raise ValueError(f"{path}: malformed request row: {line!r}")
        job, mode, dur = row[0], row[1], row[2]
        if job != seen + 1:
            raise ValueError(f"{path}: expected job {seen + 1} in requests section, got {job}")
        if mode != 1:
            raise ValueError(f"{path}: job {job} mode {mode}; only single-mode is supported")
        durations[job - 1] = dur
        demands[job - 1, :] = row[3: 3 + k]
        seen += 1
        req_end = start + off + 1          # first line after the last request row
    if seen != n:
        raise ValueError(f"{path}: found {seen} request rows, expected {n}")

    # ---- resource availability -------------------------------------------
    # Preferred: locate the section header.  Fallback (header named differently / missing):
    # the capacities are the first row of exactly K integers after the request rows.
    try:
        start = _find_line(lines, "RESOURCEAVAILABILITY", path) + 1
    except ValueError:
        start = req_end
    capacities = None
    for line in lines[start:]:
        row = _int_row(line)
        if row is not None and len(row) == k:
            capacities = np.array(row, dtype=np.int64)
            break
    if capacities is None:
        tail = " | ".join(l.strip() for l in lines[-6:] if l.strip())
        raise ValueError(f"{path}: resource capacities not found (expected a row of {k} integers "
                         f"after the requests section). Last lines of file: {tail}")

    # ---- predecessors, validation ----------------------------------------
    predecessors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in successors[i]:
            if not (0 <= j < n):
                raise ValueError(f"{path}: successor index {j + 1} out of range")
            predecessors[j].append(i)
    for j in range(n):
        predecessors[j].sort()

    if predecessors[0]:
        raise ValueError(f"{path}: dummy source must have no predecessors")
    if successors[n - 1]:
        raise ValueError(f"{path}: dummy sink must have no successors")
    if durations[0] != 0 or durations[n - 1] != 0:
        raise ValueError(f"{path}: dummy activities must have duration 0")
    topological_order(successors)                       # raises on cycles
    if np.any(demands > capacities[None, :]):
        raise ValueError(f"{path}: some activity demands more than the resource capacity (infeasible)")

    return RCPSPInstance(
        name=path.stem,
        num_activities=n,
        num_resources=k,
        durations=durations,
        resource_demands=demands,
        resource_capacities=capacities,
        predecessors=predecessors,
        successors=successors,
    )


def list_instance_files(size: str, data_root: Union[str, Path]) -> List[Path]:
    """
    All instance files of one problem size ('j30' | 'j60' | 'j90'), in a deterministic
    (natural) order.  Only files named like  j301_1.sm  are kept, so auxiliary files
    (e.g. j30hrs.sm / j30opt.sm / param files) are ignored.
    """
    size = size.lower()
    folder = Path(data_root) / size
    if not folder.is_dir():
        raise FileNotFoundError(f"data folder not found: {folder}")
    pattern = re.compile(rf"^{re.escape(size)}\d+_\d+\.sm$", flags=re.IGNORECASE)
    files = [p for p in folder.rglob("*") if p.is_file() and pattern.match(p.name)]
    files.sort(key=lambda p: natural_key(p.name))
    if not files:
        raise FileNotFoundError(f"no '{size}<set>_<inst>.sm' files found under {folder}")
    return files


def load_dataset(size: str, data_root: Union[str, Path], expected_count: int = 480) -> List[RCPSPInstance]:
    """Load every instance of a problem size, sorted deterministically by file name."""
    files = list_instance_files(size, data_root)
    if expected_count is not None and len(files) != expected_count:
        print(f"[psplib_loader] WARNING: found {len(files)} {size} instances, expected {expected_count}")
    instances = [load_instance(p) for p in files]
    expected_n = {"j30": 32, "j60": 62, "j90": 92, "j120": 122}.get(size.lower())
    if expected_n is not None:
        bad = [inst.name for inst in instances if inst.num_activities != expected_n]
        if bad:
            raise ValueError(f"{len(bad)} {size} instances do not have {expected_n} activities, e.g. {bad[:3]}")
    return instances