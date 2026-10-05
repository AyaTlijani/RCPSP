import random
from pathlib import Path
from typing import List, Tuple

from parser import RCPSPParser, Project


class RCPSPLoader:

    def __init__(
        self,
        folder: str,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 1,
    ):
        if abs(sum(split) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.")

        self.folder = Path(folder)

        files = sorted(self.folder.glob("*.sm"))

        if not files:
            raise FileNotFoundError(
                f"No .sm files found in {self.folder}"
            )

        random.Random(seed).shuffle(files)

        n = len(files)
        n_train = int(n * split[0])
        n_val = int(n * split[1])

        self.train_files = files[:n_train]
        self.val_files = files[n_train:n_train + n_val]
        self.test_files = files[n_train + n_val:]

        self.train_set: List[Project] = [
            RCPSPParser.parse_file(f)
            for f in self.train_files
        ]

        self.val_set: List[Project] = [
            RCPSPParser.parse_file(f)
            for f in self.val_files
        ]

        self.test_set: List[Project] = [
            RCPSPParser.parse_file(f)
            for f in self.test_files
        ]

        self.current_index = 0

        print(
            f"[RCPSPLoader] {n} instances found -> "
            f"train={len(self.train_set)} "
            f"val={len(self.val_set)} "
            f"test={len(self.test_set)}"
        )

    @property
    def num_train_instances(self) -> int:
        return len(self.train_set)

    def next_instance(self) -> Project:
        project = self.train_set[self.current_index]

        self.current_index += 1

        if self.current_index >= len(self.train_set):
            self.current_index = 0

        return project

    @staticmethod
    def resource_ids(project: Project) -> List[int]:
        return [resource.id for resource in project.resources]