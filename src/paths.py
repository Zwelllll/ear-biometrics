"""
Environment-aware paths.

The same code needs to run in three places:
  - your laptop (debugging on tiny subsets)
  - Kaggle (primary GPU, 30 free hours/week on a P100)
  - Colab (overflow GPU)

Each has different filesystem layouts, and hardcoding paths is the #1 cause of
"it worked yesterday" bugs. Import from here instead of writing literal paths.

Usage:
    from src.paths import P
    print(P.env)              # 'local' | 'kaggle' | 'colab'
    print(P.results_csv)      # correct location for this environment
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def detect_env() -> str:
    """Figure out which machine we're on."""
    if os.path.isdir("/kaggle"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or os.path.isdir("/content"):
        return "colab"
    return "local"


@dataclass(frozen=True)
class Paths:
    env: str
    root: Path          # project root (code lives here)
    data: Path          # raw + cached datasets
    splits: Path        # split CSVs -- these are COMMITTED to git
    results: Path       # results.csv and figures
    checkpoints: Path   # model weights (NEVER commit these)

    @property
    def results_csv(self) -> Path:
        return self.results / "results.csv"

    def ensure(self) -> "Paths":
        """Create any directories that don't exist yet."""
        for d in (self.data, self.splits, self.results, self.checkpoints):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        return "\n".join(
            [
                f"env         : {self.env}",
                f"root        : {self.root}",
                f"data        : {self.data}",
                f"splits      : {self.splits}",
                f"results     : {self.results}",
                f"checkpoints : {self.checkpoints}",
            ]
        )


def build_paths() -> Paths:
    env = detect_env()

    if env == "kaggle":
        # /kaggle/working persists for the session and is downloadable at the end.
        # /kaggle/input is READ-ONLY (that's where attached datasets appear).
        root = Path("/kaggle/working/ear-biometrics")
        return Paths(
            env=env,
            root=root,
            data=Path("/kaggle/input"),
            splits=root / "splits",
            results=root / "results",
            checkpoints=root / "checkpoints",
        )

    if env == "colab":
        # Write to mounted Drive so nothing is lost when the session dies.
        # Mount first:  from google.colab import drive; drive.mount('/content/drive')
        drive = Path("/content/drive/MyDrive/ear-biometrics")
        if Path("/content/drive/MyDrive").exists():
            root = drive
        else:
            print("[paths] WARNING: Drive not mounted -- writing to session disk, "
                  "which is DELETED on disconnect. Mount Drive first.")
            root = Path("/content/ear-biometrics")
        return Paths(
            env=env,
            root=root,
            data=root / "data",
            splits=root / "splits",
            results=root / "results",
            checkpoints=root / "checkpoints",
        )

    # local
    root = Path(__file__).resolve().parents[1]
    return Paths(
        env=env,
        root=root,
        data=root / "data",
        splits=root / "splits",
        results=root / "results",
        checkpoints=root / "checkpoints",
    )


P = build_paths()


if __name__ == "__main__":
    print(P.describe())
