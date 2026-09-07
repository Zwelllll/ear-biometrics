"""
Config loading.

Reads config.yaml once and gives you dot-access:

    from src.config import CFG
    CFG.train.lr           # 0.0003
    CFG.experiments.models # ['resnet50', ...]

Dot-access matters for a practical reason: a typo in `CFG.train.lr` raises an
AttributeError immediately, whereas a typo in `cfg["train"]["lr"]` written as
`cfg["train"]["Lr"]` raises a KeyError only when that line runs -- possibly
20 minutes into a training job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.paths import P


class Node(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(
                f"No config key '{name}'. Available here: {sorted(self.keys())}"
            ) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Node({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def find_config() -> Path:
    """Look for config.yaml next to the project root, then cwd."""
    candidates = [
        P.root / "config.yaml",
        Path(__file__).resolve().parents[1] / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "config.yaml not found. Looked in:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def load_config(path: Path | None = None) -> Node:
    path = path or find_config()
    with path.open() as f:
        raw = yaml.safe_load(f)
    cfg = _wrap(raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Node) -> None:
    """Catch config mistakes now rather than 20 minutes into a run."""
    s = cfg.splits.identification
    total = s.train + s.val + s.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"identification split fractions sum to {total}, not 1.0")

    if cfg.data.img_size <= 0:
        raise ValueError("data.img_size must be positive")

    valid_arms = {"raw", "zoom", "zoom+canny", "zoom+aug", "zoom+canny+aug"}
    bad = set(cfg.experiments.arms) - valid_arms
    if bad:
        raise ValueError(f"Unknown arm(s) {sorted(bad)}; valid: {sorted(valid_arms)}")

    if cfg.verification.head not in {"cosine", "arcface"}:
        raise ValueError("verification.head must be 'cosine' or 'arcface'")

    if cfg.preprocessing.augment.horizontal_flip_p > 0:
        print("[config] WARNING: horizontal_flip_p > 0. AMI mixes left and right "
              "ears within each identity -- flipping can collide them. Make sure "
              "you canonicalize orientation instead.")


CFG = load_config()


if __name__ == "__main__":
    print(f"config loaded from: {find_config()}")
    print(f"  models  : {CFG.experiments.models}")
    print(f"  arms    : {CFG.experiments.arms}")
    print(f"  seeds   : {CFG.seed.seeds}")
    print(f"  lr      : {CFG.train.lr}, epochs: {CFG.train.epochs}, bs: {CFG.train.batch_size}")
    n = (len(CFG.experiments.models) * len(CFG.experiments.arms))
    print(f"  runs per dataset per seed: {n}")
