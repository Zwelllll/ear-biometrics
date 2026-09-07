"""
Reproducibility.

Your whole "3 seeds, report mean +/- std" contribution depends on seeds actually
working. There are FIVE separate sources of randomness in a PyTorch training run
and missing any one of them makes your runs non-reproducible:

  1. Python's `random`
  2. NumPy
  3. PyTorch CPU
  4. PyTorch CUDA
  5. cuDNN's algorithm autotuner (non-deterministic by default!)

Call seed_everything(seed) ONCE at the top of every run, before building
anything (model, dataloaders, transforms).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed all RNGs.

    deterministic=True makes runs bit-for-bit repeatable but can be ~5-15%
    slower, because it disables cuDNN's autotuner. Keep it True -- for this
    project reproducibility matters far more than a few percent of speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Needed for full determinism in some ops. warn_only avoids a hard
        # crash if an op has no deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    Pass this to DataLoader(worker_init_fn=seed_worker).

    Each DataLoader worker is a separate process with its own RNG. Without this,
    your augmentations differ run-to-run even with seed_everything() set.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Pass this to DataLoader(generator=...) to fix shuffle order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
