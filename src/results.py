"""
Results logging.

Design rule: ONE ROW PER RUN, appended to disk the moment the run finishes.

Why this matters on free-tier GPUs: sessions disconnect without warning. If your
numbers live only in notebook cell outputs, a disconnect costs you every run you
did that day. Appending to a CSV means a disconnect costs you at most one run.

The schema covers every phase up front so you never have to migrate it:
  - Phases 4/5 (identification sweeps): test_top1, test_top5
  - Phase 6 (cross-dataset):            eval_dataset != train_dataset
  - Phase 7 (verification):             eer, tar_at_far_1e2, tar_at_far_1e3
  - Phase 8 (efficiency):               params_m, gflops, latency_ms, epoch_time_s
"""

from __future__ import annotations

import csv
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import P

# Order matters -- this is the CSV column order.
FIELDS: list[str] = [
    "run_id",
    "timestamp_utc",
    # --- what was run ---
    "task",             # 'identification' | 'verification'
    "train_dataset",    # 'ami' | 'earvn'
    "eval_dataset",     # same as train_dataset, EXCEPT in Phase 6 cross-dataset
    "arm",              # 'raw' | 'zoom' | 'zoom+canny' | 'zoom+aug' | 'zoom+canny+aug'
    "model",            # 'resnet50' | 'mobilenet_v2' | 'efficientnet_b0'
    "seed",
    "split_protocol",   # 'image_disjoint' | 'subject_disjoint'
    # --- hyperparameters ---
    "epochs",
    "batch_size",
    "lr",
    "img_size",
    # --- identification metrics ---
    "train_acc_final",
    "best_val_acc",
    "test_top1",
    "test_top5",
    # --- verification metrics (Phase 7) ---
    "eer",
    "tar_at_far_1e2",
    "tar_at_far_1e3",
    "auc",
    # --- efficiency (Phase 8) ---
    "params_m",
    "gflops",
    "latency_ms",
    "epoch_time_s",
    "train_time_s",
    # --- provenance ---
    "device",
    "env",
    "notes",
]


def _ensure_header(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def make_run_id(**parts: Any) -> str:
    """
    Deterministic, human-readable run id, e.g.
        ami__resnet50__zoom+aug__s0
    Use it for checkpoint filenames too, so a row in results.csv tells you
    exactly which checkpoint file it came from.
    """
    keys = ("train_dataset", "model", "arm", "seed")
    bits = [str(parts[k]) for k in keys if parts.get(k) is not None]
    if "seed" in parts and parts["seed"] is not None:
        bits[-1] = f"s{parts['seed']}"
    return "__".join(bits)


def log_run(**row: Any) -> str:
    """
    Append one run to results.csv. Unknown keys raise, missing keys become ''.

    Raising on unknown keys is deliberate: a silent typo like `test_topl`
    (lowercase L) would otherwise vanish and you'd lose the number.
    """
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise KeyError(f"Unknown result field(s): {sorted(unknown)}. "
                       f"Add them to FIELDS in src/results.py if intentional.")

    row.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row.setdefault("env", P.env)
    row.setdefault("device", _device_name())
    row.setdefault("eval_dataset", row.get("train_dataset", ""))
    if not row.get("run_id"):
        row["run_id"] = make_run_id(**row)

    path = P.results_csv
    _ensure_header(path)
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS, restval="").writerow(row)

    print(f"[results] logged {row['run_id']} -> {path}")
    return row["run_id"]


def _device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return f"cpu ({platform.processor() or platform.machine()})"


def load_results():
    """Load results.csv as a pandas DataFrame (for Phase 9 analysis)."""
    import pandas as pd
    if not P.results_csv.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(P.results_csv)


def summarise(group_cols=("train_dataset", "model", "arm"), metric="test_top1"):
    """
    mean +/- std across seeds -- this is the table that goes in your report.
    Returns a DataFrame with n_seeds, mean, std.
    """
    df = load_results()
    df = df[df[metric].notna()]
    if df.empty:
        return df
    out = (
        df.groupby(list(group_cols))[metric]
        .agg(n_seeds="count", mean="mean", std="std")
        .reset_index()
    )
    return out.sort_values(list(group_cols))
