"""
Step 1 of Phase 1: build a MANIFEST.

A manifest is a single CSV listing every image on disk with its subject id and
its integer class label. Everything downstream reads the manifest, never the
filesystem.

Why bother instead of globbing the folder in the training script?

  1. You verify your dataset ONCE, here, instead of hoping every script agrees.
  2. Label encoding is fixed and saved. AMI subject ids are NOT 0..99 -- subjects
     6, 15, 16, 17, 49, 50 and 60 were excluded, so ids have gaps. If two scripts
     each build their own label mapping, they can disagree, and your model will
     silently predict the wrong identities.
  3. Kaggle/Colab/laptop all produce the same manifest, so results are comparable.

Run:
    python -m src.manifest --dataset earvn
    python -m src.manifest --dataset ami
    python -m src.manifest --dataset all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.config import CFG
from src.paths import P

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dataset_root(name: str) -> Path:
    """Resolve the raw-data folder for this dataset on this machine."""
    spec = CFG.datasets[name]
    key = P.env  # 'local' | 'kaggle' | 'colab'
    if key not in spec:
        raise KeyError(f"config datasets.{name} has no '{key}' path")
    root = Path(spec[key])
    if not root.exists():
        raise FileNotFoundError(
            f"\nDataset '{name}' not found at:\n    {root}\n"
            f"Fix the path at datasets.{name}.{key} in config.yaml.\n"
            f"(Use forward slashes, and quote paths containing spaces.)"
        )
    return root


def extract_subject_id(path: Path, root: Path, id_source: str, pattern: str) -> str | None:
    """
    Pull the subject id out of a path.

    earvn: id_source='folder'   -> "001.ALI_HD/005 (1).jpg"  -> "001"
    ami:   id_source='filename' -> "001_back.jpg"             -> "001"

    Returns None if the pattern doesn't match, so the caller can report the
    file as unparseable rather than silently dropping it.
    """
    if id_source == "folder":
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            return None  # image sitting loose in root, no subject folder
        text = rel.parts[0]
    elif id_source == "filename":
        text = path.stem
    else:
        raise ValueError(f"id_source must be 'folder' or 'filename', got {id_source!r}")

    m = re.match(pattern, text)
    return m.group(1) if m else None


def build_manifest(name: str, verbose: bool = True) -> pd.DataFrame:
    spec = CFG.datasets[name]
    root = dataset_root(name)
    id_source = spec["id_source"]
    pattern = spec["id_regex"]

    all_files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in all_files if p.suffix.lower() in IMAGE_EXTS]
    skipped = [p for p in all_files if p.suffix.lower() not in IMAGE_EXTS]

    rows, unparseable = [], []
    for p in sorted(images):
        sid = extract_subject_id(p, root, id_source, pattern)
        if sid is None:
            unparseable.append(p)
            continue
        rows.append(
            {
                "dataset": name,
                "subject_id": sid.zfill(3),  # '1' and '001' must not become 2 people
                "relpath": p.relative_to(root).as_posix(),
            }
        )

    if not rows:
        raise RuntimeError(
            f"No images parsed for '{name}'. Found {len(images)} image files but "
            f"none matched id_regex {pattern!r} with id_source={id_source!r}. "
            f"Check the folder/file naming and fix config.yaml."
        )

    df = pd.DataFrame(rows)

    # Contiguous label encoding, saved so every script agrees forever.
    subjects = sorted(df["subject_id"].unique())
    label_map = {sid: i for i, sid in enumerate(subjects)}
    df["label"] = df["subject_id"].map(label_map)

    if verbose:
        counts = df.groupby("subject_id").size()
        print(f"\n=== manifest: {name} ===")
        print(f"  root            : {root}")
        print(f"  images indexed  : {len(df):,}")
        print(f"  subjects        : {len(subjects)}")
        print(f"  imgs/subject    : min {counts.min()}  max {counts.max()}  "
              f"mean {counts.mean():.1f}")
        print(f"  imbalance ratio : {counts.max() / counts.min():.2f}x")
        if skipped:
            exts = sorted({p.suffix.lower() or "(none)" for p in skipped})
            print(f"  NON-IMAGE files skipped: {len(skipped)} ({', '.join(exts)})")
        if unparseable:
            print(f"  !! UNPARSEABLE: {len(unparseable)} files, e.g. "
                  f"{unparseable[0].name} -- id_regex did not match")

        # AMI-specific: warn if the gap structure isn't what the source says.
        if name == "ami":
            ids = {int(s) for s in subjects}
            expected_missing = {6, 15, 16, 17, 49, 50, 60}
            actually_missing = set(range(min(ids), max(ids) + 1)) - ids
            print(f"  id range        : {min(ids)}..{max(ids)} "
                  f"({len(ids)} subjects)")
            if actually_missing != expected_missing:
                print(f"  NOTE: missing ids {sorted(actually_missing)} differ from the "
                      f"documented exclusions {sorted(expected_missing)}")

    # Save manifest + label map side by side.
    P.ensure()
    out_csv = P.splits / f"manifest_{name}.csv"
    out_map = P.splits / f"labelmap_{name}.json"
    df.to_csv(out_csv, index=False)
    out_map.write_text(json.dumps(label_map, indent=2))
    if verbose:
        print(f"  -> {out_csv}")
        print(f"  -> {out_map}")

    return df


def load_manifest(name: str) -> pd.DataFrame:
    path = P.splits / f"manifest_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.manifest --dataset {name}"
        )
    # subject_id must stay a string or '001' becomes 1
    return pd.read_csv(path, dtype={"subject_id": str})


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset manifest(s).")
    ap.add_argument("--dataset", default="all", choices=["ami", "earvn", "all"])
    args = ap.parse_args()

    names = ["ami", "earvn"] if args.dataset == "all" else [args.dataset]
    for n in names:
        try:
            build_manifest(n)
        except FileNotFoundError as e:
            print(f"\n[skip] {n}: {e}")


if __name__ == "__main__":
    main()
