"""
Step 2 of Phase 1: build the SPLITS.

Two protocols, because identification and verification need different things:

  IDENTIFICATION  (split = 'ident')
    All subjects appear in train/val/test. Only the IMAGES differ.
    Split is stratified PER SUBJECT (70/15/15 of each subject's own images),
    not 70/15/15 of the global pool -- with 107-300 images per subject, a
    global split could leave some subject with almost no test images, making
    its per-class accuracy pure noise.
    This matches what the paper appears to have done, so your numbers are
    comparable to theirs.

  VERIFICATION  (split = 'verif')
    A set of subjects is held out ENTIRELY -- they appear in no training data
    at all. Verification is then tested on people the model has never seen.
    Without this, EER measures memorization, not verification, and the number
    is meaningless. The paper never did this at all.
    Produces:
      verif_train : the remaining subjects (train a fresh model on these)
      verif_val   : val images of those same remaining subjects
      verif_gallery_probe : the held-out subjects, for building pairs

Run:
    python -m src.splits --dataset earvn
    python -m src.splits --dataset all
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CFG
from src.manifest import load_manifest
from src.paths import P

SPLIT_SEED = 12345  # splits are FIXED, independent of training seeds


def _split_counts(n: int, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    """
    Decide how many of a subject's n images go to train/val/test.

    Guarantees at least 1 image in val and test whenever n >= 3, so no subject
    is ever missing from a split. AMI has exactly 7 images/subject -> 5/1/1.
    """
    if n < 3:
        raise ValueError(f"subject has only {n} image(s); cannot make 3 splits")
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_val - n_test
    if n_train < 1:  # tiny-n fallback
        n_train, n_val, n_test = n - 2, 1, 1
    return n_train, n_val, n_test


def make_identification_split(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject stratified image-disjoint split. Subjects shared across splits."""
    frac = CFG.splits.identification
    rng = np.random.default_rng(SPLIT_SEED)
    out = []

    for sid, g in df.groupby("subject_id", sort=True):
        idx = np.array(g.index)
        rng.shuffle(idx)
        n_train, n_val, n_test = _split_counts(len(idx), frac.val, frac.test)
        for split, part in (
            ("train", idx[:n_train]),
            ("val", idx[n_train:n_train + n_val]),
            ("test", idx[n_train + n_val:]),
        ):
            sub = df.loc[part].copy()
            sub["split"] = split
            out.append(sub)

    res = pd.concat(out).sort_values(["subject_id", "split", "relpath"])
    res["protocol"] = "image_disjoint"
    return res.reset_index(drop=True)


def make_verification_split(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Subject-disjoint split: some identities held out from all training."""
    key = f"heldout_identities_{dataset}"
    n_heldout = CFG.splits.verification[key]

    subjects = sorted(df["subject_id"].unique())
    if n_heldout >= len(subjects) - 5:
        raise ValueError(
            f"holding out {n_heldout} of {len(subjects)} subjects leaves too few "
            f"for training; lower splits.verification.{key} in config.yaml"
        )

    rng = np.random.default_rng(SPLIT_SEED + 1)
    perm = rng.permutation(subjects)
    heldout = set(perm[:n_heldout])
    trainable = set(perm[n_heldout:])

    frac = CFG.splits.identification
    out = []

    # Held-out identities: never trained on. Used to build genuine/impostor pairs.
    sub = df[df["subject_id"].isin(heldout)].copy()
    sub["split"] = "heldout_probe"
    out.append(sub)

    # Remaining identities: normal train/val for fitting the embedding model.
    for sid, g in df[df["subject_id"].isin(trainable)].groupby("subject_id", sort=True):
        idx = np.array(g.index)
        rng.shuffle(idx)
        n_train, n_val, _ = _split_counts(len(idx), frac.val, frac.test)
        for split, part in (
            ("train", idx[:n_train]),
            ("val", idx[n_train:n_train + n_val]),
        ):
            s = df.loc[part].copy()
            s["split"] = split
            out.append(s)

    res = pd.concat(out).sort_values(["split", "subject_id", "relpath"])
    res["protocol"] = "subject_disjoint"

    # Verification models train on FEWER classes -- relabel contiguously, or
    # nn.CrossEntropyLoss will complain about label indices >= num_classes.
    tr_subjects = sorted(trainable)
    remap = {sid: i for i, sid in enumerate(tr_subjects)}
    res["verif_label"] = res["subject_id"].map(remap)  # NaN for held-out, correct

    return res.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The checks. These are the whole point of this file.
# ---------------------------------------------------------------------------

def verify_identification(split_df: pd.DataFrame, manifest: pd.DataFrame) -> None:
    sets = {s: set(g["relpath"]) for s, g in split_df.groupby("split")}

    # 1. No image in two splits. This is THE leakage check.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        assert not overlap, (
            f"LEAKAGE: {len(overlap)} image(s) in both {a} and {b}, "
            f"e.g. {sorted(overlap)[:3]}"
        )

    # 2. Every image used exactly once.
    total = sum(len(v) for v in sets.values())
    assert total == len(manifest), f"{total} split images vs {len(manifest)} in manifest"
    assert len(split_df) == len(split_df.drop_duplicates("relpath")), "duplicate rows"

    # 3. Every subject present in all three splits (else per-class metrics break).
    all_subj = set(manifest["subject_id"])
    for s in ("train", "val", "test"):
        present = set(split_df[split_df["split"] == s]["subject_id"])
        missing = all_subj - present
        assert not missing, f"{len(missing)} subject(s) missing from {s}: {sorted(missing)[:5]}"

    print("  [ OK ] no image leakage across train/val/test")
    print("  [ OK ] every image used exactly once")
    print("  [ OK ] all subjects present in all three splits")


def verify_verification(split_df: pd.DataFrame) -> None:
    heldout = set(split_df[split_df["split"] == "heldout_probe"]["subject_id"])
    trained = set(split_df[split_df["split"].isin(["train", "val"])]["subject_id"])

    # THE check: held-out identities must never appear in training.
    bleed = heldout & trained
    assert not bleed, (
        f"SUBJECT LEAKAGE: {len(bleed)} held-out identity/identities appear in "
        f"training: {sorted(bleed)[:5]} -- EER would be meaningless"
    )
    assert heldout, "no held-out identities produced"

    # Held-out subjects need >=2 images each or you cannot form a genuine pair.
    counts = split_df[split_df["split"] == "heldout_probe"].groupby("subject_id").size()
    too_few = counts[counts < 2]
    assert too_few.empty, f"held-out subjects with <2 images: {list(too_few.index)}"

    print(f"  [ OK ] {len(heldout)} identities held out, zero overlap with training")
    print(f"  [ OK ] all held-out subjects have >=2 images (genuine pairs possible)")


def build(dataset: str) -> None:
    manifest = load_manifest(dataset)
    print(f"\n=== splits: {dataset} ({len(manifest):,} images, "
          f"{manifest['subject_id'].nunique()} subjects) ===")

    ident = make_identification_split(manifest)
    print("\n-- identification (image-disjoint, subjects shared) --")
    print(ident.groupby("split").size().to_string())
    verify_identification(ident, manifest)
    ident_path = P.splits / f"{dataset}_ident.csv"
    ident.to_csv(ident_path, index=False)
    print(f"  -> {ident_path}")

    verif = make_verification_split(manifest, dataset)
    print("\n-- verification (subject-disjoint) --")
    print(verif.groupby("split").size().to_string())
    n_classes = int(verif["verif_label"].max()) + 1
    print(f"  trainable classes for verification model: {n_classes}")
    verify_verification(verif)
    verif_path = P.splits / f"{dataset}_verif.csv"
    verif.to_csv(verif_path, index=False)
    print(f"  -> {verif_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train/val/test splits.")
    ap.add_argument("--dataset", default="all", choices=["ami", "earvn", "all"])
    args = ap.parse_args()

    names = ["ami", "earvn"] if args.dataset == "all" else [args.dataset]
    for n in names:
        try:
            build(n)
        except FileNotFoundError as e:
            print(f"\n[skip] {n}: {e}")

    print("\nCommit the splits/ folder to git. Those CSVs are what make your "
          "results reproducible -- and they're what the paper never published.")


if __name__ == "__main__":
    main()
