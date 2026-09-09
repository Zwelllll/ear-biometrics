"""
Phase 3, part 1: DATA LOADING.

Reads the cached images from Phase 2 and hands batches to the model.

The single most important rule in this file:

    AUGMENTATION IS APPLIED TO THE TRAIN SPLIT ONLY.

Augmenting val or test invalidates your numbers -- you'd be measuring accuracy
on randomly distorted images, which is not the accuracy anyone cares about, and
it makes runs non-comparable because the distortions differ every time.

The arms map onto (cached variant + augment?) like this:

    arm                cached variant     augment
    raw                raw                no
    zoom               zoom               no
    zoom+canny         zoom_canny         no
    zoom+aug           zoom               YES (train only)
    zoom+canny+aug     zoom_canny         YES (train only)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import CFG
from src.paths import P
from src.preprocess import cache_dir, flat_name
from src.seed import make_generator, seed_worker

# arm -> (cached variant folder, whether to augment the train split)
ARM_SPEC: dict[str, tuple[str, bool]] = {
    "raw": ("raw", False),
    "zoom": ("zoom", False),
    "zoom+canny": ("zoom_canny", False),
    "zoom+aug": ("zoom", True),
    "zoom+canny+aug": ("zoom_canny", True),
}


def cached_path(dataset: str, variant: str, relpath: str) -> Path:
    ext = ".png" if variant == "zoom_canny" else ".jpg"
    return cache_dir(dataset, variant) / (flat_name(relpath).rsplit(".", 1)[0] + ext)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(train: bool, augment: bool) -> transforms.Compose:
    """
    Eval transform is deterministic: ToTensor + ImageNet normalization.
    Train transform adds augmentation only when augment=True AND train=True.
    """
    norm = transforms.Normalize(mean=CFG.data.norm_mean, std=CFG.data.norm_std)

    if not (train and augment):
        return transforms.Compose([transforms.ToTensor(), norm])

    a = CFG.preprocessing.augment
    ops: list = []

    # Geometric. Note horizontal flip is deliberately absent unless enabled in
    # config: AMI has 6 right-ear + 1 left-ear image per subject, so flipping
    # indiscriminately collides the two orientations within one identity.
    if float(a.horizontal_flip_p) > 0:
        ops.append(transforms.RandomHorizontalFlip(p=float(a.horizontal_flip_p)))
    ops.append(transforms.RandomRotation(degrees=float(a.rotation_degrees)))
    ops.append(transforms.RandomPerspective(
        distortion_scale=float(a.perspective_distortion),
        p=float(a.perspective_p),
    ))

    # Photometric. Applied to Canny arms too, where it does almost nothing --
    # a binary edge map has no colour to jitter. Worth a sentence in the report:
    # the paper's augmentation set is partly inert on its own contour arm.
    cj = a.color_jitter
    ops.append(transforms.ColorJitter(
        brightness=float(cj["brightness"]),
        contrast=float(cj["contrast"]),
        saturation=float(cj["saturation"]),
        hue=float(cj["hue"]),
    ))
    if float(a.grayscale_p) > 0:
        ops.append(transforms.RandomGrayscale(p=float(a.grayscale_p)))

    ops += [transforms.ToTensor(), norm]
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EarDataset(Dataset):
    """Reads one cached variant, returns (tensor, label)."""

    def __init__(self, rows: pd.DataFrame, dataset: str, variant: str,
                 transform, label_col: str = "label"):
        self.dataset = dataset
        self.variant = variant
        self.transform = transform
        self.paths = [cached_path(dataset, variant, rp) for rp in rows["relpath"]]
        self.labels = rows[label_col].astype(int).tolist()

        # Fail loudly HERE rather than mid-epoch. A missing cache file 40
        # minutes into a Kaggle run wastes the whole session.
        missing = [p for p in self.paths[:200] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of the first 200 cached files are missing for "
                f"variant '{variant}', e.g.\n  {missing[0]}\n"
                f"Run: python -m src.preprocess --dataset {dataset}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def load_split(dataset: str, protocol: str = "ident") -> pd.DataFrame:
    path = P.splits / f"{dataset}_{protocol}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.splits --dataset {dataset}"
        )
    return pd.read_csv(path, dtype={"subject_id": str})


def build_loaders(dataset: str, arm: str, seed: int, protocol: str = "ident",
                  batch_size: int | None = None) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_loader, test_loader, num_classes).

    For protocol='verif' there is no test split (the held-out identities are
    used for pair-based verification in Phase 7, not for classification), so
    the test loader comes back as None.
    """
    if arm not in ARM_SPEC:
        raise ValueError(f"unknown arm {arm!r}; valid: {sorted(ARM_SPEC)}")
    variant, augment = ARM_SPEC[arm]

    df = load_split(dataset, protocol)
    label_col = "verif_label" if protocol == "verif" else "label"
    bs = int(batch_size or CFG.train.batch_size)

    def make(split: str, is_train: bool):
        rows = df[df["split"] == split]
        if rows.empty:
            return None
        rows = rows[rows[label_col].notna()]
        ds = EarDataset(rows, dataset, variant,
                        build_transforms(train=is_train, augment=augment),
                        label_col=label_col)
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=is_train,
            num_workers=int(CFG.data.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=seed_worker,      # reproducible augmentation
            generator=make_generator(seed),  # reproducible shuffle order
            persistent_workers=int(CFG.data.num_workers) > 0,
        )

    train_loader = make("train", True)
    val_loader = make("val", False)
    test_loader = make("test", False)

    num_classes = int(df[label_col].max()) + 1

    return train_loader, val_loader, test_loader, num_classes
