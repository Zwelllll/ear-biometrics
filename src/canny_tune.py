"""
Canny threshold tuner.

The paper's thresholds (100/200) produce a COMPLETELY BLACK edge map on ear
images. Ear photos are smooth skin gradients; there are almost no hard
intensity steps for a high-threshold Canny to find. The paper never shows an
edge map, so this was never visible in it -- which plausibly explains why its
Canny columns are the worst score in nearly every row of both tables.

This script renders the same ears under several threshold settings, with the
edge density (% of pixels that are edges) printed for each, so you can pick
by eye instead of guessing.

Run:
    python -m src.canny_tune --dataset ami
    python -m src.canny_tune --dataset earvn

Then set the winner in config.yaml under preprocessing.canny and re-run:
    python -m src.preprocess --dataset ami --overwrite
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.config import CFG
from src.manifest import dataset_root, load_manifest
from src.paths import P
from src.preprocess import apply_zoom, auto_thresholds

# (label, low, high) -- 'auto' uses per-image median-based thresholds.
CANDIDATES = [
    ("paper 100/200", 100, 200),
    ("60/120", 60, 120),
    ("30/90", 30, 90),
    ("20/60", 20, 60),
    ("10/40", 10, 40),
    ("5/20", 5, 20),
    ("auto", None, None),
]


def canny_with(img_bgr: np.ndarray, low, high, ksize: int = 5) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize | 1, ksize | 1), 0)
    if low is None:  # auto
        low, high = auto_thresholds(blurred)
    edges = cv2.Canny(blurred, int(low), int(high))
    density = float((edges > 0).mean() * 100.0)
    return edges, density


def build(dataset: str, n_rows: int = 4, seed: int = 7) -> None:
    manifest = load_manifest(dataset)
    root = dataset_root(dataset)
    size = int(CFG.data.img_size)

    rng = np.random.default_rng(seed)
    picks = manifest.iloc[rng.choice(len(manifest), size=min(n_rows, len(manifest)),
                                     replace=False)]

    pad, header = 8, 30
    cols = ["zoomed"] + [c[0] for c in CANDIDATES]
    W = len(cols) * (size + pad) + pad
    H = header + len(picks) * (size + pad) + pad
    sheet = np.full((H, W, 3), 25, np.uint8)

    for ci, name in enumerate(cols):
        cv2.putText(sheet, name, (pad + ci * (size + pad), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

    densities: dict[str, list[float]] = {c[0]: [] for c in CANDIDATES}

    for ri, (_, row) in enumerate(picks.iterrows()):
        img = cv2.imread(str(root / row["relpath"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # Same order as the real pipeline: zoom -> resize -> canny.
        zoomed = cv2.resize(apply_zoom(img, dataset), (size, size),
                            interpolation=cv2.INTER_AREA)

        y = header + pad + ri * (size + pad)
        sheet[y:y + size, pad:pad + size] = zoomed

        for ci, (label, lo, hi) in enumerate(CANDIDATES, start=1):
            edges, dens = canny_with(zoomed, lo, hi)
            densities[label].append(dens)
            x = pad + ci * (size + pad)
            sheet[y:y + size, x:x + size] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(sheet, f"{dens:.1f}%", (x + 4, y + size - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    out = P.results / f"canny_tune_{dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)

    print(f"\n=== canny thresholds: {dataset} ===")
    print(f"{'setting':<16} {'mean edge density':>18}")
    for label, vals in densities.items():
        if vals:
            print(f"{label:<16} {np.mean(vals):>17.2f}%")
    th = CFG.preprocessing.canny["thresholds"][dataset]
    print(f"\n  currently configured for {dataset}: "
          f"{th['low']}/{th['high']}  (mode: {CFG.preprocessing.canny['mode']})")
    print(f"  -> {out}")
    print("""
  How to read this:
    < 1%      almost blank -- the arm carries no usable signal
    2 - 8%    usually the useful range: ear outline PLUS inner ridges
    > 15%     over-detecting -- picking up skin texture, hair and sensor noise

  Pick the setting where you can still recognise the ear's inner structure
  (helix, antihelix, concha), not just its silhouette. A silhouette alone
  carries far less identity information than the inner ridges.

  Then edit config.yaml:
      preprocessing:
        canny:
          thresholds:
            {ds}: {low: 30, high: 90}
  and re-run:
      python -m src.preprocess --dataset {ds} --overwrite
""".replace("{ds}", dataset))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune Canny thresholds visually.")
    ap.add_argument("--dataset", default="all", choices=["ami", "earvn", "all"])
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()

    names = ["ami", "earvn"] if args.dataset == "all" else [args.dataset]
    for n in names:
        try:
            build(n, n_rows=args.rows)
        except FileNotFoundError as e:
            print(f"\n[skip] {n}: {e}")


if __name__ == "__main__":
    main()
