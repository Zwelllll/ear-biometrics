"""
Phase 2: build the PREPROCESSING CACHE.

Five experiment arms, but only THREE need files on disk:

    raw              resize to 224                        -> cached
    zoom             crop to the ear, then resize         -> cached
    zoom+canny       crop, Canny edges, resize            -> cached
    zoom+aug         reuses the 'zoom' cache       + on-the-fly augmentation
    zoom+canny+aug   reuses the 'zoom+canny' cache + on-the-fly augmentation

Why cache the deterministic ones? On free-tier Kaggle/Colab you get ~2 CPU
cores. Decoding a 492x702 JPEG and running Canny on it every epoch starves the
GPU -- the dataloader becomes the bottleneck and the GPU sits idle. Doing it
once and storing 224x224 files makes every subsequent epoch a cheap read.

Why NOT cache augmentation? Because pre-rendering N augmented copies gives you
a bigger dataset with LESS diversity (the model sees the same N variants every
epoch instead of a fresh one). It is also how leakage creeps in: augment before
splitting and copies of the same photo land in train AND test. On-the-fly
augmentation of the train split only avoids both problems.

Run:
    python -m src.preprocess --dataset ami
    python -m src.preprocess --dataset earvn          # slower, ~28k images
    python -m src.preprocess --dataset ami --sheets-only   # just regenerate QA sheets
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.config import CFG
from src.manifest import dataset_root, load_manifest
from src.paths import P

# Deterministic variants that get written to disk.
CACHED_VARIANTS = ("raw", "zoom", "zoom_canny")


def cache_dir(dataset: str, variant: str) -> Path:
    return P.data / "cache" / dataset / variant


def flat_name(relpath: str) -> str:
    """
    '001.ALI_HD/005 (1).jpg' -> '001.ALI_HD__005 (1).jpg'

    Flattening keeps the cache one level deep (fast to list, fast to zip for
    upload to Kaggle) while staying collision-free, since the original relative
    path is unique.
    """
    return relpath.replace("/", "__")


# ---------------------------------------------------------------------------
# The three image operations
# ---------------------------------------------------------------------------

def apply_zoom(img: np.ndarray, dataset: str) -> np.ndarray:
    """
    AMI:   the paper's documented crop, 492x702 -> 320x490, taken centrally.
    EarVN: the paper skipped zoom here entirely, which confounds every
           AMI-vs-EarVN comparison they make. We apply a consistent centre crop
           so both datasets get the same treatment (extension #4).
    """
    spec = CFG.preprocessing.zoom[dataset]
    h, w = img.shape[:2]

    if spec["mode"] == "fixed_crop":
        out_w, out_h = int(spec["out_w"]), int(spec["out_h"])
        # Guard against images smaller than the target (never happens on AMI,
        # but a silent negative index would be a nasty bug).
        out_w, out_h = min(out_w, w), min(out_h, h)
    elif spec["mode"] == "center_crop_frac":
        frac = float(spec["frac"])
        out_w, out_h = int(w * frac), int(h * frac)
    else:
        raise ValueError(f"unknown zoom mode {spec['mode']!r}")

    x0 = (w - out_w) // 2
    y0 = (h - out_h) // 2
    return img[y0:y0 + out_h, x0:x0 + out_w]


def auto_thresholds(blurred: np.ndarray, pct: float = 90.0,
                    ratio: float = 0.4) -> tuple[int, int]:
    """
    Adaptive thresholds from the GRADIENT distribution, not the median.

    A median-based rule (the common blog recipe) fails on ear images: a bright,
    fairly uniform skin photo has a high median, which pushes the thresholds up
    and returns a blank map. What matters is how strong the edges actually are,
    so we take a high percentile of the Sobel gradient magnitude as the upper
    threshold. That adapts to lighting and contrast per image.
    """
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hi = float(np.percentile(mag, pct))
    hi = max(hi, 10.0)          # never collapse to zero on a flat image
    lo = max(1.0, hi * ratio)
    return int(lo), int(hi)


def apply_canny(img: np.ndarray, dataset: str) -> np.ndarray:
    """
    Canny edge map, returned as 3-channel so it drops into a pretrained CNN
    without changing conv1.

    Thresholds come from config per dataset -- see config.yaml for why.

    Known caveat to state in the report: ImageNet-pretrained filters expect
    natural image statistics. A binary edge map violates that assumption, which
    is very likely why the paper's Canny columns are the worst score in nearly
    every row of both its tables. We reproduce it faithfully and report why it
    fails, rather than pretending it should work.
    """
    c = CFG.preprocessing.canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = int(c["blur_ksize"]) | 1  # kernel must be odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if str(c.get("mode", "fixed")) == "auto":
        # Median-based thresholds adapt per image. Ear photos are smooth skin
        # gradients with very few hard intensity steps, so a single fixed pair
        # that works on one lighting condition can return a blank map on another.
        lo, hi = auto_thresholds(blurred, float(c.get("auto_percentile", 90.0)))
    else:
        # Per-dataset thresholds: AMI and EarVN need different values because
        # their image quality is completely different.
        th = c["thresholds"][dataset]
        lo, hi = int(th["low"]), int(th["high"])

    edges = cv2.Canny(blurred, lo, hi)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def make_variant(img: np.ndarray, variant: str, dataset: str, size: int) -> np.ndarray:
    """
    NOTE ON ORDER: Canny runs AFTER the resize, not before.

    Running Canny at full resolution and then downscaling with INTER_AREA
    averages 1-pixel-wide edge lines into their dark neighbours, dimming the
    whole map toward black. Computing edges at the final resolution keeps them
    crisp and binary. This ordering bug produced entirely black edge maps.
    """
    if variant == "raw":
        out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif variant == "zoom":
        out = cv2.resize(apply_zoom(img, dataset), (size, size),
                         interpolation=cv2.INTER_AREA)
    elif variant == "zoom_canny":
        zoomed = cv2.resize(apply_zoom(img, dataset), (size, size),
                            interpolation=cv2.INTER_AREA)
        out = apply_canny(zoomed, dataset)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_one(job: tuple) -> tuple[str, str | None]:
    """Returns (relpath, error_or_None). Runs in a separate process."""
    relpath, root_s, dataset, size, variants, out_dirs, overwrite = job
    src = Path(root_s) / relpath
    try:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return relpath, "unreadable (corrupt or unsupported)"

        for v in variants:
            # Canny maps are binary-ish; PNG avoids JPEG ringing artefacts.
            ext = ".png" if v == "zoom_canny" else ".jpg"
            dst = Path(out_dirs[v]) / (flat_name(relpath).rsplit(".", 1)[0] + ext)
            if dst.exists() and not overwrite:
                continue
            out = make_variant(img, v, dataset, size)
            ok = cv2.imwrite(str(dst), out)
            if not ok:
                return relpath, f"failed to write {dst.name}"
        return relpath, None
    except Exception as e:
        return relpath, f"{type(e).__name__}: {e}"


def build_cache(dataset: str, overwrite: bool = False, workers: int | None = None) -> None:
    manifest = load_manifest(dataset)
    root = dataset_root(dataset)
    size = int(CFG.data.img_size)

    out_dirs = {}
    for v in CACHED_VARIANTS:
        d = cache_dir(dataset, v)
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[v] = str(d)

    jobs = [
        (rp, str(root), dataset, size, CACHED_VARIANTS, out_dirs, overwrite)
        for rp in manifest["relpath"]
    ]

    print(f"\n=== preprocessing cache: {dataset} ===")
    print(f"  source   : {root}")
    print(f"  images   : {len(jobs):,}  x {len(CACHED_VARIANTS)} variants "
          f"= {len(jobs) * len(CACHED_VARIANTS):,} files")
    print(f"  output   : {P.data / 'cache' / dataset}")
    print(f"  size     : {size}x{size}")

    t0 = time.time()
    errors: list[tuple[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for relpath, err in ex.map(_process_one, jobs, chunksize=32):
            done += 1
            if err:
                errors.append((relpath, err))
            if done % 2000 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                print(f"    {done:,}/{len(jobs):,}  {rate:.0f} img/s  "
                      f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m")

    print(f"  finished in {(time.time()-t0)/60:.1f} min")

    if errors:
        print(f"  !! {len(errors)} FAILURE(S) -- these images are not cached:")
        for rp, err in errors[:10]:
            print(f"     {rp}: {err}")
        print("     Fix or exclude them from the manifest before training; a "
              "missing cache file will crash the dataloader mid-epoch.")
    else:
        print("  [ OK ] every image processed with no errors")

    # Sanity-check counts on disk.
    for v in CACHED_VARIANTS:
        n = len(list(cache_dir(dataset, v).glob("*")))
        flag = "[ OK ]" if n == len(jobs) else "[WARN]"
        print(f"  {flag} {v:<12} {n:,} files on disk (expected {len(jobs):,})")


# ---------------------------------------------------------------------------
# QA contact sheets -- LOOK AT THESE. Do not skip this.
# ---------------------------------------------------------------------------

def make_contact_sheet(dataset: str, n_rows: int = 6, seed: int = 7) -> Path:
    """
    One row per sample image, one column per variant, with labels.

    This is the cheapest bug-catcher in the whole project. Canny thresholds are
    image-dependent: on EarVN's low-resolution in-the-wild photos the edge maps
    may come out as noise or as almost-blank frames. A crop offset can also
    silently slice the ear in half. Both are obvious in two seconds of looking
    and invisible in an accuracy number.
    """
    manifest = load_manifest(dataset)
    rng = np.random.default_rng(seed)
    picks = manifest.iloc[rng.choice(len(manifest), size=min(n_rows, len(manifest)),
                                     replace=False)]

    size = int(CFG.data.img_size)
    pad, header = 8, 28
    cols = ["source"] + list(CACHED_VARIANTS)
    W = len(cols) * (size + pad) + pad
    H = header + len(picks) * (size + pad) + pad
    sheet = np.full((H, W, 3), 30, dtype=np.uint8)

    for ci, name in enumerate(cols):
        x = pad + ci * (size + pad)
        cv2.putText(sheet, name, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)

    root = dataset_root(dataset)
    for ri, (_, row) in enumerate(picks.iterrows()):
        y = header + pad + ri * (size + pad)
        img = cv2.imread(str(root / row["relpath"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        tiles = [cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)]
        tiles += [make_variant(img, v, dataset, size) for v in CACHED_VARIANTS]
        for ci, tile in enumerate(tiles):
            x = pad + ci * (size + pad)
            sheet[y:y + size, x:x + size] = tile

    out = P.results / f"contact_sheet_{dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"\n  QA sheet -> {out}")
    print("  OPEN IT. Check: is the ear fully inside the zoom crop? Do the Canny")
    print("  edges trace the ear outline, or are they noise / nearly blank?")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the preprocessing cache.")
    ap.add_argument("--dataset", default="all", choices=["ami", "earvn", "all"])
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate files that already exist")
    ap.add_argument("--sheets-only", action="store_true",
                    help="skip caching, just regenerate the QA contact sheets")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default: all cores)")
    args = ap.parse_args()

    names = ["ami", "earvn"] if args.dataset == "all" else [args.dataset]
    for n in names:
        try:
            if not args.sheets_only:
                build_cache(n, overwrite=args.overwrite, workers=args.workers)
            make_contact_sheet(n)
        except FileNotFoundError as e:
            print(f"\n[skip] {n}: {e}")


if __name__ == "__main__":
    main()
