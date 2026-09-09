"""
Phase 3, part 3: THE TRAINING SCRIPT.

Trains exactly ONE combination and appends ONE row to results.csv:

    python -m src.train --dataset ami --model resnet50 --arm zoom+aug --seed 0

Design decisions that protect the results:

  * Model selection and early stopping use the VALIDATION split only. The test
    split is touched exactly once, at the very end, using the best checkpoint.
    Peeking at test to pick a model is the most common way student projects
    accidentally inflate their numbers.

  * Reports macro accuracy alongside top-1. EarVN1.0 has 107-300 images per
    subject (2.8x imbalance), so plain accuracy is biased toward well-represented
    identities. Macro accuracy = mean of per-class accuracy, which weights every
    identity equally. The paper reports neither macro accuracy nor the imbalance.

  * Checkpoints every epoch and resumes automatically. Free-tier Kaggle/Colab
    sessions die without warning; this caps the loss at one epoch.

  * One row per run, written the moment the run finishes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import CFG
from src.data import build_loaders
from src.models import build_model, count_params
from src.paths import P
from src.results import log_run, make_run_id
from src.seed import seed_everything


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, dev, num_classes: int) -> dict:
    """Returns top-1, top-5, macro accuracy, and mean loss."""
    model.eval()  # CRITICAL: puts BatchNorm/Dropout in inference mode
    crit = nn.CrossEntropyLoss()

    correct = correct5 = total = 0
    loss_sum = 0.0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)

    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        out = model(x)
        loss_sum += crit(out, y).item() * y.size(0)

        k = min(5, out.size(1))
        topk = out.topk(k, dim=1).indices
        pred = topk[:, 0]

        correct += (pred == y).sum().item()
        correct5 += (topk == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)

        for cls, hit in zip(y.cpu().numpy(), (pred == y).cpu().numpy()):
            per_class_total[cls] += 1
            per_class_correct[cls] += int(hit)

    seen = per_class_total > 0
    macro = float((per_class_correct[seen] / per_class_total[seen]).mean()) if seen.any() else 0.0

    return {
        "top1": correct / max(total, 1),
        "top5": correct5 / max(total, 1),
        "macro": macro,
        "loss": loss_sum / max(total, 1),
    }


def train_one_epoch(model, loader, optimizer, scaler, dev, crit) -> tuple[float, float]:
    model.train()
    correct = total = 0
    loss_sum = 0.0
    use_amp = scaler is not None

    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1), loss_sum / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dataset: str, model_name: str, arm: str, seed: int,
        protocol: str = "ident", epochs: int | None = None,
        batch_size: int | None = None, freeze: bool = False,
        resume: bool = True, dry_run: bool = False,
        pretrained: bool = True) -> None:

    run_id = make_run_id(train_dataset=dataset, model=model_name, arm=arm, seed=seed)
    if protocol != "ident":
        run_id = f"{run_id}__{protocol}"
    if not pretrained:
        run_id = f"{run_id}__scratch"

    seed_everything(seed, deterministic=bool(CFG.seed.deterministic))
    dev = device()
    n_epochs = int(epochs or CFG.train.epochs)

    print(f"\n{'='*70}\n  {run_id}\n{'='*70}")
    print(f"  device   : {dev} "
          f"({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")

    train_loader, val_loader, test_loader, num_classes = build_loaders(
        dataset, arm, seed, protocol=protocol, batch_size=batch_size)

    print(f"  classes  : {num_classes}")
    print(f"  train    : {len(train_loader.dataset):,} images"
          f"   val: {len(val_loader.dataset):,}"
          f"   test: {len(test_loader.dataset):,}" if test_loader else "")

    model = build_model(model_name, num_classes, pretrained=pretrained,
                        freeze_backbone=freeze).to(dev)
    if not pretrained:
        print("  weights   : RANDOM INIT (no ImageNet pretraining)")
    total_m, train_m = count_params(model)
    print(f"  params   : {total_m:.1f}M total, {train_m:.1f}M trainable")

    if dry_run:
        # One batch forward/backward -- catches shape and dtype bugs in seconds.
        x, y = next(iter(train_loader))
        out = model(x.to(dev))
        loss = nn.CrossEntropyLoss()(out, y.to(dev))
        loss.backward()
        print(f"  [DRY RUN OK] batch {tuple(x.shape)} -> logits {tuple(out.shape)}, "
              f"loss {loss.item():.4f}")
        return

    crit = nn.CrossEntropyLoss(label_smoothing=float(CFG.train.label_smoothing))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(CFG.train.lr), weight_decay=float(CFG.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.amp.GradScaler("cuda") if (
        dev.type == "cuda" and bool(CFG.train.mixed_precision)) else None

    ckpt_path = P.checkpoints / f"{run_id}.pth"
    best_path = P.checkpoints / f"{run_id}__best.pth"
    P.checkpoints.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val, patience_left = 0, -1.0, int(CFG.train.early_stop_patience)
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        patience_left = ck["patience_left"]
        print(f"  RESUMED from epoch {start_epoch} (best val {best_val:.4f})")

    t0 = time.time()
    epoch_times: list[float] = []
    final_train_acc = 0.0

    for epoch in range(start_epoch, n_epochs):
        te = time.time()
        tr_acc, tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, dev, crit)
        va = evaluate(model, val_loader, dev, num_classes)
        scheduler.step()
        epoch_times.append(time.time() - te)
        final_train_acc = tr_acc

        improved = va["top1"] > best_val
        if improved:
            best_val = va["top1"]
            patience_left = int(CFG.train.early_stop_patience)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_top1": best_val, "num_classes": num_classes,
                        "model_name": model_name}, best_path)
        else:
            patience_left -= 1

        print(f"  epoch {epoch+1:>3}/{n_epochs}  "
              f"train {tr_acc:.4f} (loss {tr_loss:.4f})  "
              f"val {va['top1']:.4f}  macro {va['macro']:.4f}  "
              f"{epoch_times[-1]:.1f}s{'  *best' if improved else ''}")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_val": best_val, "patience_left": patience_left}, ckpt_path)

        if patience_left <= 0:
            print(f"  early stop: no val improvement for "
                  f"{CFG.train.early_stop_patience} epochs")
            break

    train_time = time.time() - t0

    # --- TEST: exactly once, with the best-by-validation checkpoint ---
    test = {"top1": None, "top5": None, "macro": None}
    if test_loader is not None and best_path.exists():
        ck = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        test = evaluate(model, test_loader, dev, num_classes)
        print(f"\n  TEST  top1 {test['top1']:.4f}  top5 {test['top5']:.4f}  "
              f"macro {test['macro']:.4f}   (best val was {best_val:.4f})")

    log_run(
        run_id=run_id,
        task="identification",
        train_dataset=dataset,
        eval_dataset=dataset,
        arm=arm,
        model=model_name,
        seed=seed,
        split_protocol="image_disjoint" if protocol == "ident" else "subject_disjoint",
        epochs=len(epoch_times) + start_epoch,
        batch_size=int(batch_size or CFG.train.batch_size),
        lr=float(CFG.train.lr),
        img_size=int(CFG.data.img_size),
        train_acc_final=round(final_train_acc, 6),
        best_val_acc=round(best_val, 6),
        test_top1=round(test["top1"], 6) if test["top1"] is not None else "",
        test_top5=round(test["top5"], 6) if test["top5"] is not None else "",
        params_m=round(total_m, 2),
        epoch_time_s=round(float(np.mean(epoch_times)), 2) if epoch_times else "",
        train_time_s=round(train_time, 1),
        notes=f"macro={test['macro']:.4f}" if test["macro"] is not None else "",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one model/arm/dataset/seed.")
    ap.add_argument("--dataset", required=True, choices=["ami", "earvn"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="ident", choices=["ident", "verif"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--freeze", action="store_true",
                    help="train the classifier head only")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="one batch forward/backward, no training -- use this first")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init instead of ImageNet weights (scratch ablation)")
    args = ap.parse_args()

    run(args.dataset, args.model, args.arm, args.seed,
        protocol=args.protocol, epochs=args.epochs, batch_size=args.batch_size,
        freeze=args.freeze, resume=not args.no_resume, dry_run=args.dry_run,
        pretrained=not args.no_pretrained)


if __name__ == "__main__":
    main()
