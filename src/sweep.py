"""
Phase 4/5 driver: run the whole experiment matrix.

Instead of typing 15-45 train commands by hand, this loops over
models x arms x seeds and calls the trainer for each.

The important behaviour: it SKIPS runs already present in results.csv. So when
a Kaggle session dies at run 9 of 15, you re-run the same command and it picks
up at run 9. Nothing is repeated, nothing is lost.

    # see what would run, without running it
    python -m src.sweep --dataset ami --dry

    # the real thing
    python -m src.sweep --dataset ami

    # one seed only (use this for the expensive EarVN sweep first)
    python -m src.sweep --dataset earvn --seeds 0

    # narrow it down
    python -m src.sweep --dataset ami --models resnet50 --arms zoom zoom+aug
"""

from __future__ import annotations

import argparse
import time
import traceback

from src.config import CFG
from src.results import load_results, make_run_id
from src.train import run as train_run


def completed_run_ids() -> set[str]:
    df = load_results()
    if df.empty or "run_id" not in df:
        return set()
    done = df[df["test_top1"].notna()]
    return set(done["run_id"].astype(str))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the experiment matrix.")
    ap.add_argument("--dataset", required=True, choices=["ami", "earvn"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="list runs, don't train")
    ap.add_argument("--force", action="store_true", help="re-run completed runs")
    args = ap.parse_args()

    models = args.models or list(CFG.experiments.models)
    arms = args.arms or list(CFG.experiments.arms)
    seeds = args.seeds if args.seeds is not None else list(CFG.seed.seeds)

    done = set() if args.force else completed_run_ids()

    jobs, skipped = [], []
    for model in models:
        for arm in arms:
            for seed in seeds:
                rid = make_run_id(train_dataset=args.dataset, model=model,
                                  arm=arm, seed=seed)
                (skipped if rid in done else jobs).append(rid)
                if rid not in done:
                    jobs[-1] = (rid, model, arm, seed)

    jobs = [j for j in jobs if isinstance(j, tuple)]

    print(f"\n{'='*70}")
    print(f"  SWEEP: {args.dataset}")
    print(f"  {len(models)} models x {len(arms)} arms x {len(seeds)} seeds "
          f"= {len(models)*len(arms)*len(seeds)} total")
    print(f"  already done : {len(skipped)}")
    print(f"  to run       : {len(jobs)}")
    print(f"{'='*70}")

    if skipped:
        print("\n  skipping (already in results.csv):")
        for rid in skipped[:10]:
            print(f"    {rid}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped)-10} more")

    if not jobs:
        print("\n  Nothing to do. Use --force to re-run.")
        return

    print("\n  queued:")
    for rid, *_ in jobs:
        print(f"    {rid}")

    if args.dry:
        print("\n  (dry run -- nothing trained)")
        return

    t0 = time.time()
    failures = []
    for i, (rid, model, arm, seed) in enumerate(jobs, 1):
        print(f"\n\n>>> [{i}/{len(jobs)}] {rid}"
              f"   (elapsed {(time.time()-t0)/60:.1f}m)")
        try:
            train_run(args.dataset, model, arm, seed,
                      epochs=args.epochs, batch_size=args.batch_size)
        except KeyboardInterrupt:
            print("\n  interrupted -- progress is saved, re-run to continue")
            raise
        except Exception:
            # One bad run must not kill the sweep. Record and continue.
            print(f"  !! FAILED: {rid}")
            traceback.print_exc()
            failures.append(rid)

    print(f"\n{'='*70}")
    print(f"  sweep finished in {(time.time()-t0)/60:.1f} min")
    if failures:
        print(f"  {len(failures)} FAILED: {failures}")
    else:
        print("  all runs completed")
    print(f"{'='*70}")
    print("\n  Summary so far:")
    from src.results import summarise
    s = summarise()
    if not s.empty:
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
