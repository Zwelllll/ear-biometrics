"""
Phase 0 smoke test.

Run this FIRST, on your laptop, before you spend a single GPU-minute:

    python smoke_test.py

It checks the four things the whole project rests on:
  1. Paths resolve for whatever machine you're on
  2. config.yaml loads and validates
  3. Seeding actually reproduces (this is the one people get wrong)
  4. results.csv can be written and read back

If this passes on your laptop, it will pass on Kaggle and Colab too.
"""

from __future__ import annotations

import sys

OK = "[ OK ]"
FAIL = "[FAIL]"


def check_paths():
    from src.paths import P
    P.ensure()
    print(P.describe())
    assert P.results.exists(), "results dir was not created"
    print(f"{OK} paths resolve and directories exist (env={P.env})")


def check_config():
    from src.config import CFG, find_config
    print(f"       config: {find_config()}")
    n_runs = len(CFG.experiments.models) * len(CFG.experiments.arms)
    print(f"       matrix: {len(CFG.experiments.models)} models x "
          f"{len(CFG.experiments.arms)} arms = {n_runs} runs per dataset per seed")
    assert CFG.train.epochs > 0
    print(f"{OK} config loads and validates")


def check_seeding():
    """
    The real test: seed, draw numbers, re-seed, draw again, expect identical.
    If this fails, your entire 'mean +/- std across seeds' story is invalid.
    """
    import random

    import numpy as np
    import torch

    from src.seed import seed_everything

    def draw():
        return (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )

    seed_everything(0)
    a = draw()
    seed_everything(0)
    b = draw()
    seed_everything(1)
    c = draw()

    assert a == b, f"same seed gave different numbers:\n  {a}\n  {b}"
    assert a != c, "different seeds gave identical numbers (seeding is a no-op)"
    print(f"       seed 0 -> {tuple(round(x, 6) for x in a)}")
    print(f"       seed 1 -> {tuple(round(x, 6) for x in c)}")
    print(f"{OK} seeding is reproducible across python/numpy/torch")


def check_results_logging():
    from src.results import load_results, log_run, summarise

    # Two fake runs with the same config but different seeds, so summarise()
    # has something to compute a std over.
    for seed, acc in [(0, 0.9123), (1, 0.9087)]:
        log_run(
            task="identification",
            train_dataset="ami",
            arm="zoom+aug",
            model="resnet50",
            seed=seed,
            split_protocol="image_disjoint",
            epochs=1,
            batch_size=32,
            lr=0.0003,
            img_size=224,
            test_top1=acc,
            notes="SMOKE TEST -- delete this row before real runs",
        )

    df = load_results()
    assert len(df) >= 2, "rows were not written"

    # Unknown-field guard should raise, not silently swallow a typo.
    try:
        log_run(train_dataset="ami", test_topl=0.5)  # note: lowercase L typo
    except KeyError:
        print(f"       typo guard works (rejected 'test_topl')")
    else:
        raise AssertionError("unknown field was silently accepted -- guard broken")

    print(summarise().to_string(index=False))
    print(f"{OK} results.csv writes, reads back, and summarises")


def main() -> int:
    checks = [
        ("paths", check_paths),
        ("config", check_config),
        ("seeding", check_seeding),
        ("results logging", check_results_logging),
    ]
    failed = []
    for name, fn in checks:
        print(f"\n--- {name} " + "-" * (60 - len(name)))
        try:
            fn()
        except Exception as e:
            print(f"{FAIL} {name}: {type(e).__name__}: {e}")
            failed.append(name)

    print("\n" + "=" * 66)
    if failed:
        print(f"{FAIL} {len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"{OK} Phase 0 complete -- skeleton is sound.")
    print("     Next: delete the smoke-test rows from results/results.csv,")
    print("     then move to Phase 1 (data + split protocol).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
