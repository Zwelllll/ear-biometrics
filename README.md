# Ear Biometrics: Reproduction + Extension

Reproduction of **Mohamed et al., "Advancing Ear Biometrics: Enhancing Accuracy and
Robustness through Deep Learning"** (arXiv:2406.00135v1), plus three extensions the
original paper did not attempt.

## What we do beyond the paper

| # | Extension | Why it matters |
|---|-----------|----------------|
| 1 | **Multi-seed runs with mean ± std** | The paper reports single-run point estimates. Several of its claimed 1–2% improvements may be noise. |
| 2 | **Cross-dataset evaluation** | Train on one dataset, test on the other. The paper only ever reports closed-set within-dataset accuracy. |
| 3 | **Open-set verification (EER / DET / TAR@FAR)** | The paper is identification-only. No embeddings, no pairs, no EER anywhere in it. |
| 4 | **Consistent zoom on both datasets** | The paper applies zoom to AMI but explicitly skips it on EarVN1.0, confounding any cross-dataset comparison. |
| 5 | **Real efficiency measurement** | Params, FLOPs, latency, memory. The paper claims an efficiency story with no numbers. |

## Setup

```bash
git clone <your-repo>
cd ear-biometrics
pip install -r requirements.txt
python smoke_test.py          # must pass before anything else
```

On **Kaggle** (primary GPU — 30 free hrs/week on a P100, same hardware the paper
used) and **Colab** (overflow), `src/paths.py` auto-detects the environment. On
Colab, mount Drive *first* or your work is deleted on disconnect:

```python
from google.colab import drive; drive.mount('/content/drive')
```

## Layout

```
config.yaml        ALL hyperparameters. Never hardcode a number in a script.
smoke_test.py      Phase 0 sanity check. Run on laptop before using GPU quota.
src/paths.py       Environment-aware paths (laptop / Kaggle / Colab)
src/config.py      Config loader with dot-access + validation
src/seed.py        Seeds all 5 sources of randomness
src/results.py     One-row-per-run CSV logging (survives disconnects)
splits/            Split CSVs -- COMMITTED to git, this is your reproducibility
results/           results.csv + figures
checkpoints/       Model weights (gitignored)
```

## Experiment matrix

3 models (ResNet50, MobileNetV2, EfficientNet-B0) × 5 arms
(`raw`, `zoom`, `zoom+canny`, `zoom+aug`, `zoom+canny+aug`) × 2 datasets (AMI, EarVN1.0).

Seed policy: all 3 seeds on AMI (minutes per run); single seed sweep on EarVN1.0
then 3 seeds on claim-bearing rows only.

## Roadmap

- [x] **Phase 0** — Scaffolding
- [ ] **Phase 1** — Data + split protocol ⚠️ *most important phase*
- [ ] **Phase 2** — Preprocessing cache
- [ ] **Phase 3** — Training harness + smoke test
- [ ] **Phase 4** — AMI sweep
- [ ] **Phase 5** — EarVN1.0 sweep
- [ ] **Phase 6** — Cross-dataset evaluation
- [ ] **Phase 7** — Verification
- [ ] **Phase 8** — Efficiency benchmarking
- [ ] **Phase 9** — Statistics + failure analysis
- [ ] **Phase 10** — Self-contained brief

## Rules that protect the results

1. **Split first, then augment.** Augmenting before splitting puts augmented copies
   of the same photo in train *and* test. This is leakage and it produces fake-high
   accuracy — possibly the explanation for the paper's 99.35%.
2. **Never augment val or test.**
3. **Never touch the test set** for model selection or early stopping. Use val.
4. **Compare test accuracy, never train accuracy.** Every model in the paper hit
   ~100% train on AMI; that column carries no information.
5. **`model.eval()` before inference**, or BatchNorm stays in training mode and
   your numbers are meaningless.
6. **Write to Drive / Kaggle output**, never the session disk.
