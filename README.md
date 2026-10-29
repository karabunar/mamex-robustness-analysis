# MAMEX robustness analysis

Reproduction of MAMEX (Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation, CIKM 2025) on Amazon Baby, a stress test with corrupted text on cold items, and a proposed fix, Visual Grounding Regularization (VGR). YZM 5110 Multimodal Learning, Project 1.

The upstream README is kept in `docs/upstream_README.md`. The full write-up is `docs/report/report.tex` (IEEE conference format).

## Research question
Does MAMEX's adaptive gate keep trusting a text embedding that has been corrupted, and does a parameter-free image-text agreement signal (VGR) reduce the damage without hurting clean accuracy? Hypotheses, written before full-training results were read: `docs/hypotheses.md`.

## What was added to MAMEX
- `--mode baseline | stress_test | improved` (`main.py`, `src/mamex_robustness/`).
- Test-time corruption of cold items' text features (hashtag / shuffle / gaussian). Training and validation stay clean.
- VGR: `w_t ← w_t · sigmoid(γ · cos(img_emb, text_emb))`, renormalised to `w_v + w_t = 1`. No learned parameters, applied at evaluation on the clean-trained checkpoint (`--train_with_vgr 1` also applies it in training, as an ablation). γ is chosen on noisy validation data (`--gamma_grid`).
- Controls: `--vgr_control shuffled` and `constant`.
- Ablation: `--train_noise_rate` corrupts that share of warm items' text in every training epoch (default 0, off); results carry the suffix `noise_aug` and are reported apart from the main tables.
- Evaluation report per run: clean / noisy, with and without VGR, gate weights, latency, GPU memory.

## Setup and commands
See `docs/reproducibility.md` (versions, data checksums, seeds, exact commands). Short form:
```
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest -q
DATA_PATH=<folder with train.txt, val.txt, test.txt, cold_item_index.npy, image_feat.npy, text_feat.npy> ./scripts/run_experiments.sh
python -m mamex_robustness.reporting.plot_results --results_dir results --k 20
```

## Data
Not included in the repository. Download `processed_data_baby.zip` from the Google Drive link in the upstream README: <https://drive.google.com/file/d/1C-18Y84lMS5xsRGwKQYa9P0Tot-gxgYY/view>. Unzip it and pass the `cold/` folder as `DATA_PATH` (`processed_data_baby/cold`). It must contain `train.txt`, `val.txt`, `test.txt`, `cold_item_index.npy`, `image_feat.npy` and `text_feat.npy` (7,050 items, 1,410 cold items, 19,442 training users).

SHA-256 checksums of the six files, the feature-provenance notes and the licence status are in `docs/reproducibility.md`. Amazon review data is distributed under its own research terms; check the original dataset page before redistributing. No data is committed here.

## Results
Test split, cold items only, hashtag noise on 40% of cold items, mean over seeds 2025–2027 (`results/tables.md`, built from `results/*.json`).

| Condition | Recall@20 | NDCG@20 |
|---|---|---|
| Clean baseline | 0.0776 | 0.0365 |
| Noisy text | 0.0668 | 0.0307 |
| Noisy text, VGR | 0.0701 | 0.0319 |
| Noisy text, shuffled-cosine control | 0.0684 | 0.0313 |

The published Baby numbers (Rec@20 0.0771, NDCG@20 0.0363) are listed separately in the report. VGR recovers about a third of the Recall@20 lost to the noise in all three seeds, and about half of that gain is also obtained by a control that does not use the image. 95% bootstrap intervals over test users, Success@K and MRR are in `results/ranking_metrics.md` (`python -m mamex_robustness.reporting.ranking_metrics`). The Recall@20 gain of VGR over the noisy model is +0.0033 [0.0024, 0.0044] and over the shuffled control +0.0017 [0.0011, 0.0023]; on clean text VGR costs about 0.0004 Recall@20 (interval includes 0) and 0.0002 NDCG@20 (interval just excludes 0). The intervals cover test-user sampling, not training-seed variance. Details, controls and failure analysis: `docs/report/report.tex`, section VII.

### Ablation: noise-augmented training (H5)
Added 2026-10-28, after H1 to H4 and their results were known; hypothesis written before the runs (`docs/hypotheses.md`). In every training epoch 40% of the warm items' text is corrupted with the same hashtag noise (`--train_noise_rate 0.4`; new mask and prototypes each epoch, cold items are never seen in training). Evaluation is as in the main grid. Three seeds, Recall@20 / NDCG@20, mean:

| Condition | Clean training | Noise-augmented training |
|---|---|---|
| Clean text | 0.0776 / 0.0365 | 0.0725 / 0.0325 |
| Noisy text | 0.0668 / 0.0307 | 0.0568 / 0.0261 |
| Noisy text, VGR | 0.0701 / 0.0319 | 0.0571 / 0.0263 |

- H5a (augmentation lifts noisy accuracy): **not supported**. Worse than the clean-trained model in 3/3 seeds (mean -0.0100 Recall@20).
- H5b (costs clean accuracy): supported, -0.0052 Recall@20 (3/3 seeds), less than the 0.0108 the noise costs the baseline.
- H5c (VGR adds less on top): supported in size (+0.0003 against +0.0033), but not in mechanism: the gate of the augmented models puts almost all weight on text (w_t 0.986 on clean and 0.917 on noisy cold items, against 0.55 and 0.56 with clean training). We did not find out why.

This is one design (rate 0.4, same noise family as the test noise, new prototypes each epoch), so it shows that this augmentation did not help, not that augmentation cannot. Per-seed numbers and paired differences: `results/tables.md`; discussion: report section VII-D. Reproduce: `NOISE_AUG=1 SKIP_CORE=1 DATA_PATH=processed_data_baby/cold ./scripts/run_experiments.sh`.

### Where the gate collapses (H6) and a reproducibility caveat
Added 2026-10-28, hypothesis written before the runs (`docs/hypotheses.md`). Six extra runs (three seeds, with and without augmentation) measure the mean text weight w_t of the gate on warm items, which the models were trained on (`--gate_probe 1`):

| Model | warm, corrupted | warm, not corrupted | warm, no noise | cold, no noise |
|---|---|---|---|---|
| Noise-augmented training | 0.313 | 0.435 | 0.436 | **0.987** |
| Clean training (control) | 0.495 | 0.487 | 0.487 | 0.549 |

- H6a (corrupted warm items get 0.2 less text weight): not supported, gap 0.11 to 0.13.
- H6b (uncorrupted warm items above 0.8, balance-loss compensation): not supported, 0.435.
- H6c (no gap in the control): supported.

The collapse belongs to the cold items: the augmented model gives warm items about 0.44 and unseen cold items 0.99. We still do not know why. Against the same-environment control, the augmented model is lower in all three seeds (-0.0105 Recall@20 on noisy text, -0.0054 on clean text).

**Caveat.** On 2026-10-28 the installed PyTorch was 2.14.1+cu130, not the pinned 2.9.1, and a rerun of a clean core baseline (seed 2026) gave Recall@20 0.0753 instead of the saved 0.0736. The code is not the cause (the pre-ablation code gives 0.0753 too); the PyTorch version was not tested as the cause. The saved core results are the earlier runs; see `docs/reproducibility.md`.

## Tests
`python -m pytest -q` runs 118 tests in about 3 seconds on CPU, with a tiny synthetic dataset (no real data needed). They cover the noise simulator and its mask rules (18), the VGR gating and controls (8 + 4), the session evaluation and report (14 + 4), the CLI (9 + 5 end to end), the legacy behaviour of the upstream code before the restructure (5), result tables, plots, failure analysis and bootstrap metrics (13 + 8 + 8 + 6). The newest ones (18) check the ablation and its probe: training noise touches only warm items, at the requested share, is redrawn each epoch, never reaches validation or the stored model state, and does not overlap the test mask; the ablation table pairs every seed with its clean-trained runs and is left out when no augmented run exists. Status on 2026-10-28: 118 passed. The tests check the code, not the scientific results.

## Literature
`docs/literature.md`, `docs/references.bib`.

## AI use
Disclosed in `docs/ai_use.md`.

## Credit
Upstream code: <https://github.com/L2R-UET/MAMEX> (commit `88a932e`).
