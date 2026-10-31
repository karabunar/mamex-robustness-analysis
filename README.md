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

**Caveat.** On 2026-10-28 the installed PyTorch was 2.14.1+cu130, not the pinned 2.9.1, and a rerun of a clean core baseline (seed 2026) gave Recall@20 0.0753 instead of the saved 0.0736. The code is not the cause (the pre-ablation code gives 0.0753 too); on 2026-10-29 the pinned 2.9.1 was installed in a separate environment and seed 2026 retrained there: it reproduces the saved baseline exactly (epoch 410, Recall@20 0.07361), so the version is the cause. Seeds 2025 and 2027 reproduce exactly there too (checkpoints in `results/ckpt291/`, weights not committed). The saved core results are the earlier runs; see `docs/reproducibility.md`.

### Where the cold items sit (H7)
Added 2026-10-29, hypothesis written before the runs. Six more runs (`--repr_probe 1`) measure where cold items sit in the space the gate reads. Cold items are **closer** to the warm centroid, not farther (distance ratio 0.72 in the augmented model, 0.78 in the control; H7a needed 1.2, not supported). Across cold items the distance hardly correlates with the text weight (Spearman +0.08 to +0.15; H7c needed 0.3, not supported). What differs is the gate's answer: logit difference text minus image on cold items is +11.7 in the augmented model against -0.14 on warm items (control +0.20 against -0.05). Cold items' expert outputs are also smaller, clearly so in the augmented model (image norm 1.16 against 1.93); whether that drives the gate was not tested. The cause is still unknown. The six runs equal the earlier gate-probe runs bit for bit.

### Training-free text repair (H8)
Added 2026-10-29, hypotheses written before the runs. On the three saved clean-trained checkpoints (`--save_final`, PyTorch 2.9.1, equal to the core baselines), part of each cold item's text is replaced by the mean text of its ten nearest warm items in image space (`python -m mamex_robustness.repair_cli`). The model, its weights and its forward pass do not change; parameters are chosen on noisy validation data only.

| Recall@20, 40% of cold text corrupted | noisy text | gain over unrepaired (seeds > 0) | clean text |
|---|---|---|---|
| Unrepaired | 0.0668 | | 0.0776 |
| VGR | 0.0701 | +0.0033 (3/3) | 0.0773 |
| **M1, blind blend with image-neighbour text** | 0.0746 | **+0.0078 (3/3)** | 0.0769 |
| M1, random neighbours (control) | 0.0666 | -0.0002 (1/3) | 0.0759 |
| M2, gated by agreement | 0.0719 | +0.0052 (3/3) | 0.0784 |
| M2, permuted agreement (control) | 0.0740 | +0.0072 (3/3) | 0.0784 |
| M1, then VGR | 0.0764 | +0.0096 (3/3) | 0.0777 |
| **M3, repair only items flagged by VGR's cosine (H9)** | 0.0761 | **+0.0094 (3/3)** | 0.0792 |
| M3, random neighbours (control) | 0.0644 | -0.0023 (0/3) | 0.0721 |
| M3, permuted cosine (control) | 0.0724 | +0.0056 (3/3) | 0.0794 |
| M3, then VGR | 0.0772 | +0.0105 (3/3) | 0.0789 |
| Oracle: true corruption mask (not an upper bound) | 0.0735 | +0.0068 (3/3) | 0.0776 |

- H8a supported (M2 beats the unrepaired model and VGR), H8e supported (0.6 s once, 41 MB, forward 7.9 against 8.1 ms), H8b, H8c and H8d not supported.
- The image-neighbour text is what helps (random neighbours gain nothing). The per-item agreement score does not detect corrupted items (AUROC 0.51 to 0.71) and M2 loses to its permuted-score control, so the simple blind blend M1 is the best method: about 72% of the loss recovered alone, about 89% with VGR, no measurable clean-text cost.
- VGR's own cosine detects the corrupted items almost perfectly (AUROC 0.999), so its limit is what it does with a detected item. Not tested.
- One synthetic noise family (the same used throughout), one dataset, three seeds. The M2 random-neighbour control was added after the first run, on the same checkpoints.
- **H9** (written after H8, before the runs): repairing only the items VGR's cosine flags beats the blind blend in all three seeds (+0.0094 against +0.0078, margins +0.0029, +0.0005, +0.0011), beats both controls in all three, changes clean text by +0.0016, and the cosine costs 0.017 s. With VGR it reaches +0.0105, about 96% of the loss (M3 alone about 86%). The oracle that repairs exactly the corrupted items is *not* an upper bound: M3 and M1 both beat it, so the neighbour text also helps clean items (prediction wrong). The cosine was chosen as detector after seeing its AUROC in H8, and `q` is chosen on validation data that carry the noise rate, as VGR's gamma is.
- **H10** (written after H9, before the runs): the same checkpoints under hashtag, shuffle and Gaussian noise at rates 0.2, 0.4 and 0.6, and with the validation noise fixed at hashtag 0.4 while the test noise differs. M3 beats the unrepaired model in all three seeds in all 9 settings (H10a) and keeps 99% to 116% of its gain under mismatched validation noise (H10f); more noise gives more gain (H10d), clean text changes by +0.0009 to +0.0019 (H10e). H10c does not hold: the blind blend M1 is as good or better in 4 of 9 settings, notably at rate 0.6 (hashtag: M1 +0.0135, M3 +0.0101), and for Gaussian noise at 0.4 M3 is only +0.0004 above its permuted-cosine control. M3 followed by VGR reaches 71% to 124% of the loss (above 100% means the repaired model on noisy text is above the unrepaired one on clean text). The table is in the section "Text repair under other noise" of `results/tables.md`; commands in `docs/reproducibility.md`.
- **H11** (written after H10, before the runs): two more seeds (2028, 2029) for the main setting, retrained in PyTorch 2.9.1, identical to their earlier seed-extension baselines. M3 beats the unrepaired model and both controls in both new seeds. Over five seeds M3 gains +0.0090 Recall@20 (M3 then VGR +0.0099, M1 +0.0075, VGR +0.0028; unrepaired 0.0661, clean 0.0767), is at least M1 in 4 of 5 seeds, and the bootstrap interval of M3 minus M1 is [+0.0004, +0.0025], a small advantage at rate 0.4. The three-seed numbers in H8 to H10 are kept as first reported; `results/tables.md` shows five seeds for the hashtag 0.4 row.
- **H12 and H13** (written after H11, before the runs; two open questions of the report). H12, the Rec@10 gap: the three clean baselines were retrained with per-epoch validation and test curves (`--curve 1`; same best epochs 480, 410, 470). Choosing the epoch with the test set (diagnostic only) gives a mean test Rec@10 of 0.0502 against the published 0.0512 (the threshold, met by 0.00004); validation-based rules give 0.0482 to 0.0491, so none closes the gap and the gap is described as not identified. H13, why the gate of the noise-augmented model picks text on cold items: the gate's bias (+0.18, +0.22 at zero input) and the norm of the cold expert outputs (rescaling leaves w_t at 0.99) do not explain it; replacing a cold item's image-side expert output by a warm item's lowers w_t from 0.98 to 0.32 to 0.36, replacing the text-side output does not (exploratory follow-up). Two seeds. Tables: `results/h12_h13.md` (`python scripts/h12_h13_tables.py`).
- **H14 to H18** (written before the runs). H14: ranking among the 705 test items only raises every metric by 53% to 59% and the per-metric best epochs match the published vector in no seed, so neither explains the Rec@10 gap (`results/h14.md`). H15: 72% on average of M3's gain comes from the truly corrupted items (54% to 87% across five seeds; `results/h15.md`). H16: with seeds 2030 to 2034 added, M3 minus the blind blend M1 is +0.0019 Recall@20 over ten seeds, paired bootstrap interval [+0.0004, +0.0037], M3 at least M1 in 8 of 10 seeds; one seed carries the mean (`results/h16.md`). H17: items with no training interactions behave like cold items (text weight 0.96 against 0.35 on trained items in the noise-augmented model; 0.48 to 0.54 in the clean-trained control; `results/h17.md`). H18: a second dataset (Amazon Sports, our own cold split built from the LATTICE files): the method ranking is the same (M3 above M1 above VGR, all above the unrepaired model and the controls in 3 of 3 seeds) but M3 recovers about 38% of the loss against about 85% on Baby (`results/h18_sports.md`).
- **H19 and H20** (written before the analyses). The gate's text weight falls with the number of training interactions of a warm item (Spearman -0.69 and -0.76 in the augmented model, -0.39 and -0.32 in the control; `results/h19.md`); of four statistics of the image output, the direction towards the users carries information beyond the interaction count and the raw feature density does not explain it (`results/h20.md`). No single trace is identified.
- **H21 to H23** (written before the runs). H21: swapping the user-direction component of the image expert output between cold and warm items moves the gate's text weight by 0.18 to 0.38 (cold) and 0.20 (warm), random directions by 0.001 (`results/h21.md`). H22: the neighbour text is less informative on Sports (keeps 76% of the clean Recall@20 against 93% on Baby; `results/h22.md`). H23: the parts of M3's gain do not add up for random partitions either, so the interaction is generic (`results/h23.md`).
- Tables: `results/tables.md`; discussion: report section VII-E.

## Tests
`python -m pytest -q` runs 158 tests in about 3 seconds on CPU, with a tiny synthetic dataset (no real data needed). They cover the noise simulator and its mask rules (18), the VGR gating and controls (8 + 4), the session evaluation and report (14 + 4), the CLI (9 + 5 end to end), the legacy behaviour of the upstream code before the restructure (5), result tables, plots, failure analysis and bootstrap metrics (13 + 8 + 8 + 6). The newest ones (18) check the ablation and its probe: training noise touches only warm items, at the requested share, is redrawn each epoch, never reaches validation or the stored model state, and does not overlap the test mask; the ablation table pairs every seed with its clean-trained runs and is left out when no augmented run exists. Status on 2026-10-29: 149 passed. The tests check the code, not the scientific results.

## Quick demo
`scripts/demo.sh` runs three short steps (about 2.5 minutes on one GPU, measured; steps 1 and 2 take seconds and need neither data nor GPU):
```
pip install -r requirements.txt
PY=python DATA_PATH=<folder with the Baby data, see Data> ./scripts/demo.sh
```
1. The tables of `results/` are rebuilt from the saved result files (`results/tables.md`, `results/h12_h13.md`); `git diff` shows no change when they are reproduced exactly.
2. The 158 unit tests run on a tiny synthetic dataset.
3. A clean baseline is trained for 21 epochs with a checkpoint, evaluated under corrupted cold-item text (40% of the cold items), with VGR, and with the training-free text repair (M1, M3), and the Recall@20 of each is printed.

Step 3 is a demonstration of the pipeline, not a result: a 21-epoch model is far from converged (best epochs in the reported runs are 230 to 730), so the effects of VGR and of the repair that the report measures on converged models are not reliably visible in it. The reported numbers come from `scripts/run_experiments.sh`, `scripts/run_repair.sh` and the commands in `docs/reproducibility.md`.

## Reused components
`docs/reused_components.md`: what comes from the upstream MAMEX code (compared line by line with commit `88a932e`), the data and the literature, and what was written for this project.

## Literature
`docs/literature.md`, `docs/references.bib`.

## AI use
Disclosed in `docs/ai_use.md`.

## Credit
Upstream code: <https://github.com/L2R-UET/MAMEX> (commit `88a932e`).
