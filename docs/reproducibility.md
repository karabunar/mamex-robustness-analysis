# Reproduction record

Facts below were taken from the machine and repository on 2026-10-16. Only facts that were checked are stated as facts.

## Code version
- Reproduced code: upstream MAMEX, <https://github.com/L2R-UET/MAMEX>, commit `88a932eca821a5657b2b7fc0b6445e4685f4dfab` (the base of this branch; remote `upstream`).
- This project: a few commits on top of the upstream commit above, all made on 2026-10-27 after the experiments, so commit timestamps do not show when the runs were made (submission tag pending). The experiments ran between 2026-10-16 and 2026-10-23 during development. The training and evaluation code did not change in behaviour after 2026-10-17: later edits touched reporting scripts, file layout and documentation. The one exception is the noise-augmented ablation of 2026-10-28, which added the `--train_noise_rate` switch (default 0, training on clean text as before) and three runs; the core grids, sweeps and diagnostics were not rerun and the existing tables regenerate identically from their saved outputs.
- Changes to the upstream code are listed in the report's deviation section (package restructure, `--data_path`, test-time text corruption, VGR, evaluation report). The model, loss and training loop are otherwise unchanged except for the `--train_with_vgr` and `--train_noise_rate` switches (both default off).

## Environment
| Item | Value |
|---|---|
| OS | Linux 7.0.0 |
| CPU / RAM | Intel Core i9-12900K, 62 GB |
| GPU | NVIDIA RTX A4000 |
| Python | 3.13.9 |
| Packages | pinned in `requirements.txt` |

Setup: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt` (install the matching `torch` wheel first).

## Data
Amazon Baby, item cold-start split, used from `processed_data_baby/cold/` (gitignored; never committed).

| File | SHA-256 |
|---|---|
| train.txt | `330c8bdff9b82a3b9b83f24dae5830c915a9aff239352f788959a15b71c3b83f` |
| val.txt | `4768ff30900b4ff84d053ee2df416cb9b51248a6b20eed1fc139f7f47ed82382` |
| test.txt | `d72c5f69bbe1a3ae6683efecdb6ed40e24613c05dc0d5f1617f40dc79be18aea` |
| cold_item_index.npy | `fd57f2947047eed66be739be65355b0214c129d0d6e31fa55ecf95e3d3b20abd` |
| image_feat.npy | `2fc1695bc4fdf58a13d325dd0a63fe270d864ec4c8b26b780f8a1eee9682747e` |
| text_feat.npy | `c0c9b38fc9871d635814484e5948f4c4eeb2e0c33668babdf3747898ce4400b2` |

Shapes: 7,050 items; 1,410 cold items; 19,442 training users; `image_feat` 7050×4096 (float64); `text_feat` 7050×1024 (float32).

- Source: verified by checksum. `processed_data_baby.zip` was downloaded from the Google Drive link in the upstream README (<https://drive.google.com/file/d/1C-18Y84lMS5xsRGwKQYa9P0Tot-gxgYY/view>). The SHA-256 of the six files above, streamed out of that zip (`cold/…`), equals the checksums in the table. File dates inside the zip are 2023. The zip holds no raw item text, images or ASIN mapping.
- Feature extractor. The MAMEX paper describes CLIP features and the README says "CLIP-based feature extraction, inspired by MMSRec"; the authors host only the processed data, and issue #2 on their repository (August 2025) asks for the preprocessing; the reply names no script but points to MMSRec's code and McAuley's raw data (19 September 2025). MMSRec's README (read 2026-10-23) names CLIP without variant or dimensions and lists Beauty, Sports, Clothing, Home and MovieLens-1M, not Baby. The distributed files are 4096-d image (all values >= 0, 79.2% zeros, row norms 36-153) and 1024-d text (dense, signed, row norms 23-30). No released CLIP image encoder outputs a 4096-d embedding (the last ResNet stage of RN50x64 is 4,096 channels wide, Table 19 of the CLIP paper, but CLIP does not output it as the embedding and a pooled map of it would not be 79% exact zeros; our reading), and non-negative sparse 4096-d vectors are what an ImageNet CNN's ReLU layer gives (VBPR's fc7 features). LATTICE (arXiv 2104.09036) describes the same Baby dataset with "4,096-dimensional visual features that have been extracted and published" and 1,024-d sentence-transformers text; DGMRec says its Amazon features come from "a CNN model and SBERT". The data folder equals the authors' `processed_data_baby.zip` (SHA-256 of all 30 files identical, none missing or extra; checked 2026-10-30). Checked directly on 2026-10-30: the `baby` folder published by the LATTICE authors (Google Drive link in the LATTICE README, files dated 2021-09-04) contains `image_feat.npy` (7050x4096, float64) and `text_feat.npy` (7050x1024, float32) that are identical to the distributed ones (same order, maximum absolute difference 0.0). The features are therefore LATTICE's published features, not CLIP features; how LATTICE extracted them was not re-checked, and why the paper and README say CLIP is not known. Recorded as a deviation: the code reproduces MAMEX on the authors' distributed features.
- `extracted_modality_features.zip` (also in `~/Downloads`) is MicroLens-100k (CLIP-RN50, BGE-M3, VideoMAE), a different dataset. It is not used here.
- Licence / access. Amazon review data is distributed for research under its own terms by its authors; check the original dataset page before redistributing. The upstream repository's README claims the MIT licence but ships no `LICENSE` file at the base commit. This repository does not redistribute any data, only the instructions and checksums above.

## Seeds and determinism
- Training seeds: 2025, 2026, 2027 (`--seed`). 2025 is the default seed of the upstream MAMEX code; 2026 and 2027 are the next two integers. None was chosen or replaced after looking at results. Seeds 2028 and 2029 were added later for H11 in the same way, 2028 to 2034 having been used before in the seed-extension runs (D1). `--noise_seed` defaults to the training seed, so a given seed gives the same corruption mask in every mode.
- Validation noise (for γ selection) uses `noise_seed + 10000`, a different mask from the test mask.
- Same seed and same code give the same clean result within one environment (checked on 2026-10-28: baseline and stress_test with train_noise_rate 0 give identical results; six identical jobs run at once give identical results; the pre-ablation code and the current code give identical results over 40 epochs and, for seed 2026, over a full run).
- Across environments they do not. A repeat of the core grid between 2026-10-17 and 2026-10-23 reproduced every saved number exactly. On 2026-10-28 the installed PyTorch was 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24), not the pinned 2.9.1, and a clean baseline of seed 2026 gave Recall@20 0.0753 (best epoch 480) against the saved 0.0736 (410); seed 2025 gave 0.0809 against 0.0819 and seed 2027 0.07740 against 0.07741. The code is not the cause (the pre-ablation code gives 0.0753 too). On 2026-10-29 the pinned `torch==2.9.1` (cu128) was installed in a separate environment and seed 2026 retrained: it reproduces the saved baseline exactly (best epoch 410, Recall@20 0.07361, NDCG@20 0.03410; the noisy block is equal too), so the PyTorch version is the cause. Seeds 2025 and 2027, retrained there with `--save_final 1` (results in `results/ckpt291/`, weights gitignored), reproduce their saved baselines exactly too (epochs 480 and 470; Recall@20 0.08193 and 0.07741; clean and noisy blocks equal). The core, sweep and diagnostic results in `results/` are the saved earlier runs; the ablation and gate-probe runs (suffixes `noise_aug`, `gateprobe_aug`, `gateprobe_clean`) were made in the 2.14.1 environment, so the ablation is compared with the clean-trained control of the gate-probe runs as well as with the saved core runs. The differences between the two comparisons are small (noisy Recall@20: -0.0105 against -0.0100).

## Commands
```
export PYTHONPATH=src
python -m pytest -q                       # 158 tests, no GPU needed
# short real-data check (3 epochs, not a reported result); delete the *_smoketest files it writes in results/
python main.py --data_path processed_data_baby/cold --mode baseline --seed 1 --epoch 3 --eva_interval 1 --suffix smoketest
DATA_PATH=processed_data_baby/cold SEEDS="2025 2026 2027" ./scripts/run_experiments.sh   # baseline, stress_test, improved per seed
python -m mamex_robustness.reporting.plot_results --results_dir results --k 20   # figures
NOISE_AUG=1 SKIP_CORE=1 SEEDS="2025 2026 2027" DATA_PATH=processed_data_baby/cold ./scripts/run_experiments.sh   # noise-augmented training ablation (H5), suffix noise_aug
```
Cost and checkpoints. All reported runs used one RTX A4000, about 19 minutes of training each. The 9 core runs (baseline, stress test and improved, three seeds) took 2.8 hours of training in total; the 14 sweep runs (noise rates, manual γ, low γ, extended grid) bring the sum of `train_seconds` over those 23 result files to 10.8 hours. The tests need no GPU; running the experiments on CPU was not tried. No checkpoints are saved or committed: `--save` defaults to 0 and `scripts/run_experiments.sh` does not set it, the best-epoch weights are kept in memory during a run and used for the evaluation, and every result comes from retraining with the seeds above. Evaluation-time conditions (noise rate, VGR, γ, controls) do not change training, so runs with the same seed retrain the same clean model; one checkpoint per seed evaluated under all conditions would have cut most of the total training time.

Discrepancy diagnostics (Rec@10 gap, `docs/hypotheses.md`, results in report §7). The commands are in `scripts/run_discrepancy.sh`, run as three parallel streams on the same GPU:
```
STREAM=A DATA_PATH=processed_data_baby/cold ./scripts/run_discrepancy.sh   # baseline seeds 2028-2034, suffix seedext
STREAM=B DATA_PATH=processed_data_baby/cold ./scripts/run_discrepancy.sh   # align_coeff 1.28 (3 seeds), 12.8 (seed 2025)
STREAM=C DATA_PATH=processed_data_baby/cold ./scripts/run_discrepancy.sh   # align_coeff 12.8 (seeds 2026, 2027), 128 (3 seeds)
```
Result files carry the suffix `seedext` or `align<coeff>`; `make_tables.py` ignores suffixed files, so the main tables are unchanged. These 16 runs add 11.9 hours to the sum of `train_seconds` (about 2,000 to 3,600 s each, slower than a solo run because three ran at once), which brings the total over all 39 result files to 22.7 hours. The three noise-augmented runs (suffix `noise_aug`, three in parallel, 2,333 to 2,554 s each, ablation H5) bring it to 24.7 hours over 42 files, and the six gate-probe runs (below, six in parallel, 4,400 to 5,100 s each) to 32.7 hours over 48 files, and the six representation-probe runs (H7, five to six thousand seconds each with seven jobs on the GPU) to 42.0 hours over 54 files; `make_tables.py` reports them in separate sections. A solo run takes about 20 minutes.

Outputs: `results/*.json` (one per run, containing args, metrics, gate statistics, latency, GPU memory, training seconds) and the two figures under `results/`. Reported tables must be generated from these files only.

## Tables, failure analysis and sweeps
```
python -m mamex_robustness.reporting.make_tables --results_dir results      # results/tables.md
python -m mamex_robustness.reporting.failure_analysis --results_dir results  # results/failure_analysis.md
python -m mamex_robustness.reporting.ranking_metrics --results_dir results   # results/ranking_metrics.md (Success@K, MRR, bootstrap CIs; seed 0, 1000 resamples)
PARALLEL=1 SKIP_CORE=1 SWEEPS=1 SEEDS=2025 DATA_PATH=processed_data_baby/cold ./scripts/run_experiments.sh   # single-seed sweeps
```
A repeated run of the core grid reproduced every saved number exactly (timing and memory excluded) when it was made, in the environment pinned in `requirements.txt`. It no longer does in the 2.14.1 environment, see the section on seeds and determinism.

## Gate probe (H6)
Mean gate weights on warm items, split by a training-style corruption, for the noise-augmented model and a clean-trained control (three seeds each, suffixes `gateprobe_aug` and `gateprobe_clean`); the table is the "Gate probe" section of `results/tables.md`:
```
for seed in 2025 2026 2027; do
  python main.py --data_path processed_data_baby/cold --mode stress_test --seed $seed --noise_rate 0.4 --train_noise_rate 0.4 --gate_probe 1 --suffix gateprobe_aug
  python main.py --data_path processed_data_baby/cold --mode stress_test --seed $seed --noise_rate 0.4 --train_noise_rate 0   --gate_probe 1 --suffix gateprobe_clean
done
```

## Representation probe (H7)
Where the cold items sit in the space the gate reads, for the noise-augmented model and a clean-trained control (three seeds each, suffixes `repr_aug` and `repr_clean`, section "Representation probe" of `results/tables.md`). `--save_final 1` additionally writes the best-validation weights to `results/checkpoints/<run>.pt` (gitignored); the older `--save 1` is not used because it deletes the file it has just written:
```
for seed in 2025 2026 2027; do
  python main.py --data_path processed_data_baby/cold --mode stress_test --seed $seed --noise_rate 0.4 --train_noise_rate 0.4 --repr_probe 1 --suffix repr_aug
  python main.py --data_path processed_data_baby/cold --mode stress_test --seed $seed --noise_rate 0.4 --train_noise_rate 0   --repr_probe 1 --suffix repr_clean
done
```

## Text repair on saved checkpoints (H8)
The three clean-trained checkpoints come from the pinned environment (`torch==2.9.1`, cu128) and equal the core baselines (seeds 2025 to 2027; best epochs 480, 410, 470):
```
for seed in 2025 2026 2027; do
  python main.py --data_path processed_data_baby/cold --mode baseline --seed $seed --save_final 1 --results_dir results/ckpt291
done
for seed in 2025 2026 2027; do
  python -m mamex_robustness.repair_cli --data_path processed_data_baby/cold --seed $seed --noise_rate 0.4
done
```
The second loop also evaluates M3 and the oracle of H9. It trains nothing; it loads `results/ckpt291/checkpoints/*.pt` (gitignored) and writes `results/repair/*.json`, which `make_tables.py` turns into the section "Text repair" of `results/tables.md`. The three checkpoint runs took 2,548 to 2,747 s each (three in parallel), 2.2 hours in total.

## Other noise and mismatched validation (H10)
No training; same checkpoints. `scripts/run_repair.sh` runs the repairs for one noise setting (default hashtag 0.4), `GRID=1` for hashtag, shuffle and Gaussian at rates 0.2, 0.4 and 0.6 (validation noise equal to test noise) and `MISMATCH=1` for the four settings whose validation noise is fixed at hashtag 0.4:
```
PY=/path/to/venv/bin/python GRID=1 MISMATCH=1 DATA_PATH=processed_data_baby/cold ./scripts/run_repair.sh
```
Results go to `results/repair/` (mismatched ones carry `_val<type><rate>` in the name); `make_tables.py` writes the section "Text repair under other noise". The 13 settings (39 evaluations, three in parallel) took about 20 minutes.

## Five-seed check of the repair (H11)
Seeds 2028 and 2029 were trained in the pinned environment like the first three and evaluated with the default setting (hashtag 0.4):
```
for seed in 2028 2029; do
  python main.py --data_path processed_data_baby/cold --mode baseline --seed $seed --save_final 1 --results_dir results/ckpt291
done
PY=/path/to/venv/bin/python SEEDS="2028 2029" DATA_PATH=processed_data_baby/cold ./scripts/run_repair.sh
```
Both retrained baselines equal the earlier seed-extension runs of the discrepancy investigation, made in the 2.14.1 environment (best epochs 320 and 730; Recall@20 0.07673 and 0.07393), so these two seeds reproduce across both environments. Training took 968 s and 1,656 s.

## Epoch rule and gate-cause checks (H12, H13)
H12: the clean baselines are retrained with per-epoch curves (they must reproduce the saved best epochs 480, 410, 470). H13: two noise-augmented models are trained with checkpoints, then the gate input of the cold items is changed on saved checkpoints (no training):
```
for seed in 2025 2026 2027; do
  python main.py --data_path processed_data_baby/cold --mode baseline --seed $seed --curve 1 --results_dir results/curves291
done
for seed in 2025 2026; do
  python main.py --data_path processed_data_baby/cold --mode stress_test --seed $seed --noise_rate 0.4 --train_noise_rate 0.4 --save_final 1 --results_dir results/ckpt_aug
  python -m mamex_robustness.gate_cause_cli --data_path processed_data_baby/cold --seed $seed --ckpt results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed$seed.pt --out results/gate_cause/clean_seed$seed.json
  python -m mamex_robustness.gate_cause_cli --data_path processed_data_baby/cold --seed $seed --ckpt results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed$seed.pt --out results/gate_cause/aug_seed$seed.json
done
python scripts/h12_h13_tables.py   # writes results/h12_h13.md
```
All runs in the pinned environment (`torch==2.9.1`). The augmented training of H13 took about one hour each with five jobs sharing the GPU. The curve flag only records metrics; it does not change training or the choice of epoch.

## Follow-ups on the open points (H14 to H20)
Training-free analyses on the saved checkpoints (CPU is enough):
```
PYTHONPATH=src python scripts/h14_candidates.py --data_path processed_data_baby/cold      # results/h14.md
PYTHONPATH=src python scripts/h15_decomposition.py --data_path processed_data_baby/cold   # results/h15.md
PYTHONPATH=src python scripts/h19_dose_response.py                                          # results/h19.md
PYTHONPATH=src python scripts/h20_trace.py                                                  # results/h20.md
```
H16: seeds 2030 to 2034 are retrained with checkpoints (`--save_final 1 --results_dir results/ckpt291`), then `python -m mamex_robustness.repair_cli --data_path processed_data_baby/cold --seed $s --noise_rate 0.4` for each, then `python scripts/h16_ten_seeds.py` (paired bootstrap, 10,000 resamples, seed 0). H17: for seeds 2025 and 2026, `python main.py --data_path processed_data_baby/cold --mode stress_test --seed $s --noise_rate 0.4 --train_noise_rate 0.4 --holdout_warm 0.1 --holdout_probe 1 --suffix h17_aug --results_dir results/h17` and the same with `--train_noise_rate 0.0 --suffix h17_clean`, then `python scripts/h17_table.py`. H18 (a second dataset): `SRC=<folder of the LATTICE Sports files> PY=<python> ./scripts/run_second_dataset.sh` builds the cold split (`scripts/build_cold_dataset.py`, seed 2025, 20% of the items cold), trains three clean baselines with checkpoints, runs the repair evaluation and writes `results/h18_sports.md`. The split is our construction, not the authors' split. The Sports data (about 300 MB) are not in the repository.

Item ids can be mapped to ASINs for the Baby items through the `5-core/item_list.txt` of the LATTICE Baby files (7,050 items); `space/build_items.py` joins them to the 2023 Amazon catalogue for titles and image links (4,353 of the 7,050 items have both) for the item browser of the demo app.

Cost: the five H16 checkpoint runs took 10.6 hours, the four H17 runs 6.9 hours and the three Sports runs 15.1 hours (3 to 5 jobs sharing one GPU). All in the pinned environment (`torch==2.9.1`).

## Not done
- Other noise-augmentation designs (rate, prototype scheme, noise on the image side) and other noise types in training.
