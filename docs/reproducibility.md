# Reproduction record

Facts below were taken from the machine and repository on 2026-10-16. Only facts that were checked are stated as facts.

## Code version
- Reproduced code: upstream MAMEX, <https://github.com/L2R-UET/MAMEX>, commit `88a932eca821a5657b2b7fc0b6445e4685f4dfab` (the base of this branch; remote `upstream`).
- This project: the final state as a single commit on top of the upstream commit above (submission tag pending). The experiments ran between 2026-10-16 and 2026-10-23 during development. The training and evaluation code did not change after 2026-10-17; later edits only touched reporting scripts, file layout and documentation. The tables regenerate identically from the saved outputs.
- Changes to the upstream code are listed in the report's deviation section (package restructure, `--data_path`, test-time text corruption, VGR, evaluation report). The model, loss and training loop are otherwise unchanged except for the `--train_with_vgr` switch (default off).

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
- Feature extractor. The MAMEX paper describes CLIP features and the README says "CLIP-based feature extraction, inspired by MMSRec"; the authors host only the processed data, and issue #2 on their repository (August 2025) asks for the preprocessing; the reply names no script but points to MMSRec's code and McAuley's raw data (19 September 2025). MMSRec's README (read 2026-10-23) names CLIP without variant or dimensions and lists Beauty, Sports, Clothing, Home and MovieLens-1M, not Baby. The distributed files are 4096-d image (all values >= 0, 79.2% zeros, row norms 36-153) and 1024-d text (dense, signed, row norms 23-30). No released CLIP image encoder outputs a 4096-d embedding (the last ResNet stage of RN50x64 is 4,096 channels wide, Table 19 of the CLIP paper, but CLIP does not output it as the embedding and a pooled map of it would not be 79% exact zeros; our reading), and non-negative sparse 4096-d vectors are what an ImageNet CNN's ReLU layer gives (VBPR's fc7 features). LATTICE (arXiv 2104.09036) describes the same Baby dataset with "4,096-dimensional visual features that have been extracted and published" and 1,024-d sentence-transformers text; DGMRec says its Amazon features come from "a CNN model and SBERT". Our reading: the distributed features are probably CNN and SBERT features, not CLIP. Not proven: item ids are not mapped to ASINs, so the vectors cannot be matched to the published 2014 features, and no CLIP features were extracted here (that would need the raw images and titles plus the mapping). Recorded as a deviation: the code reproduces MAMEX on the authors' distributed features.
- `extracted_modality_features.zip` (also in `~/Downloads`) is MicroLens-100k (CLIP-RN50, BGE-M3, VideoMAE), a different dataset. It is not used here.
- Licence / access. Amazon review data is distributed for research under its own terms by its authors; check the original dataset page before redistributing. The upstream repository's README claims the MIT licence but ships no `LICENSE` file at the base commit. This repository does not redistribute any data, only the instructions and checksums above.

## Seeds and determinism
- Training seeds: 2025, 2026, 2027 (`--seed`). `--noise_seed` defaults to the training seed, so a given seed gives the same corruption mask in every mode.
- Validation noise (for γ selection) uses `noise_seed + 10000`, a different mask from the test mask.
- Same seed and same code give the same clean result (checked: baseline and stress_test with seed 2025 gave identical clean metrics). GPU nondeterminism was not otherwise audited.

## Commands
```
export PYTHONPATH=src
python -m pytest -q                       # 100 tests, no GPU needed
# short real-data check (3 epochs, not a reported result); delete the *_smoketest files it writes in results/
python main.py --data_path processed_data_baby/cold --mode baseline --seed 1 --epoch 3 --eva_interval 1 --suffix smoketest
DATA_PATH=processed_data_baby/cold SEEDS="2025 2026 2027" ./scripts/run_experiments.sh   # baseline, stress_test, improved per seed
python -m mamex_robustness.reporting.plot_results --results_dir results --k 20   # figures
```
Cost and checkpoints. All reported runs used one RTX A4000, about 19 minutes of training each. The 9 core runs (baseline, stress test and improved, three seeds) took 2.8 hours of training in total; the 14 sweep runs (noise rates, manual γ, low γ, extended grid) bring the sum of `train_seconds` over those 23 result files to 10.8 hours. The tests need no GPU; running the experiments on CPU was not tried. No checkpoints are saved or committed: `--save` defaults to 0 and `scripts/run_experiments.sh` does not set it, the best-epoch weights are kept in memory during a run and used for the evaluation, and every result comes from retraining with the seeds above. Evaluation-time conditions (noise rate, VGR, γ, controls) do not change training, so runs with the same seed retrain the same clean model; one checkpoint per seed evaluated under all conditions would have cut most of the total training time.

Discrepancy diagnostics (Rec@10 gap, `docs/hypotheses.md`, results in report §7). The commands are in `scripts/run_discrepancy.sh`, run as three parallel streams on the same GPU:
```
STREAM=A DATA_PATH=processed_data_baby/cold ./scripts/run_discrepancy.sh   # baseline seeds 2028-2034, suffix seedext
STREAM=B DATA_PATH=processed_data_baby/cold ./scripts/run_discrepancy.sh   # align_coeff 1.28 (3 seeds), 12.8 (seed 2025)
STREAM=C DATA_PATH=processed_data_baby/cold ./scripts/run_discrepancy.sh   # align_coeff 12.8 (seeds 2026, 2027), 128 (3 seeds)
```
Result files carry the suffix `seedext` or `align<coeff>`; `make_tables.py` ignores suffixed files, so the main tables are unchanged. These 16 runs add 11.9 hours to the sum of `train_seconds` (about 2,000 to 3,600 s each, slower than a solo run because three ran at once), which brings the total over all 39 result files to 22.7 hours. A solo run takes about 20 minutes.

Outputs: `results/*.json` (one per run, containing args, metrics, gate statistics, latency, GPU memory, training seconds) and the two figures under `results/`. Reported tables must be generated from these files only.

## Tables, failure analysis and sweeps
```
python -m mamex_robustness.reporting.make_tables --results_dir results      # results/tables.md
python -m mamex_robustness.reporting.failure_analysis --results_dir results  # results/failure_analysis.md
python -m mamex_robustness.reporting.ranking_metrics --results_dir results   # results/ranking_metrics.md (Success@K, MRR, bootstrap CIs; seed 0, 1000 resamples)
PARALLEL=1 SKIP_CORE=1 SWEEPS=1 SEEDS=2025 DATA_PATH=processed_data_baby/cold ./scripts/run_experiments.sh   # single-seed sweeps
```
A repeated run of the core grid reproduced every saved number exactly (timing and memory excluded).

## Not done
- Noise-augmented-training ablation (not implemented).
