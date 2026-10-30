# Reused-component inventory

What this repository takes from other work, what it changes, and what is new. The upstream reference is the MAMEX repository (`L2R-UET/MAMEX`, commit `88a932e`, 2026-09-12; remote `upstream`). Every statement below was produced by comparing that commit with the current tree (`git diff -M 88a932e HEAD`), not from memory. Who wrote which part (author or AI assistant) is recorded in `docs/ai_use.md`, not here.

## 1. Reused from the upstream MAMEX code

| Upstream file | Now at | Change | What it holds |
|---|---|---|---|
| `criterion.py` | `src/mamex_robustness/models/criterion.py` | none (byte-identical) | BPR, MSE and related loss functions |
| `evaluation.py` | `src/mamex_robustness/application/evaluation.py` | one comment translated | Faiss top-K search (`IndexFlatIP`, exact) and Recall, HR and NDCG |
| `model.py` | `src/mamex_robustness/models/mamex.py` | comments translated; added `set_eval_text`, an optional VGR step after the gate, and storing the gate weights and cosine; training forward pass unchanged when VGR is off and no evaluation text is set | MoE experts, two-way gate, fusion and balance losses |
| `session.py` | `src/mamex_robustness/application/session.py` | split into a package; the training loop (`train_epoch`, early stopping, best-epoch choice on validation NDCG@10) is kept; added corrupted-text evaluation, noise-augmented training, probes, `save_best` | training and evaluation loop |
| `loader.py` | `src/mamex_robustness/data/loader.py` | comments translated; added `noisy_text_feat` and `train_noisy_text_feat` | data loading, splits, negative sampling |
| `tool.py` | `src/mamex_robustness/tool.py` | comments translated; added `apply_mode_defaults` | logging and small helpers |
| `enviroment.py` | `src/mamex_robustness/application/environment.py` | file renamed; the data path is read from `--data_path` instead of being hard-coded | environment and device set-up |
| `main.py` | `main.py` | replaced by a three-line launcher; the argument parsing and run logic moved to `src/mamex_robustness/cli.py` | entry point |
| `README.md` | `README.md` | replaced by this project's README | |

The model, the losses, the sampling and the evaluation metric are the upstream ones. Training and evaluation logic are otherwise unchanged; characterisation tests (`tests/characterization/`) were written to hold the legacy behaviour fixed across the restructure.

## 2. Data (not ours)

| Item | Source | Check |
|---|---|---|
| `processed_data_baby/cold/` (30 files) | the authors' `processed_data_baby.zip` | SHA-256 of all 30 files identical |
| `image_feat.npy`, `text_feat.npy` | published earlier by the LATTICE authors (`baby` folder of their Google Drive link) | identical, maximum absolute difference 0.0 (`scripts/compare_features.py`) |
| Baby item split (705 validation, 705 test cold items) | the same split as MILK and MAMEX | counts match MILK's Table 1 |

The data were not modified. The data are not committed here (`.gitignore`); the README's Data section gives the download link and where to put them.

## 3. Libraries

PyTorch 2.9.1, NumPy, SciPy, Numba, faiss-cpu, tensorboardX, tqdm, Matplotlib, pytest (versions in `requirements.txt`). Nothing is vendored.

## 4. Written for this project

New files, none of which exist upstream (line counts of the current files):

| Area | Files | Lines |
|---|---|---|
| Corrupting cold-item text (hashtag, shuffle, Gaussian) | `src/mamex_robustness/domain/noise.py` | 66 |
| VGR and its controls | `src/mamex_robustness/domain/gating.py` | 38 |
| Training-free text repair (M1, M2, M3) | `src/mamex_robustness/domain/repair.py`, `src/mamex_robustness/application/repair_eval.py`, `src/mamex_robustness/repair_cli.py` | 262 |
| Command line (modes baseline, stress_test, improved; flags for ablation, probes, checkpoints, curves) | `src/mamex_robustness/cli.py` | 155 |
| Probes: gate, representation, gate-cause interventions | in `src/mamex_robustness/application/session.py`, `src/mamex_robustness/gate_cause_cli.py` | (inside the session) |
| Per-pair ranks for failure analysis | `src/mamex_robustness/application/diagnostics.py` | 39 |
| Tables, figures, ranking metrics, failure analysis | `src/mamex_robustness/reporting/*.py` | 866 |
| Run scripts | `scripts/*.sh`, `scripts/compare_features.py` | |
| Tests (including characterisation tests that pin the upstream behaviour) | `tests/` | 1561 |

`src/mamex_robustness/application/session.py` combines both: the upstream training loop and the additions above. The line count of the current file (420) includes both; the upstream file had 191 lines.

## 5. Ideas and methods from the literature (cited in the report)

| Idea | Used for | Source |
|---|---|---|
| Adaptive modality gate over experts | the model under study | MAMEX (Nguyen et al., CIKM 2025) |
| Cold-start evaluation by item split | the protocol | MAMEX, CLCRec, MILK |
| Missing or failing modality as a research topic | motivation and related work | DGMRec, MILK |
| Cosine between image and text expert outputs | the signal that VGR and repair M3 read | designed in this project; we did not take it from a specific cited method |
| Nearest-neighbour imputation of text from image space | repair M1, M2, M3 | a generic imputation idea; the form used here is ours |
