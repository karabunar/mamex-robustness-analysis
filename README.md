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
Not included. Where to get it, its licence status and what is still unknown: `docs/reproducibility.md`.

## Results
Test split, cold items only, hashtag noise on 40% of cold items, mean over seeds 2025–2027 (`results/tables.md`, built from `results/*.json`).

| Condition | Recall@20 | NDCG@20 |
|---|---|---|
| Clean baseline | 0.0776 | 0.0365 |
| Noisy text | 0.0668 | 0.0307 |
| Noisy text, VGR | 0.0701 | 0.0319 |
| Noisy text, shuffled-cosine control | 0.0684 | 0.0313 |

The published Baby numbers (Rec@20 0.0771, NDCG@20 0.0363) are listed separately in the report. VGR recovers about a third of the Recall@20 lost to the noise in all three seeds, and about half of that gain is also obtained by a control that does not use the image. Details, controls and failure analysis: `docs/report/report.tex`, section VII.

## Literature
`docs/literature.md`, `docs/references.bib`.

## AI use
Disclosed in `docs/ai_use.md`.

## Credit
Upstream code: <https://github.com/L2R-UET/MAMEX> (commit `88a932e`).
