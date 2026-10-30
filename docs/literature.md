# Literature search and comparison

Compiled on 2026-10-16 from web search results and fetched paper pages (see `docs/ai_use.md`). Rows based on an abstract only say so.

Labels used throughout: **[Authors]** = a claim or number the cited paper reports; **[Interpretation]** = an inference drawn here, not stated in the paper; **[Measured]** = a result from this project's own runs (none appear here yet).

## 1. How the papers were found and selected

Question. Does the adaptive modality gate of a multimodal cold-start recommender over-rely on text when the text is corrupted, and can a visual-grounding signal fix it?

Search (2026-10-16, web search tool, standard mode). Queries by title for each paper, plus topic queries: "MAMEX Multi-modal Adaptive Mixture of Experts cold-start", "MoToRec multimodal tokenization cold-start", "Contrastive Learning for Cold-Start Recommendation", "VBPR", "MMGCN". One extra hit (DGMRec) surfaced while looking for MMGCN follow-ups. This is a targeted selection, not a systematic review; keyword searches were not run over Google Scholar, DBLP or ACM DL.

Selection criteria. (1) the paper that introduced the base idea (VBPR: content features for ranking and cold items; MMGCN: per-modality graph propagation); (2) the paper being reproduced (MAMEX); (3) a competing cold-start approach with a different philosophy (CLCRec: contrastive alignment; MoToRec: discrete tokens); (4) work on a related failure mode (DGMRec: missing modalities in recommendation).

Depth of reading.

| Paper | What was read |
|---|---|
| MAMEX | arXiv HTML (method, tables, ablations); later the full PDF (arXiv 2508.08042v1, 5 pages), which confirmed the notes below |
| MoToRec | arXiv HTML (method, tables, ablations) |
| CLCRec | ar5iv HTML |
| VBPR | ar5iv HTML |
| MMGCN | full PDF (method, equations, protocol, tables, ablations), checked against the matrix row |
| MILK | full PDF (arXiv 2405.15783v1) except the hyperparameter figures; added 2026-10-30 after the acknowledgements of MAMEX's README named it; tables checked against claims in the text |
| DGMRec | full PDF (arXiv 2504.16352v1: method, setup, results, ablations), checked against the matrix row |

## 2. Comparison matrix

Fields follow the course template (one row per paper).

| Paper | Question and contribution | Modalities and assumptions | Mechanism and objective | Data and protocol | Baselines and evidence | Resources and reproducibility | Limitations and connections |
|---|---|---|---|---|---|---|---|
| VBPR (He & McAuley, AAAI 2016) | Can product-image appearance improve implicit-feedback ranking, including for items with little feedback? [Authors]: adds visual factors to BPR-MF. | Image (4096-d CNN FC7, ImageNet-pretrained) + implicit feedback. Items need only an image to be scored. Users with <5 positives dropped. | `x̂ = α + β_u + β_i + γ_uᵀγ_i + θ_uᵀ(E f_i) + β′ᵀf_i`; BPR-OPT. 20 factors, half visual. | Amazon Women/Men/Phones, Tradesy. One random item per user for val and test. Cold = test items with <5 training positives (not zero). AUC. | RAND, MP, IBR, MM-MF, BPR-MF; hyperparameters tuned on validation; no seeds or significance tests stated. Women cold-start AUC: VBPR 0.6813 vs BPR-MF 0.5281; Phones cold-start AUC: VBPR 0.6056 < IBR 0.6319 [Authors]. | Desktop, 4 cores, 32 GB; ~3.5 h on Women. Code "at publication time", no link in text. | Single image per item; visual/latent split fixed 50/50; no component ablation. Image only. |
| MMGCN (Wei et al., ACM MM 2019) | Model modality-specific user preferences for micro-video recommendation. [Authors]: one user-item graph per modality with message passing. | Visual, acoustic, textual pre-extracted features. Items must already sit in the interaction graph, so it is not a cold-item method [Interpretation, from the architecture]. | Per-modality GCN aggregation (mean or max) and combination layers with a shared user-ID embedding; user and item representations are the sum of the modality-specific ones, scored by inner product; BPR loss with L2 regularisation; SGD. | Tiktok 726,065 / Kwai 1,664,305 / MovieLens 1,239,508 interactions. Random 8:1:1 split of interactions; each test pair ranked against 1000 unobserved videos; Precision/Recall/NDCG@10. | VBPR, ACF, GraphSAGE, NGCF. Recall gains over the strongest baseline 15.59% / 10.23% / 8.76% (Kwai / Tiktok / MovieLens) [Authors]. Modality ablation: the multimodal model beats each single modality; visual is the most effective, text the least on Kwai and Tiktok. | Authors state they release codes, parameters and baselines. | Random interaction split: no cold-item evaluation, no corrupted-modality test. The paper itself calls the Kwai and Tiktok descriptions "noisy, incomplete, and even irrelevant" while text helps on MovieLens, so modality quality varies by dataset, but no corrupted-text test is run. |
| CLCRec (Wei et al., ACM MM 2021) | Cold-start item embeddings from an information-theoretic view: maximize mutual information between content and collaborative signals. | Pre-extracted visual/acoustic/text features (ResNet, VGGish, Sentence2Vector); Amazon uses 64-d visual only. Cold items scored from the content encoder alone. | Contrastive U-I and R-E objectives, `λ L_RE + (1-λ) L_UI`; hybrid training swaps in content features with probability ρ. | Movielens, Tiktok, Kwai, Amazon; cold items split 1:1 into val/test, warm items 8:1:1. Recall@10, NDCG@10. | DUIF, DropoutNet, MTPR, CB2CF, Heater; same 64-d embedding for all. Single runs, no variance. | Code and data released; hardware not stated. | Single runs without variance. No text-corruption test. |
| MoToRec (Liu, Zhang & Cheung, AAAI 2026) | Argues aligning noisy continuous multimodal embeddings causes "semantic fog"; proposes discrete semantic tokens. | Visual (BEiT), text (BGE), learnable ID embeddings. Cold = items with <10 training interactions. Deployment procedure for new items not described [Authors' text]. | Sparsity-regularized RQ-VAE per modality, rarity weighting, LightGCN-based multi-source graph encoder; `L_BPR + L_CL + L_RQ + L_reg`. | Amazon Baby 19,445 users / 7,050 items; Sports; Clothing. Random 8:1:1 over interactions; cold group = test items with <10 training interactions. R@/N@{10,20}. | MF-BPR, LightGCN, VBPR, MMGCN, LATTICE, FREEDOM, BM3, LGMRec, others, in MMRec; grid search only described for MoToRec [Interpretation: baseline tuning fairness unverifiable]. Baby R@20 0.1077; Baby cold-start R@20 0.0347 [Authors]. | Hardware and code not stated. | Cold-start differs from complete cold-start (items here still have some interactions). Claims noisy embeddings are a problem but does not corrupt inputs to test it. |
| MAMEX (Nguyen et al., CIKM 2025) (the paper reproduced) | Cold-start fusion fails because concat/average/fixed weights ignore modality relations; adaptive gating across modality experts. | CLIP image and text features [Authors]; no dimensions stated. Complete item cold-start: all interactions of dev/test items removed. | Modality-specific MoE (linear experts, top-k) + sparse-softmax gate over `[z_img ∥ z_text]`; `L_BPR + λ1 L_align + λ2 L_adapter + λ3 L_fusion + λ4‖Θ‖²`. | Amazon Baby, Clothing, Sports; 8:1:1; Rec/NDCG@{10,20}. | MTPR, AlignRec, MetaEmbed, DropoutNet, CLCRec, GAR, MILK. Baby: Rec@20 0.0771, NDCG@20 0.0363 [Authors]. Modality ablation (figure only): text beats vision, both beat either alone. | Code public (github.com/L2R-UET/MAMEX); batch size, epochs, embedding size and hardware not stated. | No limitations section; future work lists missing modalities. Evaluated on clean features only; the gate's behavior under corrupted text is untested. Direct target of this project. |
| DGMRec (Kim et al., SIGIR 2025) | Multimodal recommenders degrade when item modalities are missing, and injecting a global or nearest-neighbour mean for the missing feature does not prevent it [Authors]. Proposes disentangling general and modality-specific features and generating the missing ones. | Image and text on Amazon (pre-extracted: a CNN for images, SBERT for text, following earlier work [Authors]); TikTok adds audio. Some or all modality features of an item are absent; present features are assumed correct [Interpretation]. | Autoencoder; general and specific encoders trained with two information-based losses; a missing modality is generated from aligned features of the other modalities and from the modality preferences of the item's users; generated features refine the item-item graph. | Amazon Baby, Sports, Clothing and TikTok; 8:1:1; Recall and NDCG@{20, 50}. Missing setting: for two-modality data, a third of items miss 0, 1 or 2 modalities; new-item setting: 20% of items appear only in test; missing ratios 0 to 80%. | 5 CF models, 10 multimodal recommenders (VBPR, MMGCN, LATTICE, LGMRec, GUME and others) and 3 missing-modality-aware ones; DGMRec is best in the missing and new-item settings and drops less at an 80% missing ratio (10.2% vs 15.1% for the baseline it is compared with) [Authors]. | Code public (github.com/ptkjw1997/DGMRec); Adam, embedding size 64, up to 1,000 epochs, early stopping 30, λ1 and λ2 tuned in {1, 0.1, 0.01, 0.001}. | No limitations section. Handles absent modalities, not present but misleading ones; the injected features it criticises are shown to hurt, which is the closest the paper comes to a corrupted input, but no corrupted-text test is run [Interpretation]. |
| MILK (Bai et al., SIGIR 2024) (supporting) | New-item recommendation when items miss a modality. [Authors]: invariant learning, so the preference prediction does not change across modality-missing environments. | Visual and text on Baby and Clothing, TikTok adds audio; pre-extracted, extractor not stated in the text read. Missing modality mean-filled; present features assumed correct [Interpretation]. | Modality-specific extractors with cross-modal alignment; Dirichlet fusion weights with cyclic shifts as environments; loss = mean environment loss + beta x variance across environments + lambda x alignment. | Baby 19,442 users, 5,640 train / 705 val / 705 test items; 20% of items new; FTMT (one modality missing at test) and MTMT (30% of training items also miss one); Recall, NDCG@{10,20}. | DUIF, MICRO, DropoutNet, MTPR, Heater, CLCRec, CCFCRec, GAR, GoRec. Baby FTMT Rec@20 0.0628, NDCG@20 0.0277, +11.2% NDCG@20 over the best baseline; mean of five runs, no variance in the text read [Authors]. MAMEX lists MILK at Rec@20 0.0730 on Baby, a different setting [Interpretation]. | PyTorch, 64-d, batch 2048, Adam 0.001, early stopping; code public (github.com/HaoyueBai98/MILK); hardware not stated; beta tuned per dataset. | Tests absence, not misleading content. Same item split as MAMEX and this project. Imputation by zeros, means or a learned image-to-text map did not help (Baby Rec@20 0.0501, 0.0525, 0.0507 vs 0.0628). |

## 3. Synthesis

Where the papers agree [Interpretation]
- All recommendation papers use *pre-extracted* content features and learn only what sits on top (VBPR FC7, MMGCN/CLCRec dataset features, MoToRec BEiT/BGE, MAMEX CLIP).
- Content helps items with little or no history: VBPR's Women cold-start AUC and MAMEX's multimodal-over-single-modality ablation point the same way.
- Text tends to be the stronger modality in the Amazon setting (MAMEX's ablation). That makes text the natural shortcut.

Where they disagree or are not comparable
- *Cold-start definition.* VBPR: <5 positives; MoToRec: <10 interactions; CLCRec and MAMEX: complete cold start. Their numbers cannot be compared: MoToRec reports Baby R@20 0.0347 on its cold group and 0.1077 overall, MAMEX reports Baby Rec@20 0.0771 on a different protocol [Authors' numbers].
- *Continuous vs discrete.* MoToRec claims aligning continuous embeddings is the source of noise ("semantic fog"); MAMEX stays continuous and relies on a gate to weight modalities. Neither tests corrupted inputs, so the disagreement is untested [Interpretation].
- *Split units.* MMGCN and MoToRec split interactions at random; MAMEX and CLCRec split by item. Random splits cannot evaluate cold items.
- *Reproducibility.* MAMEX, MoToRec and VBPR (in the text read) leave key settings or code links unstated; CLCRec releases code but gives single runs.

Research gap this project tests [Interpretation]. In the papers read here, none reports performance when a modality is present but *misleading*. DGMRec and MILK address a missing modality (MILK trains with Dirichlet-weighted fusion environments and a variance penalty across them; it reports that zero, mean or learned-map imputation does not help on Baby). Both are training-time remedies, whereas VGR and the text repair work on a trained model; the settings differ, so no numbers are compared. The testable question is whether MAMEX's gate keeps trusting text when the text disagrees with the image, and whether a parameter-free visual-grounding signal helps.

## 4. Reproduction target (kept apart from measured results)

MAMEX paper, Baby, as published [Authors]:

| Rec@10 | Rec@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|
| 0.0512 | 0.0771 | 0.0292 | 0.0363 |

Setup differences already visible (to be listed in the report's deviations section):
- The paper says CLIP features; the processed Baby data used here has 4096-d image and 1024-d text features. The data folder equals the authors' zip and the two feature files equal the ones LATTICE published (`docs/reproducibility.md`), so they are LATTICE's features, not CLIP features.
- Batch size, epochs, embedding size and hardware are not stated in the paper; this project uses the repository defaults (batch 2048, 128-d embeddings, up to 1000 epochs, early stop 200).
- The repository's `Env` hard-coded the data path; it is now `--data_path`.

Measured baseline numbers will be added only from `results/*.json`.

## 5. Open items

1. Every row was checked against the full text, as a PDF (MAMEX, MMGCN, DGMRec) or the HTML version of the paper (VBPR, CLCRec, MoToRec).
2. Fields marked `TO VERIFY` in `references.bib` are unconfirmed.
3. The matrix has the five papers the course requires (VBPR, MMGCN, CLCRec, MoToRec, MAMEX); DGMRec and MILK are supporting (seventh paper added 2026-10-30). A second MoE or gating cold-start paper would strengthen it.
