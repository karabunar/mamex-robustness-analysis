# Pre-registered hypotheses

Written before any full-training result was read. Course rule: the rationale is stated *before* results are examined (YZM5110 §4).

Written on 2026-10-16, before the first full-training results of 2026-10-17. Until then the only output seen was a 3-epoch smoke run (not trained, not interpreted); the seed-2025 grid was running but unread. The repository history was built on 27 October, after the experiments, so commit timestamps do not show when the hypotheses or the runs were made; this date is the author's statement.

## Research question

Does MAMEX's adaptive gate over-rely on the textual expert when an item's text is corrupted (clickbait-like), and does a parameter-free Visual Grounding Regularization (VGR) that down-weights text when it disagrees with the image reduce that failure without hurting clean-input quality?

Setting: Amazon Baby, item cold-start (1410 cold items), text corrupted at test time only on a fraction `noise_rate` of cold items; training and validation stay clean. Metrics: Recall@20 and NDCG@20 on the test set (HR/Recall/NDCG at K ∈ {10,…,50} are all saved); mean ± std over seeds 2025–2027.

## H1: the failure mode exists

*Claim.* With `noise_rate=0.4` (hashtag noise), the clean-trained baseline loses Recall@20 and NDCG@20, and its gate does not shift weight away from text on the corrupted items (mean `w_t` on corrupted items stays within ±0.05 of its clean value).

*Why I expect it.* The gate sees only the post-MoE embeddings and was trained on clean data, where text and image agree. Nothing in training taught it that a text embedding can be confidently wrong.

*What would make me doubt it.* (a) the drop in Recall@20/NDCG@20 is within one seed standard deviation of zero, or (b) `w_t` on corrupted items falls by more than 0.10 under noise (the gate already adapts). Either result means the "language shortcut" story is unsupported for this model and noise type, and the report must say so.

## H2: VGR reduces the failure

*Claim.* On the same clean-trained checkpoint, enabling VGR lowers `w_t` on corrupted items and raises noisy Recall@20/NDCG@20 compared with the stress-test condition. Expected ordering of means: clean baseline > MAMEX-VGR (noisy) > stress test (noisy). I do not predict a magnitude.

*Why I expect it.* The alignment loss pulls image and text embeddings together on clean data, so a low cross-modal cosine signals that the text disagrees with the visual evidence; `sigmoid(γ·cos)` turns that into a lower text weight.

*What would make me doubt it.* No improvement, or an improvement that is not the same sign in all three seeds. With 3 seeds I will report paired differences and will not claim statistical significance.

## H3: the cost is small

*Claim.* On clean inputs VGR changes Recall@20 by less than one seed standard deviation, and the forward pass slows by less than 5%.

*Why.* VGR adds one cosine and one sigmoid per item. But `sigmoid(γ·cos) < 1` always, so text is also down-weighted on clean items; a small clean-input loss is plausible and must be measured, not assumed away.

*What would make me doubt it.* A clean-input drop larger than one seed standard deviation, which would mean the robustness is bought with clean quality.

## H4: the gain comes from grounding, not from muting text

*Claim.* Two controls, shuffled-cosine (each item receives another item's cosine) and constant (every item receives the mean VGR factor), recover less than VGR on noisy inputs.

*What would make me doubt the rationale.* If either control matches VGR, the benefit is just "trust text less", and the cross-modal-grounding explanation is not supported.

## H5 (added 2026-10-28, after H1 to H4 and the discrepancy runs were read, before any noise-augmented run)

This ablation was listed as "not run" in the report. H1 to H4 and their results were known when H5 was written; no noise-augmented run existed. It is a post-hoc addition, not part of the original pre-registration.

*Setting.* Same data, seeds 2025 to 2027, same test mask and `noise_rate` 0.4 hashtag noise at evaluation. In every training epoch 40% of the *warm* items' text is corrupted with the same hashtag simulator (new mask and new prototypes each epoch, seeded from `noise_seed + 20000 + epoch`; cold items are never seen in training, so the noise can only act on warm items). Validation, model selection and the evaluation conditions are unchanged: best epoch on clean validation NDCG, `gamma` on noisy validation data from the same grid. Command: `--mode improved --train_noise_rate 0.4 --gamma_grid 1 2 4 8 16 --suffix noise_aug`; the `noisy` block (VGR off) is the augmented model, the `noisy_vgr` block is VGR on top of it.

*Caveat.* The training noise is the same family as the test noise, so this is the best case for augmentation. It does not show that augmentation helps against noise it has not seen; other noise types were not run.

*H5a (augmentation reduces the loss).* The augmented model's noisy Recall@20 is higher than the stress-test model's noisy Recall@20 in all three seeds. I expect it to close most of the 0.0108 gap to the clean baseline, but I do not predict how much.

*H5b (it costs clean accuracy).* The augmented model's clean Recall@20 is lower than the clean baseline's, because a share of the training text is wrong. I expect a loss smaller than the loss that noise causes to the baseline (0.0108) but I do not predict a size.

*H5c (VGR adds less on top).* The gain of VGR over the noisy condition (clean-trained: +0.0033 Recall@20) is smaller on the augmented model, because the gate has already learned to discount text that disagrees with the image.

*What would make me doubt it.* H5a: no gain, or a gain that does not have the same sign in all three seeds; then the clean-trained-baseline loss is not something training on noise removes, and the gate-shortcut story needs another explanation. H5b: a clean loss at or above 0.0108 means augmentation trades one failure for another. H5c: a VGR gain equal to or larger than +0.0033 means VGR and augmentation address different things. A negative result for any of them is a valid outcome and will be reported as such.

## H6 (added 2026-10-28, after the noise-augmented results of H5 were read, before the gate-probe runs)

H5 showed that models trained on corrupted warm-item text put almost all gate weight on text (w_t 0.986 on clean cold items, 0.917 on noisy ones, against 0.55 and 0.56 with clean training). The only gate numbers seen so far are on cold items. This probe asks what the gate does on the *warm* items it was trained on. Candidate explanation to test (untested when written): the fusion balance loss pulls the batch-mean gate weight to 0.5, so a gate that gives corrupted warm items little text weight can keep the mean at 0.5 only by giving the other items a lot.

*Setting.* Same data and seeds 2025 to 2027, `--mode stress_test --noise_rate 0.4 --gate_probe 1`, once with `--train_noise_rate 0.4` (suffix `gateprobe_aug`) and once with `--train_noise_rate 0` (suffix `gateprobe_clean`, the clean-trained control). After training, the best-validation model is probed with a fresh training-style corruption of 40% of the warm items (a mask no training epoch used) and the mean text weight `w_t` is measured on the corrupted warm items, on the other warm items, and on all warm items without noise. No VGR, no test-set tuning; the probe uses warm items only, so it touches no evaluation data.

*H6a (the gate discriminates on warm items).* In the augmented model, `w_t` on corrupted warm items is lower than on uncorrupted warm items by more than 0.2 in all three seeds.

*H6b (balance compensation).* In the augmented model, `w_t` on uncorrupted warm items is above 0.8, and the mean over all warm items without noise is above 0.8 too.

*H6c (control).* In the clean-trained control the gap between corrupted and uncorrupted warm items is below 0.05 and `w_t` is near 0.55 on both.

*What would make me doubt the balance explanation.* A gap below 0.1 in the augmented model (the gate does not tell corrupted warm items apart, so it is not compensating for them), or `w_t` above 0.9 on corrupted and uncorrupted warm items alike (the gate ignores the noise and the text weight is high for another reason). A control gap above 0.05 would mean the probe itself separates items for a reason other than training noise. Also as checks: the control's clean metrics must equal the core baseline runs of the same seed, and the augmented runs' clean and noisy metrics must equal the earlier `noise_aug` runs, since training is deterministic for a given seed; a difference would mean the run is not repeatable.

A result against any of these is a valid outcome and will be reported as such.

## H7 (added 2026-10-29, after the gate-probe results of H6 were read, before the representation-probe runs)

H6 showed that in the noise-augmented model the text weight is 0.99 on cold items but about 0.44 on warm items, while the clean-trained control stays near 0.5 on both. The balance explanation failed, so the cause is open. Candidate explanation to test (untested when written): cold items' expert outputs lie outside the region of embedding space the gate saw in training, and the augmented gate extrapolates to a high text weight there.

*Setting.* Same seeds 2025 to 2027, `--mode stress_test --noise_rate 0.4 --repr_probe 1`, once with `--train_noise_rate 0.4` (suffix `repr_aug`) and once with 0 (suffix `repr_clean`). After training, on clean text and with the best-validation model, the probe takes the image and text expert outputs of every item (the 128-d vectors the gate reads, concatenated to 256-d) and measures: the mean norm of each; the Euclidean distance of each item's concatenated vector to the centroid of the warm items, averaged over warm items and over cold items, and their ratio cold over warm; the mean gate logit difference (text minus image) on each group; and the Spearman correlation across cold items between the distance to the warm centroid and `w_t`. Training is unchanged, no VGR, no test-set tuning, the probe reads no labels.

*H7a (cold items are far from the warm region).* In the augmented model the cold-over-warm distance ratio is at least 1.2 in all three seeds.

*H7b (the shift is a property of the data, not of augmentation).* In each seed the augmented model's ratio is within 20% of the control's ratio. If the augmented ratio is more than 1.2 times the control's, augmentation itself moved cold items away from the warm region.

*H7c (distance drives the text weight).* In the augmented model the Spearman correlation between a cold item's distance to the warm centroid and its `w_t` is above 0.3 in all three seeds.

*What would make me doubt the explanation.* A ratio below 1.1 in the augmented model (cold items sit as close to the warm centroid as warm items do, so the gate is not extrapolating in this sense), or a Spearman correlation below 0.1 (the text weight does not vary with distance across cold items). Distance to the warm centroid is one crude measure of "outside the training region"; a pass would not prove the explanation, and a failure would only rule out this measure. As in H6, the augmented runs' metrics must equal the earlier `noise_aug` runs and the control's must equal the earlier `gateprobe_clean` runs (same environment, deterministic), otherwise the run is not repeatable.

A result against any of these is a valid outcome and will be reported as such.

*Outcome (added 2026-10-29, after the runs).* H7a not supported: the cold-over-warm distance ratio is 0.72 in the augmented model (0.720, 0.719, 0.734), below the 1.1 set as a doubt condition, so cold items sit closer to the warm centroid, not farther. H7b supported but empty: the control's ratio is 0.78, within 20% of the augmented one. H7c not supported: the Spearman correlations are +0.08, +0.15 and +0.14 (above 0.3 was needed). The candidate explanation, in this measure, is not what the numbers show. Repeatability held: the six runs equal the earlier gate-probe runs bit for bit.

## H8 (added 2026-10-29, after H5 to H7 were read and after VGR's small gain was known, before any repair run)

VGR only lowers the text weight of a suspect item; it puts nothing in its place, and its gain is small (+0.0033 Recall@20 on noisy text). Two training-free repairs are tested on the three saved clean-trained checkpoints (PyTorch 2.9.1, equal to the core baselines; `--save_final`), evaluated on the same noisy cold-item text as before (hashtag, rate 0.4, mask seed = training seed).

*Setting.* Both repairs build a replacement for an item's text from the text of its k = 10 nearest *warm* items in image space (cosine on the normalised image features), `t_knn`, and change the text features before the model reads them; the model, its weights and its forward pass are unchanged. k is fixed in advance, not tuned.
- *Blind blend (M1):* `t' = (1 - lam) * t + lam * t_knn` for every cold item; `lam` from {0.25, 0.5, 0.75, 1.0}.
- *Gated repair (M2):* the agreement `a_i = cos(t_i, t_knn_i)` of the raw features; `lam_i = lam_max * sigmoid(-g * (a_i - median(a)))` over the cold items (the median needs no knowledge of the noise rate); `lam_max` from {0.5, 1.0}, `g` from {5, 10, 20}.
- Parameters are chosen on noisy *validation* data (validation mask = seed + 10000, NDCG@10, as for VGR's gamma); the test set is never used for tuning. VGR is re-evaluated on the same checkpoints with the same grid {1, 2, 4, 8, 16} as the comparison, and each repair is also reported followed by VGR (gamma reselected after the repair).
- *Controls:* M1 and M2 with k random warm items instead of the image neighbours (same `lam`, same grid), and M2 with the agreement scores permuted across items (fixed seed).
- *Conditions:* noisy text and clean text (a repair that damages clean text is a cost), Recall@20 and NDCG@20, paired per seed. *Resources:* time to build the repaired features and the extra memory, measured, and the forward time with and without repair.

*H8a (the gated repair helps more than VGR).* M2 raises noisy-text Recall@20 above the unrepaired model in all three seeds, and its mean gain exceeds VGR's mean gain measured in the same evaluation.

*H8b (the gain comes from the image-guided neighbours and from per-item detection).* M2 beats both its controls (random neighbours; permuted agreement) in all three seeds on noisy text.

*H8c (blind repair costs clean text, gated repair costs less).* On clean text, M1 loses more than 0.003 Recall@20 against the unrepaired model (mean over seeds) and M2 loses less than 0.002.

*H8d (the agreement score detects corrupted items).* The AUROC of `-a_i` for separating corrupted from uncorrupted cold items is at least 0.7 in all three seeds. For reference the same AUROC is computed for VGR's score (the cosine of the two expert outputs); no order between the two is predicted.

*H8e (cost).* Building the repaired features takes under 1 s per evaluation and under 100 MB extra memory; the forward time of the model is unchanged (within 5%).

*What would make me doubt the repair.* A mean M2 gain below VGR's; a control within 0.001 Recall@20 of M2 (the gain would not come from image guidance or detection); an AUROC below 0.6. Known weakness of the design: the corruption is the one used throughout this study, a mix of the item's text with one of five warm-item vectors, so a repair that works here need not work on real bad text; the same limit applies to VGR. A result against any of these is a valid outcome and will be reported as such.

*Outcome (added 2026-10-29, after the runs).* H8a supported: M2 beats the unrepaired model in all three seeds (+0.0077, +0.0010, +0.0068; mean +0.0052) and VGR's mean gain (+0.0033). H8b not supported: both image-guided repairs beat their random-neighbour controls in all three seeds (M1 by +0.0080, M2 by +0.0047), but M2 loses to its permuted-agreement control in all three (-0.0020), so the image-guided neighbours help and the per-item agreement score does not. H8c not supported: the blind blend M1 costs clean text only -0.0007 Recall@20 (more than 0.003 was predicted); M2 changes it by +0.0008. H8d not supported: the AUROC of the agreement score is 0.511, 0.559 and 0.708 (0.7 needed in all three; below the 0.6 doubt limit in two), while VGR's cosine reaches 0.999 in all three seeds. H8e supported: 0.60 s once for the neighbour search, 0.015 s to build the repaired features, 41 MB, forward pass 7.88 ms against 8.07 ms (+2.4%). The blind blend M1 was not a headline hypothesis and is the best method: +0.0078 over the unrepaired model in all three seeds, +0.0096 when followed by VGR (about 89% of the loss from corruption). The M2 random-neighbour control named above was left out of the first run by oversight and added afterwards on the same checkpoints; no other number changed. The unrepaired and VGR numbers equal the saved runs exactly.

## H9 (added 2026-10-29, after the H8 results were read, before any H9 run)

H8 found that VGR's cosine of the two expert outputs separates corrupted from clean cold items almost perfectly (AUROC 0.999 in all three seeds, measured on the test corruption as a diagnostic, nothing was tuned on it) while the agreement score used by M2 does not (0.51 to 0.71), and that blending image-neighbour text into every item (M1) is the best repair so far (+0.0078 Recall@20). This design choice, to take the cosine as the detector, was made after seeing that AUROC. Untested candidate: repairing only the items the cosine flags keeps M1's gain on corrupted items and avoids touching clean ones, so it does better than M1.

*Setting.* Same three saved checkpoints, same noisy test text and validation protocol as H8 (parameters chosen on noisy validation data, NDCG@10; the test set is never used for tuning). *M3 (cosine-gated repair):* the cosine `c_i` of the image and text expert outputs of cold item i, computed by one forward pass on the text being evaluated; `lam_i = lam_max * sigmoid(-g * (c_i - Q_q(c)))` where `Q_q` is the q-quantile of `c` over the cold items, and the neighbour text `t_knn` is the same as in H8 (k = 10); `t' = (1 - lam_i) * t + lam_i * t_knn`. Grid: `lam_max` in {0.75, 1.0}, `q` in {0.3, 0.4, 0.5}, `g` in {20, 50}. Like VGR's gamma, `q` is a setting chosen on a validation set that carries noise at the stated rate. *Controls:* M3 with ten random warm items in place of the image neighbours; M3 with the cosine scores permuted across items (fixed seed); and, as an upper bound only (it reads the true corruption mask, so it is not a method), the blind blend applied only to the truly corrupted items with `lam` from {0.25, 0.5, 0.75, 1.0} chosen on validation. M1 and VGR are the references; M3 followed by VGR is also reported.

*H9a (detection helps).* M3's mean noisy-text Recall@20 gain over the unrepaired model exceeds M1's (+0.0078), and M3 is above M1 in at least two of three seeds.

*H9b (the gain comes from the image neighbours and from the flagged items).* M3 beats both controls (random neighbours; permuted cosine) in all three seeds on noisy text.

*H9c (clean text is safe).* On clean text M3 changes Recall@20 by less than 0.002 in absolute mean.

*H9d (cost).* Computing the cosine adds under 0.1 s per evaluation and under 5 MB over M1.

*What would make me doubt the idea.* M3 not above M1 on average, or a control within 0.001 of M3, or M3 reaching less than the oracle by more than half of M1's gain (then the cosine flags the right items but the threshold is poor). A result against any of these is a valid outcome and will be reported as such.

*Outcome (added 2026-10-29, after the runs).* H9a supported: M3 gains +0.0094 Recall@20 over the unrepaired model against M1's +0.0078, and is above M1 in all three seeds (+0.0029, +0.0005, +0.0011; the margin in seed 2026 is small). H9b supported: M3 beats the random-neighbour control (+0.0156, +0.0087, +0.0108) and the permuted-cosine control (+0.0049, +0.0021, +0.0043) in all three seeds; the random-neighbour version loses accuracy (-0.0023 on noisy, -0.0056 on clean text). H9c supported: on clean text M3 changes Recall@20 by +0.0016 on average (-0.0004, +0.0034, +0.0017). H9d supported: the cosine takes 0.017 s and 0.005 MB. The 'oracle' was meant as an upper bound but is not one: M3 beats it in all three seeds (+0.0022, +0.0036, +0.0020) and so does the blind blend M1 (+0.0078 against +0.0068), so repairing exactly the corrupted items is not the best use of the neighbour text; this was predicted wrongly. M3 selected q = 0.5 in all three seeds, not the true corruption rate 0.4. M3 followed by VGR reaches +0.0105 (about 96% of the loss, M3 alone about 86%). Earlier results in the same files did not change. The cosine was chosen as the detector after the AUROC of H8 was seen, and q is chosen on a validation set with noise at the stated rate, as VGR's gamma is.

## Protocol commitments

- `γ` is chosen on the validation set (noisy validation, grid {1, 2, 4, 8, 16}); the test set is never used to tune the method. A test-set γ sweep, if shown, is a sensitivity table reported after selection.
- Best epoch is selected on clean validation NDCG for all models.
- Same seed ⇒ same corruption mask across all conditions.
- Unfavorable cases (items/users where VGR hurts) are reported alongside favorable ones.
- A negative result for H1–H4 is a valid outcome and will be reported as such.

## Discrepancy investigation (written before these runs; the baseline results were already known)

The clean baseline matches the published Baby numbers except Rec@10 (0.0487 against 0.0512, three seeds). Two candidate explanations are checked. Neither changes the main baseline or any reported table; the runs are diagnostic only and carry their own file suffix.

*D1: seed variation.* Seven more baseline seeds (2028 to 2034, ten in total). If the published value lies within two standard errors of the ten-seed mean, seed variation is consistent with the gap. If it lies outside, seed variation alone is not enough. The share of seeds at or above the published value is reported as well.

*D2: scale of the alignment loss.* The paper's Eq. 8 is a squared L2 norm, the code uses a mean over all elements, which is smaller by the embedding size (128). The paper's weights {1, 0.1, 0.01} therefore correspond to `--align_coeff` {128, 12.8, 1.28} in the code, against the default 1. Baseline runs with these three values, seeds 2025 to 2027. If one of them brings the mean Rec@10 within 0.001 of the published value without lowering Rec@20 or NDCG, the scale is a candidate explanation. No value is adopted as the main baseline: that would need a validation-based choice and a re-run of every condition.

A null result for either is reported as such.
