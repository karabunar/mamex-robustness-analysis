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
