# Pre-registered hypotheses

Written before any full-training result was read. Course rule: the rationale is stated *before* results are examined (YZM5110 §4).

Written on 2026-10-16, before the first full-training results of 2026-10-17. Until then the only output seen was a 3-epoch smoke run (not trained, not interpreted); the seed-2025 grid was running but unread. The repository holds the final state as a single commit, so this date is the author's statement, not a commit timestamp.

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
