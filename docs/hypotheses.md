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

## H10 (added 2026-10-29, after the H9 results were read, before any H10 run)

H8 and H9 tested the repairs on one noise family, the hashtag mix at rate 0.4, with validation data carrying the same noise. Two questions remain: do M1 and M3 work under other noise, and do they survive a validation set whose noise differs from the test noise, which is the realistic case. No new training: the same three saved checkpoints, all methods and controls of H8 and H9, parameters chosen on noisy validation data only, the test set never used for tuning, `k = 10` unchanged.

*Setting 1 (generalisation, 9 settings).* Noise type in {hashtag, shuffle, gaussian} x rate in {0.2, 0.4, 0.6}, validation noise of the same type and rate as the test noise (hashtag 0.4 is the H8 and H9 result). `shuffle` replaces the text of a corrupted item by that of a random warm item; `gaussian` adds noise with standard deviation 0.8 times that of the text features (the defaults of the simulator).

*Setting 2 (mismatch, 4 settings).* Validation noise is always hashtag at rate 0.4 (a validation set that carries the noise assumed when the method was set up); the test noise is hashtag 0.2, hashtag 0.6, shuffle 0.4 and gaussian 0.4. Settings are compared with the matched result of the same test noise from Setting 1.

*H10a (M3 generalises).* In at least 8 of the 9 matched settings M3 beats the unrepaired model in all three seeds, and its mean gain exceeds VGR's mean gain.

*H10b (the gain comes from the image neighbours and the flagged items).* In every matched setting M3's mean is above both its random-neighbour and permuted-cosine controls.

*H10c (M3 against the blind blend).* M3's mean is at least M1's in at least 6 of the 9 matched settings.

*H10d (more noise, more to repair).* For each noise type, M3's mean gain at rate 0.6 exceeds its mean gain at rate 0.2.

*H10e (clean text stays safe).* In every matched setting the mean change of clean-text Recall@20 under M3 lies within +-0.003.

*H10f (mismatch).* In each of the four mismatch settings M3 beats the unrepaired model in all three seeds and keeps at least 70% of its matched mean gain.

*What would make me doubt the repairs.* M3 failing to beat the unrepaired model in all three seeds in two or more matched settings; a control within 0.001 of M3 in a setting; or less than half of the matched gain kept under mismatch. The detection AUROC of VGR's cosine is reported per setting, with no prediction. Where a method fails, the failing setting is reported next to the successes. A result against any of these is a valid outcome and will be reported as such.

*Outcome (added 2026-10-29, after the runs).* H10a supported: in all nine matched settings M3 beats the unrepaired model in all three seeds and its mean gain exceeds VGR's. H10b supported on the means (M3 is above both controls in every matched setting), but the doubt condition 'a control within 0.001 of M3' is met once: gaussian 0.4, where M3 (+0.0039) is only +0.0004 above the permuted-cosine control (+0.0035). H10c not supported: M3's mean is at least M1's in only 5 of 9 settings; the blind blend M1 is better at rate 0.6 (hashtag +0.0135 against +0.0101, shuffle +0.0236 against +0.0224, gaussian +0.0077 against +0.0075) and for gaussian 0.4. H10d supported for all three types (M3 gain at rate 0.6 against 0.2: hashtag +0.0101/+0.0060, shuffle +0.0224/+0.0094, gaussian +0.0075/+0.0048). H10e supported: M3 changes clean-text Recall@20 by +0.0009 to +0.0019 in every setting. H10f supported: under a validation noise fixed at hashtag 0.4, M3 beats the unrepaired model in all three seeds in all four mismatch settings and keeps 99% to 116% of its matched gain (hashtag 0.2: 99%, hashtag 0.6: 104%, shuffle 0.4: 101%, gaussian 0.4: 116%). The AUROC of VGR's cosine is 0.993 or higher in every setting. M3 followed by VGR reaches 71% to 124% of the loss from corruption (at rate 0.2: 99% to 124%; at rate 0.6: 71% to 83%); above 100% means the repaired model on noisy text is above the unrepaired model on clean text.

## H11 (added 2026-10-29, after the H10 results were read, before the seed 2028 and 2029 runs)

H8 to H10 rest on three seeds, and the M3 minus M1 differences are as small as +0.0005 per seed. Two more clean baselines (seeds 2028 and 2029, PyTorch 2.9.1, `--save_final 1`, results in `results/ckpt291/`) give five seeds for the main setting (hashtag, rate 0.4, validation noise matched). Seeds 2028 and 2029 were not used in any earlier repair run. The seeds were chosen before training and are not replaced if the outcome is unfavourable. Selection of the repair parameters on noisy validation data is unchanged, no value is re-tuned.

*H11a (M3 holds).* M3 beats the unrepaired model in both new seeds, and M3 plus VGR beats VGR alone in both.

*H11b (controls).* M3 is above both its controls (random neighbours, permuted cosine) in both new seeds.

*H11c (M3 against M1).* Over five seeds M3 is at least M1 in four. I expect this to be close: at rate 0.4 the three-seed differences were +0.0029, +0.0005 and +0.0011. The paired bootstrap interval (10000 resamples over seeds) of the mean M3 minus M1 difference is reported with no prediction about whether it excludes zero; with five seeds it may well include it, and then the advantage of M3 over M1 at rate 0.4 is not established.

*What would make me doubt the repair.* M3 not beating the unrepaired model in a new seed, or a control within 0.001 of M3 in a new seed. A result against any of these is a valid outcome and will be reported as such. If the baseline of a new seed does not train normally (best epoch at the first evaluation, or Recall@20 below 0.06) the run is reported and the cause looked for, not dropped.

*Outcome (added 2026-10-29, after the runs).* Disclosure first: the clean baselines of seeds 2028 and 2029 were already known from the D1 seed-extension runs (made in the PyTorch 2.14.1 environment); retrained in 2.9.1 they are identical (best epochs 320 and 730; Recall@20 0.07673 and 0.07393), so the baselines are not new information, but no repair had been run on them. Both trained normally. H11a supported: M3 beats the unrepaired model in both new seeds (Recall@20 +0.0078 and +0.0090), and M3 followed by VGR beats VGR alone (0.0712 against 0.0651 and 0.0770 against 0.0689). H11b supported: M3 is above the random-neighbour control by +0.0076 and +0.0092 and above the permuted-cosine control by +0.0056 and +0.0038, far from the 0.001 doubt limit. H11c supported by the count: over five seeds M3 is at least M1 in four (the exception is seed 2028, -0.0001); the per-seed differences are +0.0029, +0.0005, +0.0011, -0.0001, +0.0027, mean +0.0014, and the bootstrap interval over the five seeds (10000 resamples, seed 0) is [+0.0004, +0.0025], which excludes zero. Five seeds is a thin basis for a bootstrap and the lower end is close to zero, so we read it as a small advantage of M3 over M1 at rate 0.4, not as an established one. Five-seed means (Recall@20, hashtag 0.4): unrepaired 0.0661, VGR +0.0028, M1 +0.0075, M3 +0.0090, M3 then VGR +0.0099 (clean unrepaired 0.0767, loss 0.0106, so M3 recovers about 85% and M3 then VGR about 93%); M3 changes clean text by +0.0019 (4/5 seeds). Nothing in the doubt conditions occurred.

## H12 and H13 (added 2026-10-30, after H11 and before any of the runs below; two of the three open questions of the report)

Neither question concerns the main claims; both narrow an unexplained point. Neither result changes a main table.

### H12: where does the Rec@10 gap come from (published 0.0512, measured 0.0487 over three seeds)?
The released code picks the best epoch by validation NDCG@10 and evaluates on the test set only at those epochs. D1 (more seeds) and D2 (alignment-loss scale) did not explain the gap. New check: retrain the three clean baselines (seeds 2025 to 2027, PyTorch 2.9.1) with `--curve 1`, which records validation and test metrics at every evaluation epoch and changes neither training nor selection; the best epochs must equal the saved ones (480, 410, 470), otherwise the curve run is not used.

*H12a (the epoch rule can produce it).* If the epoch is chosen by test Recall@10 (an oracle that uses the test set; diagnostic only, never adopted), the three-seed mean test Recall@10 reaches at least 0.0512 minus 0.001. This would show that a selection that looks at the test set, or a best-run report, is enough to produce the published value; it would not show that the authors did it.

*H12b (another validation rule does not).* If the epoch is chosen by validation Recall@10, validation NDCG@20 or validation Recall@20 instead, the mean test Recall@10 stays within 0.001 of 0.0487, so the gap is not a matter of which validation metric selects the epoch.

*What would change my reading.* An oracle mean below 0.0502 means even test-set epoch picking does not reach the published value on these checkpoints, and the gap then points to data or features. Whatever the outcome, the gap is described as 'not identified' unless a rule is found that reaches the published value without the test set.

### H13: why does the gate choose text on cold items in the noise-augmented model?
Known (H6, H7): in the augmented model the gate puts weight 0.99 on text for cold items and 0.44 on warm items; the clean-trained control stays near 0.5 for both. The gate reads the concatenation of the two expert outputs, and only warm items receive training loss, so cold items are out of distribution for the gate. Cold expert outputs have smaller norms (H7). Proposed mechanism: small-norm inputs sit near the origin of the gate's input space, where the gate decides mainly by its bias, and in the augmented model that bias leans to text. Test on saved checkpoints (augmented and clean-trained, seeds 2025 and 2026, PyTorch 2.9.1, clean text), with no training beyond the checkpoints and no new parameters.

*H13a (bias leans to text).* With both expert outputs set to zero, the text-minus-image gate logit of the augmented model is at least +3, and at least 3 higher than the clean-trained control's.

*H13b (norm matters causally).* Scaling each cold item's two expert outputs to the mean norm of the warm items lowers the mean cold text weight of the augmented model from about 0.99 to below 0.8.

*H13c (control).* The same scaling changes the clean-trained control's cold text weight by less than 0.1.

*H13d (norm is not all).* If H13b fails, replacing a cold item's two expert outputs by those of a random warm item (a full replacement) is used to ask whether the gate responds to the position and not only the norm; reported with no prediction.

*What would make me drop the mechanism.* H13a or H13b failing: the norm explanation is then rejected, which narrows the cause but does not name another. The result is a statement about two seeds of one model and is described that way.

*Outcome of H12 (added 2026-10-30, after the runs).* The three curve runs equal the saved baselines (best epochs 480, 410, 470; identical test results). H12a supported, narrowly: epoch chosen by test Recall@10 gives a three-seed mean test Recall@10 of 0.0502 (0.05024), exactly the threshold 0.0502 and still 0.0010 below the published 0.0512; per seed 0.0516, 0.0491, 0.0500, so only seed 2025 passes the published value. Choosing by test NDCG@10 or test Recall@20 gives 0.0501 and 0.0500. H12b supported: validation Recall@10 gives 0.0491, validation NDCG@20 0.0487, validation Recall@20 0.0482 (the released rule 0.0487), all within 0.001 of 0.0487. The doubt condition (oracle below 0.0502) did not occur, by a margin of 0.00004. Reading: no validation-based rule closes the gap; a rule that uses the test set closes about 60% of it (0.0015 of 0.0025), which makes a published single run with a test-chosen epoch consistent with the numbers, without any evidence that this happened. The gap is described as not identified.

*Outcome of H13 (added 2026-10-30, after the runs).* Seeds 2025 and 2026, augmented and clean-trained checkpoints (PyTorch 2.9.1). H13a not supported: the text-minus-image logit at zero input is +0.181 and +0.219 in the augmented model (control +0.060, +0.054), far from +3. H13b not supported: scaling each cold item's expert outputs to the warm mean norm leaves w_t at 0.993 and 0.991 (0.991 and 0.981 before; the threshold was below 0.8). H13c supported: the control's w_t moves from 0.551 to 0.551 and 0.549. H13d (no prediction): replacing both expert outputs of a cold item by those of a random warm item gives w_t 0.436 and 0.434 (warm level 0.437 and 0.435), so the gate responds to the content of the cold items' expert outputs. Exploratory, added after seeing this: replacing only the image output gives 0.315 and 0.359, only the text output 0.953 and 0.908, so the cold items' image-side expert outputs drive the text preference. The mechanism proposed (bias plus small norm) is rejected; the cause of the image-side outputs looking unfamiliar to the gate is not found. Two seeds, one augmentation design.

## H14 to H17 (added 2026-10-30, after the H12 and H13 results were read, before any of the runs below)

Four follow-ups on open points. Seeds 2030 to 2034 were used before only for clean baselines without checkpoints (D1, PyTorch 2.14.1); no repair or probe has been run on them.

### H14: two more candidate explanations of the Rec@10 gap (training-free, on the saved checkpoints and curves)
*H14a (candidate set).* The evaluation ranks a user's test items among all 1,410 cold items (validation and test items). Ranking among the test items only would raise Recall@10 and Recall@20 together; I predict both rise by more than 20% in relative terms, which would exclude this protocol, since the published Rec@20 (0.0771) already matches the measured one under the 1,410-item candidate set.
*H14b (best epoch per metric).* If each published number is the best over the evaluation epochs of one run, the per-metric maxima of one seed (or the three-seed mean of maxima) would match the published vector (0.0512, 0.0771, 0.0292, 0.0363). I predict that no seed and not the mean of the maxima is within 0.001 (Recall) and 0.0005 (NDCG) of all four numbers, because the maxima of Recall@20 of the better seeds lie far above 0.0771. Neither is adopted as a baseline; both are diagnostics.

### H15: where does the gain of the cosine-gated repair M3 come from?
On the five checkpoints of seeds 2025 to 2029 (hashtag 0.4, noisy text, the parameters of M3 as selected on validation data), M3 is applied only to the truly corrupted cold items, then only to the uncorrupted ones.
*H15a.* The corrupted-only gain is at least 70% of the full M3 gain.
*H15b.* The clean-only gain is positive but below +0.003 Recall@20 in every seed.
*Reading rule.* If the clean-only gain is at least 30% of the full gain, a large part of M3's benefit is neighbour text on items that were not corrupted, which would explain why the repair can exceed the clean unrepaired model.

### H16: ten seeds for M3 against M1 (hashtag 0.4)
Clean baselines of seeds 2030 to 2034 retrained in PyTorch 2.9.1 with checkpoints, then evaluated exactly like seeds 2025 to 2029.
*H16a.* M3 beats the unrepaired model in all five new seeds.
*H16b.* M3 is at least M1 in at least three of the five new seeds.
*H16c.* The mean difference M3 minus M1 over the ten seeds is between +0.0005 and +0.0025. I make no prediction on whether its bootstrap interval excludes zero. A ten-seed mean below +0.0005 means the advantage of M3 over M1 at rate 0.4 is not supported.

### H17: is the collapse caused by an item having no training interactions?
Noise-augmented training (rate 0.4) and a clean-trained control, seeds 2025 and 2026, with 10% of the warm items drawn at random and removed from the training interactions (they stay in the item set; validation and test are unchanged). The gate's mean text weight on clean text is read on three groups: trained warm items, held-out warm items, cold items.
*H17a.* In the augmented model the held-out warm items behave like cold items: mean text weight at least 0.8, against about 0.43 on trained warm items.
*H17b.* In the clean-trained control all three groups stay between 0.4 and 0.65.
*Reading.* H17a holding means the collapse follows from not being in the training loss, not from being a cold item; failing, that something else separates cold from held-out warm items (for example that cold items were also never in the item graph used for negative sampling). Two seeds, one augmentation design.

## H18 (added 2026-10-30, after the H14 to H16 setup and before any run on a second dataset)

The report's main limit is one dataset. The LATTICE authors also publish Amazon Sports and Clothing (same folder as the Baby data used here). The Baby interactions of MAMEX's processed data are exactly the union of LATTICE's 5-core train, validation and test files (160,792 pairs, identical sets), and MAMEX's cold split divides the items into warm, validation-cold and test-cold items (705 each for the last two). A second dataset is therefore built the same way from LATTICE's files: 20% of the items drawn at random with a fixed seed are cold, half of them validation and half test; the interactions of warm items form the training file and those of the cold items the validation and test files. This is our construction, not the authors' split (which cannot be reproduced without their seed), so the second dataset is a robustness check of the findings, not a reproduction.

On three clean-trained seeds (2025 to 2027, one checkpoint each, PyTorch 2.9.1), evaluated with corruption of 40% of the cold items' text (hashtag), the same protocol and selection rules as on Baby (parameters on noisy validation data only; no setting tuned on the test set):
*H18a.* The corruption lowers Recall@20 in all three seeds.
*H18b.* VGR raises Recall@20 above the unrepaired model in all three seeds.
*H18c.* The cosine-gated repair M3 beats the unrepaired model and both of its controls (random neighbours, permuted cosine) in all three seeds.
*H18d.* M3 beats VGR in all three seeds, as on Baby (+0.0062 mean difference there).
*H18e.* M3 is at least as good as the blind blend M1 in at least two of three seeds. I expect this to be close, as on Baby.
*What would make me doubt the findings.* M3 not beating the unrepaired model in a seed, a control within 0.001 of M3, or VGR failing to beat the unrepaired model in two or more seeds. Any of these is reported as a failure on the second dataset, next to the Baby results. The reproduction of MAMEX's published numbers is not attempted on the second dataset.

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
