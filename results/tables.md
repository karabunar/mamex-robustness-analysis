## Published (MAMEX paper, Amazon Baby; not measured here)

| Rec@10 | Rec@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|
| 0.0512 | 0.0771 | 0.0292 | 0.0363 |

## Measured (cold items, test set, noise=hashtag rate 0.4; mean ± sample std over seeds)

| Condition | n | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|
| Clean baseline | 3 | 0.049 ± 0.002 | 0.078 ± 0.004 | 0.029 ± 0.002 | 0.037 ± 0.002 |
| Stress test (noisy text) | 3 | 0.042 ± 0.002 | 0.067 ± 0.002 | 0.024 ± 0.002 | 0.031 ± 0.001 |
| MAMEX-VGR (noisy text) | 3 | 0.043 ± 0.002 | 0.070 ± 0.002 | 0.024 ± 0.001 | 0.032 ± 0.001 |
| MAMEX-VGR (clean text) | 3 | 0.048 ± 0.002 | 0.077 ± 0.005 | 0.028 ± 0.002 | 0.036 ± 0.002 |
| VGR control: shuffled cosine (noisy text) | 3 | 0.042 ± 0.002 | 0.068 ± 0.002 | 0.024 ± 0.002 | 0.031 ± 0.001 |
| VGR control: constant factor (noisy text) | 3 | 0.042 ± 0.002 | 0.068 ± 0.002 | 0.024 ± 0.001 | 0.031 ± 0.001 |

## Paired differences per seed (A - B)

| A - B | metric | per-seed differences | mean | seeds with A > B |
|---|---|---|---|---|
| noisy_vgr - noisy | recall@20 | +0.0043, +0.0023, +0.0034 | +0.0033 | 3/3 |
| noisy_vgr - noisy | ndcg@20 | +0.0018, +0.0008, +0.0011 | +0.0012 | 3/3 |
| noisy_vgr - noisy_vgr_shuffled | recall@20 | +0.0018, +0.0011, +0.0021 | +0.0017 | 3/3 |
| noisy_vgr - noisy_vgr_shuffled | ndcg@20 | +0.0009, +0.0005, +0.0007 | +0.0007 | 3/3 |
| noisy_vgr - noisy_vgr_constant | recall@20 | +0.0020, +0.0012, +0.0024 | +0.0018 | 3/3 |
| noisy_vgr - noisy_vgr_constant | ndcg@20 | +0.0009, +0.0006, +0.0008 | +0.0008 | 3/3 |
| clean_vgr - clean | recall@20 | +0.0001, -0.0011, -0.0000 | -0.0004 | 1/3 |
| clean_vgr - clean | ndcg@20 | +0.0000, -0.0005, -0.0001 | -0.0002 | 1/3 |

## Gate weights (improved runs, corrupted cold items)

| Setting | w_v | w_t |
|---|---|---|
| clean | 0.450 ± 0.002 | 0.550 ± 0.002 |
| noisy | 0.437 ± 0.004 | 0.563 ± 0.004 |
| noisy_vgr | 0.584 ± 0.007 | 0.416 ± 0.007 |

## Cost and selected gamma

| forward ms (plain) | forward ms (VGR) | GPU memory MB | train seconds | selected gamma per seed |
|---|---|---|---|---|
| 6.47 | 6.83 | 471 | 1127 | [1.0, 1.0, 1.0] |

## Sensitivity (single seed 2025; Recall@20; not an average)

| noise rate | clean | noisy | noisy + VGR | gamma selected on validation |
|---|---|---|---|---|
| 0.2 | 0.0819 | 0.0752 | 0.0770 | 1.0 |
| 0.4 | 0.0819 | 0.0667 | 0.0710 | 1.0 |
| 0.6 | 0.0819 | 0.0618 | 0.0659 | 1.0 |
| 0.8 | 0.0819 | 0.0523 | 0.0593 | 1.0 |

| gamma (set by hand, test sensitivity only) | noisy | noisy + VGR | shuffled cosine | constant factor | clean | clean + VGR |
|---|---|---|---|---|---|---|
| 0.25 | 0.0667 | 0.0721 | 0.0714 | 0.0714 | 0.0819 | 0.0819 |
| 0.5 | 0.0667 | 0.0717 | 0.0710 | 0.0709 | 0.0819 | 0.0818 |
| 1.0 | 0.0667 | 0.0710 | 0.0692 | 0.0690 | 0.0819 | 0.0820 |
| 2.0 | 0.0667 | 0.0708 | 0.0673 | 0.0679 | 0.0819 | 0.0819 |
| 4.0 | 0.0667 | 0.0695 | 0.0667 | 0.0668 | 0.0819 | 0.0819 |
| 8.0 | 0.0667 | 0.0691 | 0.0666 | 0.0669 | 0.0819 | 0.0819 |
| 16.0 | 0.0667 | 0.0678 | 0.0666 | 0.0668 | 0.0819 | 0.0819 |

## Noise-augmented training (hashtag noise on warm items' text in every training epoch, rate 0.4; cold-item test; seeds [2025, 2026, 2027]; mean ± sample std)

| Condition | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|
| Noise-augmented model, clean text | 0.044 ± 0.002 | 0.072 ± 0.001 | 0.025 ± 0.002 | 0.032 ± 0.002 |
| Noise-augmented model, noisy text | 0.036 ± 0.001 | 0.057 ± 0.001 | 0.020 ± 0.001 | 0.026 ± 0.001 |
| Noise-augmented model, noisy text + VGR | 0.036 ± 0.001 | 0.057 ± 0.001 | 0.020 ± 0.000 | 0.026 ± 0.000 |
| Noise-augmented model, clean text + VGR | 0.044 ± 0.002 | 0.072 ± 0.002 | 0.025 ± 0.002 | 0.032 ± 0.002 |

### Paired differences per seed

| Comparison | metric | per-seed differences | mean | seeds > 0 |
|---|---|---|---|---|
| noisy: noise-augmented - stress test | recall@20 | -0.0106, -0.0082, -0.0111 | -0.0100 | 0/3 |
| noisy: noise-augmented - stress test | ndcg@20 | -0.0036, -0.0045, -0.0057 | -0.0046 | 0/3 |
| clean: noise-augmented - clean baseline | recall@20 | -0.0079, -0.0026, -0.0050 | -0.0052 | 0/3 |
| clean: noise-augmented - clean baseline | ndcg@20 | -0.0043, -0.0035, -0.0043 | -0.0040 | 0/3 |
| noisy: clean baseline - noise-augmented (loss left) | recall@20 | +0.0259, +0.0167, +0.0200 | +0.0209 | 3/3 |
| noisy: clean baseline - noise-augmented (loss left) | ndcg@20 | +0.0123, +0.0086, +0.0104 | +0.0104 | 3/3 |
| VGR gain on noisy text: noise-augmented - clean-trained | recall@20 | -0.0042, -0.0016, -0.0034 | -0.0031 | 0/3 |
| VGR gain on noisy text: noise-augmented - clean-trained | ndcg@20 | -0.0016, -0.0004, -0.0011 | -0.0010 | 0/3 |

Selected gamma per seed: [16.0, 2.0, 8.0]

| Gate (corrupted cold items) | w_v | w_t |
|---|---|---|
| clean | 0.014 ± 0.004 | 0.986 ± 0.004 |
| noisy | 0.083 ± 0.048 | 0.917 ± 0.048 |
| noisy_vgr | 0.099 ± 0.065 | 0.901 ± 0.065 |

## Gate probe: mean text weight w_t on warm items (seeds [2025, 2026, 2027]; mean ± sample std)

| Model | warm, corrupted | warm, not corrupted | warm, no noise | cold, no noise | gap (not corrupted - corrupted) per seed |
|---|---|---|---|---|---|
| Noise-augmented training | 0.313 ± 0.010 | 0.435 ± 0.001 | 0.436 ± 0.001 | 0.987 ± 0.004 | +0.120, +0.114, +0.132 |
| Clean training (control) | 0.495 ± 0.018 | 0.487 ± 0.001 | 0.487 ± 0.001 | 0.549 ± 0.002 | -0.029, +0.008, -0.005 |

Same-environment comparison (both models from the gate-probe runs; paired per seed):

| Comparison | metric | per-seed differences | mean | seeds > 0 |
|---|---|---|---|---|
| clean text: noise-augmented - clean-trained | recall@20 | -0.0069, -0.0043, -0.0050 | -0.0054 | 0/3 |
| clean text: noise-augmented - clean-trained | ndcg@20 | -0.0041, -0.0048, -0.0042 | -0.0044 | 0/3 |
| noisy text: noise-augmented - clean-trained | recall@20 | -0.0123, -0.0097, -0.0096 | -0.0105 | 0/3 |
| noisy text: noise-augmented - clean-trained | ndcg@20 | -0.0046, -0.0054, -0.0056 | -0.0052 | 0/3 |

## Text repair on saved clean-trained checkpoints (seeds [2025, 2026, 2027]; Recall@20; mean ± sample std)

| Method | noisy text | clean text |
|---|---|---|
| Unrepaired | 0.067 ± 0.002 | 0.078 ± 0.004 |
| VGR | 0.070 ± 0.002 | 0.077 ± 0.005 |
| M1 blind blend | 0.075 ± 0.002 | 0.077 ± 0.004 |
| M1, random neighbours | 0.067 ± 0.003 | 0.076 ± 0.005 |
| M2 gated repair | 0.072 ± 0.005 | 0.078 ± 0.003 |
| M2, random neighbours | 0.067 ± 0.003 | 0.077 ± 0.005 |
| M2, permuted agreement | 0.074 ± 0.004 | 0.078 ± 0.002 |
| M3 cosine-gated repair | 0.076 ± 0.003 | 0.079 ± 0.002 |
| M3, random neighbours | 0.064 ± 0.002 | 0.072 ± 0.005 |
| M3, permuted cosine | 0.072 ± 0.001 | 0.079 ± 0.003 |
| Oracle (true mask; upper bound only) | 0.074 ± 0.004 | 0.078 ± 0.004 |
| M1 blind blend, then VGR | 0.076 ± 0.002 | 0.078 ± 0.003 |
| M2 gated repair, then VGR | 0.074 ± 0.004 | 0.079 ± 0.004 |
| M3 cosine-gated repair, then VGR | 0.077 ± 0.003 | 0.079 ± 0.003 |

Paired differences per seed (Recall@20):

| Comparison | per-seed differences | mean | seeds > 0 |
|---|---|---|---|
| noisy: vgr - unrepaired | +0.0043, +0.0023, +0.0034 | +0.0033 | 3/3 |
| noisy: m1 - unrepaired | +0.0084, +0.0074, +0.0077 | +0.0078 | 3/3 |
| noisy: m1_random - unrepaired | -0.0003, -0.0012, +0.0010 | -0.0002 | 1/3 |
| noisy: m2 - unrepaired | +0.0077, +0.0010, +0.0068 | +0.0052 | 3/3 |
| noisy: m2_random - unrepaired | -0.0007, -0.0005, +0.0025 | +0.0004 | 1/3 |
| noisy: m2_shuffled - unrepaired | +0.0100, +0.0045, +0.0071 | +0.0072 | 3/3 |
| noisy: m3 - unrepaired | +0.0114, +0.0080, +0.0088 | +0.0094 | 3/3 |
| noisy: m3_random - unrepaired | -0.0043, -0.0007, -0.0020 | -0.0023 | 0/3 |
| noisy: m3_shuffled - unrepaired | +0.0065, +0.0059, +0.0044 | +0.0056 | 3/3 |
| noisy: oracle - unrepaired | +0.0092, +0.0043, +0.0068 | +0.0068 | 3/3 |
| clean: vgr - unrepaired | +0.0001, -0.0011, -0.0000 | -0.0004 | 1/3 |
| clean: m1 - unrepaired | -0.0030, -0.0008, +0.0015 | -0.0007 | 1/3 |
| clean: m1_random - unrepaired | +0.0003, -0.0013, -0.0041 | -0.0017 | 1/3 |
| clean: m2 - unrepaired | -0.0005, +0.0014, +0.0015 | +0.0008 | 2/3 |
| clean: m2_random - unrepaired | +0.0007, -0.0011, -0.0005 | -0.0003 | 1/3 |
| clean: m2_shuffled - unrepaired | -0.0018, +0.0025, +0.0015 | +0.0007 | 2/3 |
| clean: m3 - unrepaired | -0.0004, +0.0034, +0.0017 | +0.0016 | 2/3 |
| clean: m3_random - unrepaired | -0.0047, -0.0050, -0.0069 | -0.0056 | 0/3 |
| clean: m3_shuffled - unrepaired | +0.0009, +0.0032, +0.0012 | +0.0017 | 3/3 |
| clean: oracle - unrepaired | +0.0000, +0.0000, +0.0000 | +0.0000 | 0/3 |
| noisy_plus_vgr: m1 - unrepaired | +0.0107, +0.0095, +0.0087 | +0.0096 | 3/3 |
| noisy_plus_vgr: m2 - unrepaired | +0.0100, +0.0042, +0.0075 | +0.0072 | 3/3 |
| noisy_plus_vgr: m3 - unrepaired | +0.0126, +0.0088, +0.0100 | +0.0105 | 3/3 |
| clean_plus_vgr: m1 - unrepaired | -0.0022, +0.0006, +0.0019 | +0.0001 | 2/3 |
| clean_plus_vgr: m2 - unrepaired | -0.0006, +0.0007, +0.0025 | +0.0009 | 2/3 |
| clean_plus_vgr: m3 - unrepaired | -0.0002, +0.0021, +0.0017 | +0.0012 | 2/3 |
| noisy: m1 - m1_random | +0.0087, +0.0087, +0.0067 | +0.0080 | 3/3 |
| noisy: m2 - m2_random | +0.0084, +0.0015, +0.0043 | +0.0047 | 3/3 |
| noisy: m2 - m2_shuffled | -0.0024, -0.0035, -0.0003 | -0.0020 | 0/3 |
| noisy: m3 - m3_random | +0.0156, +0.0087, +0.0108 | +0.0117 | 3/3 |
| noisy: m3 - m3_shuffled | +0.0049, +0.0021, +0.0043 | +0.0038 | 3/3 |
| noisy: m3 - m1 | +0.0029, +0.0005, +0.0011 | +0.0015 | 3/3 |
| noisy: m3 - oracle | +0.0022, +0.0036, +0.0020 | +0.0026 | 3/3 |
| noisy: m3 - vgr | +0.0070, +0.0057, +0.0054 | +0.0060 | 3/3 |

Detection of the 564 corrupted cold items (AUROC per seed): agreement with the image neighbours' text 0.511, 0.559, 0.708; VGR cosine of the expert outputs 0.999, 0.999, 0.999.

| Selected on noisy validation data | seed 2025 | seed 2026 | seed 2027 |
|---|---|---|---|
| vgr | 1.0 | 1.0 | 1.0 |
| m1 | 0.75 | 0.75 | 0.5 |
| m1_plus_vgr | 1.0 | 1.0 | 2.0 |
| m1_random | 0.25 | 0.25 | 0.5 |
| m2 | [1.0, 5.0] | [1.0, 20.0] | [1.0, 5.0] |
| m2_plus_vgr | 2.0 | 1.0 | 2.0 |
| m2_random | [0.5, 5.0] | [0.5, 5.0] | [0.5, 5.0] |
| m2_shuffled | [1.0, 5.0] | [1.0, 5.0] | [1.0, 5.0] |
| m3 | [1.0, 0.5, 20.0] | [0.75, 0.5, 20.0] | [0.75, 0.5, 20.0] |
| m3_plus_vgr | 2.0 | 1.0 | 1.0 |
| m3_random | [0.75, 0.5, 20.0] | [0.75, 0.4, 20.0] | [0.75, 0.5, 20.0] |
| m3_shuffled | [0.75, 0.5, 20.0] | [1.0, 0.5, 20.0] | [0.75, 0.5, 20.0] |
| oracle | 0.75 | 0.75 | 0.75 |

Cost: nearest-neighbour search 0.54 s once, building the repaired features 0.012 s, extra memory 41 MB (computed from the array sizes), forward pass 10.39 ms without and 10.43 ms with repaired text; the cosine for M3 takes 0.016 s and 0.005 MB.

## Text repair under other noise (Recall@20; gains over the unrepaired model, mean over seeds, seeds above 0 in brackets)

Validation noise equal to the test noise:

| Test noise | validation noise | unrepaired | VGR | M1 | M3 | M3, random | M3, permuted | M3 - M1 | M3 + VGR | M3 on clean | AUROC cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian 0.2 | gaussian 0.2 | 0.0736 | +0.0009 (3/3) | +0.0039 (3/3) | +0.0048 (3/3) | -0.0038 (0/3) | +0.0021 (3/3) | +0.0009 (2/3) | +0.0050 (3/3) | +0.0016 | 0.995 |
| gaussian 0.4 | gaussian 0.4 | 0.0721 | +0.0009 (3/3) | +0.0043 (3/3) | +0.0039 (3/3) | -0.0007 (1/3) | +0.0035 (3/3) | -0.0003 (1/3) | +0.0047 (3/3) | +0.0014 | 0.994 |
| gaussian 0.6 | gaussian 0.6 | 0.0664 | +0.0024 (3/3) | +0.0077 (3/3) | +0.0075 (3/3) | -0.0025 (0/3) | +0.0042 (3/3) | -0.0001 (1/3) | +0.0079 (3/3) | +0.0013 | 0.993 |
| hashtag 0.2 | hashtag 0.2 | 0.0716 | +0.0015 (3/3) | +0.0049 (3/3) | +0.0060 (3/3) | -0.0028 (0/3) | +0.0024 (3/3) | +0.0012 (2/3) | +0.0061 (3/3) | +0.0017 | 0.999 |
| hashtag 0.4 | hashtag 0.4 | 0.0668 | +0.0033 (3/3) | +0.0078 (3/3) | +0.0094 (3/3) | -0.0023 (0/3) | +0.0056 (3/3) | +0.0015 (3/3) | +0.0105 (3/3) | +0.0016 | 0.999 |
| hashtag 0.6 | hashtag 0.6 | 0.0599 | +0.0042 (3/3) | +0.0135 (3/3) | +0.0101 (3/3) | -0.0008 (0/3) | +0.0050 (3/3) | -0.0034 (0/3) | +0.0129 (3/3) | +0.0019 | 0.999 |
| shuffle 0.2 | shuffle 0.2 | 0.0681 | +0.0028 (3/3) | +0.0065 (3/3) | +0.0094 (3/3) | +0.0006 (2/3) | +0.0038 (3/3) | +0.0029 (3/3) | +0.0094 (3/3) | +0.0016 | 1.000 |
| shuffle 0.4 | shuffle 0.4 | 0.0604 | +0.0072 (3/3) | +0.0128 (3/3) | +0.0152 (3/3) | +0.0028 (3/3) | +0.0097 (3/3) | +0.0023 (3/3) | +0.0159 (3/3) | +0.0009 | 1.000 |
| shuffle 0.6 | shuffle 0.6 | 0.0485 | +0.0101 (3/3) | +0.0236 (3/3) | +0.0224 (3/3) | +0.0077 (3/3) | +0.0093 (3/3) | -0.0012 (1/3) | +0.0241 (3/3) | +0.0009 | 1.000 |

Validation noise fixed at hashtag 0.4, test noise different (share of the matched M3 gain kept in the last column):

| Test noise | validation noise | unrepaired | VGR | M1 | M3 | M3, random | M3, permuted | M3 - M1 | M3 + VGR | M3 on clean | AUROC cosine | M3 gain kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian 0.4 | hashtag 0.4 | 0.0721 | +0.0009 (3/3) | +0.0036 (3/3) | +0.0046 (3/3) | -0.0059 (0/3) | +0.0034 (3/3) | +0.0010 (3/3) | +0.0055 (3/3) | +0.0016 | 0.994 | 116% |
| hashtag 0.2 | hashtag 0.4 | 0.0716 | +0.0015 (3/3) | +0.0042 (3/3) | +0.0060 (3/3) | -0.0030 (0/3) | +0.0028 (3/3) | +0.0018 (3/3) | +0.0059 (3/3) | +0.0016 | 0.999 | 99% |
| hashtag 0.6 | hashtag 0.4 | 0.0599 | +0.0042 (3/3) | +0.0135 (3/3) | +0.0106 (3/3) | -0.0010 (1/3) | +0.0071 (3/3) | -0.0029 (0/3) | +0.0126 (3/3) | +0.0016 | 0.999 | 104% |
| shuffle 0.4 | hashtag 0.4 | 0.0604 | +0.0074 (3/3) | +0.0134 (3/3) | +0.0153 (3/3) | +0.0034 (3/3) | +0.0099 (3/3) | +0.0019 (3/3) | +0.0164 (3/3) | +0.0016 | 1.000 | 101% |

## Representation probe: warm and cold items in the space the gate reads (clean text; seeds [2025, 2026, 2027]; mean ± sample std)

| Model | group | norm (image) | norm (text) | distance to warm centroid | gate logit (text - image) | w_t |
|---|---|---|---|---|---|---|
| Noise-augmented training | warm | 1.931 ± 0.148 | 1.500 ± 0.083 | 2.373 ± 0.164 | -0.139 ± 0.037 | 0.436 ± 0.001 |
| Noise-augmented training | cold | 1.160 ± 0.071 | 1.249 ± 0.076 | 1.718 ± 0.100 | 11.671 ± 0.839 | 0.987 ± 0.004 |
| Clean training (control) | warm | 2.779 ± 0.024 | 2.706 ± 0.026 | 3.771 ± 0.028 | -0.054 ± 0.003 | 0.487 ± 0.001 |
| Clean training (control) | cold | 2.089 ± 0.044 | 2.310 ± 0.031 | 2.954 ± 0.041 | 0.201 ± 0.008 | 0.549 ± 0.002 |

| Model | cold / warm distance ratio per seed | Spearman(distance, w_t) over cold items per seed |
|---|---|---|
| Noise-augmented training | 0.720, 0.719, 0.734 | +0.079, +0.150, +0.136 |
| Clean training (control) | 0.784, 0.788, 0.778 | +0.006, -0.036, +0.073 |
