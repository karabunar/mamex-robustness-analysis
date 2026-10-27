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
