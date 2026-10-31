## H18: sports, Recall@20, hashtag noise on 40% of the cold items (3 seeds)

| seed | clean | noisy, unrepaired | VGR | M1 | M3 | M3 then VGR | M3 - random control | M3 - permuted control | M3 - VGR | M3 - M1 | AUROC of the cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | 0.0942 | 0.0769 | +0.0021 | +0.0040 | +0.0086 | +0.0075 | +0.0087 | +0.0073 | +0.0065 | +0.0045 | 0.996 |
| 2026 | 0.0951 | 0.0749 | +0.0016 | +0.0024 | +0.0070 | +0.0070 | +0.0073 | +0.0057 | +0.0054 | +0.0046 | 0.998 |
| 2027 | 0.0970 | 0.0806 | +0.0017 | +0.0003 | +0.0049 | +0.0049 | +0.0050 | +0.0059 | +0.0032 | +0.0045 | 0.998 |
| mean | 0.0954 | 0.0775 | +0.0018 | +0.0023 | +0.0068 | +0.0065 | +0.0070 | +0.0063 | +0.0050 | +0.0046 | |

Seeds above zero: VGR 3/3, M1 3/3, M3 3/3, M3 then VGR 3/3, M3 - random control 3/3, M3 - permuted control 3/3, M3 - VGR 3/3, M3 - M1 3/3
Loss from the corruption (clean minus noisy, unrepaired): +0.0173, +0.0201, +0.0164
Selected on noisy validation data (M3): seed 2025 [1.0, 0.4, 50.0]; seed 2026 [0.75, 0.4, 20.0]; seed 2027 [0.75, 0.4, 50.0]
