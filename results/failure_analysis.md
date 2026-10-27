# Failure analysis (cold-item test set, top-20; ranks are among the cold candidates)

Only item ids are available (the processed data has no titles), so items are listed by id.
helped / hurt = a user whose NDCG@k is higher / lower with the first-named condition; the mean differences are first minus second.

## Users

### noisy_vgr vs noisy

| seed | group | users | helped | hurt | unchanged | mean Δrecall | mean Δndcg |
|---|---|---|---|---|---|---|---|
| 2025 | all | 10474 | 319 | 304 | 9851 | +0.0043 | +0.0018 |
| 2025 | exposed to corrupted item | 5336 | 238 | 117 | 4981 | +0.0101 | +0.0043 |
| 2025 | not exposed | 5138 | 81 | 187 | 4870 | -0.0016 | -0.0008 |
| 2026 | all | 10474 | 257 | 297 | 9920 | +0.0023 | +0.0008 |
| 2026 | exposed to corrupted item | 5225 | 172 | 109 | 4944 | +0.0073 | +0.0027 |
| 2026 | not exposed | 5249 | 85 | 188 | 4976 | -0.0027 | -0.0011 |
| 2027 | all | 10474 | 283 | 299 | 9892 | +0.0034 | +0.0011 |
| 2027 | exposed to corrupted item | 5097 | 182 | 104 | 4811 | +0.0088 | +0.0031 |
| 2027 | not exposed | 5377 | 101 | 195 | 5081 | -0.0017 | -0.0008 |

### clean_vgr vs clean

| seed | group | users | helped | hurt | unchanged | mean Δrecall | mean Δndcg |
|---|---|---|---|---|---|---|---|
| 2025 | all | 10474 | 225 | 231 | 10018 | +0.0001 | +0.0000 |
| 2025 | exposed to corrupted item | 5336 | 151 | 148 | 5037 | +0.0001 | +0.0001 |
| 2025 | not exposed | 5138 | 74 | 83 | 4981 | +0.0000 | -0.0001 |
| 2026 | all | 10474 | 179 | 254 | 10041 | -0.0011 | -0.0005 |
| 2026 | exposed to corrupted item | 5225 | 111 | 140 | 4974 | -0.0015 | -0.0004 |
| 2026 | not exposed | 5249 | 68 | 114 | 5067 | -0.0007 | -0.0005 |
| 2027 | all | 10474 | 193 | 252 | 10029 | -0.0000 | -0.0001 |
| 2027 | exposed to corrupted item | 5097 | 110 | 133 | 4854 | -0.0010 | -0.0002 |
| 2027 | not exposed | 5377 | 83 | 119 | 5175 | +0.0009 | -0.0000 |

### noisy_vgr vs noisy_vgr_shuffled

| seed | group | users | helped | hurt | unchanged | mean Δrecall | mean Δndcg |
|---|---|---|---|---|---|---|---|
| 2025 | all | 10474 | 189 | 151 | 10134 | +0.0018 | +0.0009 |
| 2025 | exposed to corrupted item | 5336 | 155 | 63 | 5118 | +0.0050 | +0.0020 |
| 2025 | not exposed | 5138 | 34 | 88 | 5016 | -0.0015 | -0.0004 |
| 2026 | all | 10474 | 147 | 133 | 10194 | +0.0011 | +0.0005 |
| 2026 | exposed to corrupted item | 5225 | 122 | 55 | 5048 | +0.0028 | +0.0013 |
| 2026 | not exposed | 5249 | 25 | 78 | 5146 | -0.0005 | -0.0002 |
| 2027 | all | 10474 | 173 | 141 | 10160 | +0.0021 | +0.0007 |
| 2027 | exposed to corrupted item | 5097 | 140 | 50 | 4907 | +0.0047 | +0.0016 |
| 2027 | not exposed | 5377 | 33 | 91 | 5253 | -0.0004 | -0.0003 |

## Items

### noisy_vgr vs noisy

| seed | items | pairs | mean rank gain | share improved | share worse |
|---|---|---|---|---|---|
| 2025 | corrupted | 6472 | +18.0 | 0.59 | 0.40 |
| 2025 | not_corrupted | 9510 | +4.2 | 0.50 | 0.41 |
| 2026 | corrupted | 6248 | +11.4 | 0.57 | 0.42 |
| 2026 | not_corrupted | 9734 | +6.2 | 0.53 | 0.38 |
| 2027 | corrupted | 6128 | +16.5 | 0.59 | 0.40 |
| 2027 | not_corrupted | 9854 | +2.5 | 0.47 | 0.44 |

Most hurt / most helped test items by id (mean rank gain; positive = first-named condition better):

- seed 2025: most hurt [(6173, -123.8), (38, -78.4), (888, -75.0), (2081, -65.8), (3520, -57.4)]; most helped [(2902, 158.5), (33, 115.8), (5981, 105.0), (202, 98.8), (4783, 97.4)]
- seed 2026: most hurt [(3831, -141.2), (3568, -78.2), (6797, -68.1), (4136, -65.3), (4917, -62.0)]; most helped [(4502, 125.6), (6482, 120.8), (3124, 118.2), (3021, 110.6), (509, 92.5)]
- seed 2027: most hurt [(3132, -94.5), (3007, -61.6), (1551, -61.5), (5260, -57.6), (4551, -56.8)]; most helped [(5589, 150.2), (6927, 142.2), (4213, 134.5), (1161, 123.6), (841, 115.4)]

### clean_vgr vs clean

| seed | items | pairs | mean rank gain | share improved | share worse |
|---|---|---|---|---|---|
| 2025 | corrupted | 6472 | -0.3 | 0.43 | 0.45 |
| 2025 | not_corrupted | 9510 | -0.6 | 0.42 | 0.47 |
| 2026 | corrupted | 6248 | -0.3 | 0.43 | 0.47 |
| 2026 | not_corrupted | 9734 | -0.4 | 0.42 | 0.48 |
| 2027 | corrupted | 6128 | -0.5 | 0.41 | 0.48 |
| 2027 | not_corrupted | 9854 | -0.3 | 0.42 | 0.47 |

Most hurt / most helped test items by id (mean rank gain; positive = first-named condition better):

- seed 2025: most hurt [(6116, -22.2), (4797, -15.2), (33, -14.0), (6995, -13.8), (6302, -13.7)]; most helped [(4412, 40.6), (1688, 31.1), (7011, 18.0), (495, 11.6), (4499, 11.0)]
- seed 2026: most hurt [(6899, -19.5), (3325, -19.4), (3553, -17.4), (3269, -17.1), (6649, -15.4)]; most helped [(1933, 28.8), (1399, 24.2), (1742, 18.9), (1688, 18.7), (7011, 16.2)]
- seed 2027: most hurt [(938, -18.1), (3325, -14.2), (2793, -13.7), (3977, -12.5), (1513, -12.0)]; most helped [(1688, 27.9), (882, 16.2), (2381, 16.0), (4637, 14.9), (4608, 14.9)]

### noisy_vgr vs noisy_vgr_shuffled

| seed | items | pairs | mean rank gain | share improved | share worse |
|---|---|---|---|---|---|
| 2025 | corrupted | 6472 | +5.9 | 0.56 | 0.41 |
| 2025 | not_corrupted | 9510 | +2.5 | 0.50 | 0.34 |
| 2026 | corrupted | 6248 | +3.7 | 0.54 | 0.43 |
| 2026 | not_corrupted | 9734 | +3.3 | 0.54 | 0.30 |
| 2027 | corrupted | 6128 | +5.4 | 0.57 | 0.40 |
| 2027 | not_corrupted | 9854 | +1.8 | 0.47 | 0.36 |

Most hurt / most helped test items by id (mean rank gain; positive = first-named condition better):

- seed 2025: most hurt [(6173, -63.2), (38, -45.4), (888, -40.6), (2081, -33.2), (4840, -31.4)]; most helped [(2902, 89.0), (33, 66.6), (4886, 47.8), (4783, 45.2), (3932, 44.4)]
- seed 2026: most hurt [(3831, -62.3), (3568, -36.4), (1448, -35.8), (4136, -33.2), (6797, -31.2)]; most helped [(4502, 67.8), (6482, 59.3), (3124, 52.3), (509, 45.9), (5836, 43.8)]
- seed 2027: most hurt [(3132, -49.7), (1551, -39.0), (5260, -32.8), (3007, -28.1), (3786, -21.8)]; most helped [(4213, 68.9), (5589, 64.2), (1161, 60.0), (5233, 53.8), (841, 49.4)]

