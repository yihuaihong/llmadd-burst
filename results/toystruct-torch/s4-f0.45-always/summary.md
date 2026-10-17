# toy_struct s4, frac 0.45, question split, budget 50000 steps, lam 1.0, cell windows ['None', '[0, 2000]', 'always']

chance (max of 1/n_out and the majority-of-train baseline): 0.042; floor line = chance + 0.05 = 0.092

| arm | seeds | t_mem median | t_grok censored median | grokked | AUC median (co-primary) | t_0.5 median | final test mean | final test_fresh mean | CKA own target @2000 / final |
|---|---|---|---|---|---|---|---|---|---|
| none | 5 | 100 | > 50000 | 0/5 | 0.016 | > 50000 | 0.018 | 0.015 |  |
| R | 5 | 200 | > 50000 | 0/5 | 0.105 | > 50000 | 0.141 | 0.131 | 1.00 / 0.78 |
| R_sign | 5 | 100 | > 50000 | 0/5 | 0.140 | > 50000 | 0.127 | 0.112 | 1.00 / 0.78 |
| absR | 5 | 200 | > 50000 | 0/5 | 0.045 | > 50000 | 0.045 | 0.039 | 1.00 / 0.76 |
| specrand | 5 | 200 | > 50000 | 0/5 | 0.014 | > 50000 | 0.019 | 0.015 | 1.00 / 0.62 |
| shuf | 5 | 100 | > 50000 | 0/5 | 0.022 | > 50000 | 0.027 | 0.024 | 1.00 / 0.64 |
| R@always | 5 | 200 | > 50000 | 1/5 | 0.487 | 13800 | 0.608 | 0.593 | 1.00 / 1.00 |
| R_sign@always | 5 | 100 | > 50000 | 0/5 | 0.529 | 22700 | 0.765 | 0.758 | 1.00 / 1.00 |
| specrand@always | 5 | 200 | > 50000 | 0/5 | 0.005 | > 50000 | 0.007 | 0.007 | 1.00 / 1.00 |

qualifying fraction for rule (a) (none's censored median t_grok >= 4000): True

comparison T(a, b), paired by seed; PASS on either co-primary, each at p < 0.025 (t_grok: a earlier; AUC: a higher by >= 0.05 in the medians); wins need >= ceil(0.8 n)

| test | n | t_grok median a / b | wins | p | AUC median a / b | wins | p | verdict |
|---|---|---|---|---|---|---|---|---|
| R < specrand | 5 | > budget / > budget | 0/5 | 1.000 | 0.105 / 0.014 | 5/5 | 0.004 | PASS (AUC) |
| R_sign < specrand | 5 | > budget / > budget | 0/5 | 1.000 | 0.140 / 0.014 | 5/5 | 0.004 | PASS (AUC) |
| absR < specrand | 5 | > budget / > budget | 0/5 | 1.000 | 0.045 / 0.014 | 5/5 | 0.004 | floor: uninformative |
| R < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.105 / 0.016 | 5/5 | 0.004 | PASS (AUC) |
| R_sign < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.140 / 0.016 | 5/5 | 0.004 | PASS (AUC) |
| specrand < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.014 / 0.016 | 2/5 | 0.726 | floor: uninformative |
| shuf < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.022 / 0.016 | 4/5 | 0.028 | floor: uninformative |
| absR < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.045 / 0.016 | 5/5 | 0.004 | floor: uninformative |
| R@always < specrand@always | 5 | > budget / > budget | 1/5 | 0.500 | 0.487 / 0.005 | 5/5 | 0.004 | PASS (AUC) |
| R_sign@always < specrand@always | 5 | > budget / > budget | 0/5 | 1.000 | 0.529 / 0.005 | 5/5 | 0.004 | PASS (AUC) |
| R@always < none | 5 | > budget / > budget | 1/5 | 0.500 | 0.487 / 0.016 | 5/5 | 0.004 | PASS (AUC) |
| R_sign@always < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.529 / 0.016 | 5/5 | 0.004 | PASS (AUC) |
| specrand@always < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.005 / 0.016 | 0/5 | 0.992 | floor: uninformative |
| R@always < R | 5 | > budget / > budget | 1/5 | 0.500 | 0.487 / 0.105 | 5/5 | 0.004 | PASS (AUC) |
| R < R@always | 5 | > budget / > budget | 0/5 | 1.000 | 0.105 / 0.487 | 0/5 | 1.000 | not shown at n=5 |

rule (a): PASS: group structure helps (R and R_sign both beat specrand)

rule (a) [@always arms, secondary]: PASS: group structure helps (R and R_sign both beat specrand)

rule (b): coset_share of wrong test answers pooled over evals in [1000, 2000] (uniform-error chance 0.130, approximate for dim: answers are not uniform); per seed (share, n_wrong); a seed counts only with n_wrong >= 20

- none: s0 0.19 (3452), s1 0.21 (3458), s2 0.25 (3432), s3 0.15 (3466), s4 0.15 (3473)
- R: s0 0.02 (3329), s1 0.03 (3235), s2 0.04 (3089), s3 0.06 (3116), s4 0.06 (3161)
- R_sign: s0 0.06 (3069), s1 0.05 (3053), s2 0.04 (3047), s3 0.05 (3098), s4 0.05 (2871)
- absR: s0 0.75 (3351), s1 0.86 (3345), s2 0.82 (3347), s3 0.84 (3331), s4 0.74 (3302)
- specrand: s0 0.16 (3479), s1 0.14 (3440), s2 0.13 (3484), s3 0.21 (3471), s4 0.15 (3459)
- shuf: s0 0.28 (3429), s1 0.31 (3453), s2 0.23 (3443), s3 0.28 (3429), s4 0.24 (3453)
- R@always: s0 0.02 (3329), s1 0.03 (3234), s2 0.04 (3089), s3 0.06 (3116), s4 0.06 (3161)
- R_sign@always: s0 0.06 (3069), s1 0.05 (3053), s2 0.04 (3049), s3 0.05 (3099), s4 0.05 (2869)
- specrand@always: s0 0.16 (3479), s1 0.14 (3440), s2 0.13 (3484), s3 0.21 (3471), s4 0.15 (3458)

rule (b) absR: share > 0.5 in 5/5 determined seeds -> True
control: none's share > 0.5 in 0/5 determined seeds (>= 4 -> the error signature is generic, not evidence that the prior was read)
rule (c) (read at the lowest fraction of the design only): none failed 5/5, R grokked 0/5: PARTIAL RESCUE (held-out AUC above none's, AUC median 0.11 vs 0.02; no grok)
rule (c) [@always] (read at the lowest fraction of the design only): none failed 5/5, R@always grokked 1/5: PARTIAL RESCUE (held-out AUC above none's, AUC median 0.49 vs 0.02; no grok)
