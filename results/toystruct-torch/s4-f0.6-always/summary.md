# toy_struct s4, frac 0.6, question split, budget 25000 steps, lam 1.0, cell windows ['None', '[0, 2000]', 'always']

chance (max of 1/n_out and the majority-of-train baseline): 0.042; floor line = chance + 0.05 = 0.092

| arm | seeds | t_mem median | t_grok censored median | grokked | AUC median (co-primary) | t_0.5 median | final test mean | final test_fresh mean | CKA own target @2000 / final |
|---|---|---|---|---|---|---|---|---|---|
| none | 5 | 100 | > 25000 | 0/5 | 0.014 | > 25000 | 0.045 | 0.037 |  |
| R | 5 | 200 | > 25000 | 1/5 | 0.506 | 3800 | 0.694 | 0.687 | 1.00 / 0.88 |
| R_sign | 5 | 200 | > 25000 | 2/5 | 0.728 | 1800 | 0.810 | 0.804 | 1.00 / 0.91 |
| absR | 5 | 200 | > 25000 | 0/5 | 0.041 | > 25000 | 0.068 | 0.062 | 1.00 / 0.75 |
| specrand | 5 | 200 | > 25000 | 0/5 | 0.031 | > 25000 | 0.031 | 0.017 | 1.00 / 0.69 |
| shuf | 5 | 200 | > 25000 | 0/5 | 0.024 | > 25000 | 0.029 | 0.018 | 1.00 / 0.70 |
| R@always | 5 | 200 | 10200 | 5/5 | 0.838 | 2900 | 1.000 | 1.000 | 1.00 / 1.00 |
| R_sign@always | 5 | 200 | 9400 | 4/5 | 0.875 | 1800 | 0.967 | 0.963 | 1.00 / 1.00 |
| specrand@always | 5 | 200 | > 25000 | 0/5 | 0.016 | > 25000 | 0.017 | 0.013 | 1.00 / 1.00 |

qualifying fraction for rule (a) (none's censored median t_grok >= 4000): True

comparison T(a, b), paired by seed; PASS on either co-primary, each at p < 0.025 (t_grok: a earlier; AUC: a higher by >= 0.05 in the medians); wins need >= ceil(0.8 n)

| test | n | t_grok median a / b | wins | p | AUC median a / b | wins | p | verdict |
|---|---|---|---|---|---|---|---|---|
| R < specrand | 5 | > budget / > budget | 1/5 | 0.500 | 0.506 / 0.031 | 5/5 | 0.004 | PASS (AUC) |
| R_sign < specrand | 5 | > budget / > budget | 2/5 | 0.222 | 0.728 / 0.031 | 5/5 | 0.004 | PASS (AUC) |
| absR < specrand | 5 | > budget / > budget | 0/5 | 1.000 | 0.041 / 0.031 | 4/5 | 0.075 | floor: uninformative |
| R < none | 5 | > budget / > budget | 1/5 | 0.500 | 0.506 / 0.014 | 5/5 | 0.004 | PASS (AUC) |
| R_sign < none | 5 | > budget / > budget | 2/5 | 0.222 | 0.728 / 0.014 | 5/5 | 0.004 | PASS (AUC) |
| specrand < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.031 / 0.014 | 3/5 | 0.274 | floor: uninformative |
| shuf < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.024 / 0.014 | 3/5 | 0.274 | floor: uninformative |
| absR < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.041 / 0.014 | 4/5 | 0.075 | floor: uninformative |
| R@always < specrand@always | 5 | 10200 / > budget | 5/5 | 0.004 | 0.838 / 0.016 | 5/5 | 0.004 | PASS (t_grok + AUC) |
| R_sign@always < specrand@always | 5 | 9400 / > budget | 4/5 | 0.024 | 0.875 / 0.016 | 5/5 | 0.004 | PASS (t_grok + AUC) |
| R@always < none | 5 | 10200 / > budget | 5/5 | 0.004 | 0.838 / 0.014 | 5/5 | 0.004 | PASS (t_grok + AUC) |
| R_sign@always < none | 5 | 9400 / > budget | 4/5 | 0.024 | 0.875 / 0.014 | 5/5 | 0.004 | PASS (t_grok + AUC) |
| specrand@always < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.016 / 0.014 | 1/5 | 0.726 | floor: uninformative |
| R@always < R | 5 | 10200 / > budget | 5/5 | 0.016 | 0.838 / 0.506 | 5/5 | 0.028 | PASS (t_grok) |
| R < R@always | 5 | > budget / 10200 | 0/5 | 0.988 | 0.506 / 0.838 | 0/5 | 0.984 | not shown at n=5 |

rule (a): PASS: group structure helps (R and R_sign both beat specrand)

rule (a) [@always arms, secondary]: PASS: group structure helps (R and R_sign both beat specrand)

rule (b): coset_share of wrong test answers pooled over evals in [1000, 2000] (uniform-error chance 0.130, approximate for dim: answers are not uniform); per seed (share, n_wrong); a seed counts only with n_wrong >= 20

- none: s0 0.15 (2505), s1 0.49 (2474), s2 0.23 (2518), s3 0.14 (2540), s4 0.11 (2538)
- R: s0 0.02 (1585), s1 0.01 (1913), s2 0.01 (1820), s3 0.05 (1917), s4 0.05 (1819)
- R_sign: s0 0.02 (1687), s1 0.01 (1538), s2 0.02 (1424), s3 0.01 (1463), s4 0.00 (1090)
- absR: s0 0.85 (2472), s1 0.90 (2476), s2 0.89 (2462), s3 0.91 (2469), s4 0.83 (2501)
- specrand: s0 0.18 (2490), s1 0.22 (2508), s2 0.14 (2528), s3 0.20 (2538), s4 0.14 (2540)
- shuf: s0 0.34 (2511), s1 0.45 (2472), s2 0.40 (2534), s3 0.39 (2511), s4 0.24 (2518)
- R@always: s0 0.02 (1585), s1 0.01 (1913), s2 0.01 (1820), s3 0.05 (1917), s4 0.05 (1819)
- R_sign@always: s0 0.02 (1687), s1 0.01 (1538), s2 0.02 (1424), s3 0.01 (1463), s4 0.00 (1090)
- specrand@always: s0 0.18 (2490), s1 0.22 (2508), s2 0.14 (2527), s3 0.20 (2538), s4 0.14 (2540)

rule (b) absR: share > 0.5 in 5/5 determined seeds -> True
control: none's share > 0.5 in 0/5 determined seeds (>= 4 -> the error signature is generic, not evidence that the prior was read)
rule (c) (read at the lowest fraction of the design only): none failed 5/5, R grokked 1/5: PARTIAL RESCUE (held-out AUC above none's, AUC median 0.51 vs 0.01; no grok)
rule (c) [@always] (read at the lowest fraction of the design only): none failed 5/5, R@always grokked 5/5: RESCUE (without the prior it is not learned within the budget)
