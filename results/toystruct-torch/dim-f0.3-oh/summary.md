# toy_struct dim, frac 0.3, class split, budget 50000 steps, lam 1.0, cell windows ['None', '[0, 2000]']

chance (max of 1/n_out and the majority-of-train baseline): 0.037; floor line = chance + 0.05 = 0.087

| arm | seeds | t_mem median | t_grok censored median | grokked | AUC median (co-primary) | t_0.5 median | final test mean | final test_fresh mean | CKA own target @2000 / final |
|---|---|---|---|---|---|---|---|---|---|
| none | 5 | 100 | > 50000 | 0/5 | 0.038 | > 50000 | 0.041 | 0.041 |  |
| expo | 5 | 200 | 4800 | 4/5 | 0.932 | 1700 | 0.778 | 0.778 | 1.00 / 0.88 |
| quot | 5 | 200 | > 50000 | 0/5 | 0.093 | > 50000 | 0.236 | 0.236 | 1.00 / 0.72 |
| onehot | 5 | 100 | > 50000 | 0/5 | 0.603 | 1800 | 0.603 | 0.603 | 1.00 / 0.78 |
| specrand | 5 | 200 | > 50000 | 0/5 | 0.042 | > 50000 | 0.053 | 0.053 | 1.00 / 0.39 |
| shuf | 5 | 200 | > 50000 | 0/5 | 0.032 | > 50000 | 0.036 | 0.036 | 1.00 / 0.46 |
| specrand_oh | 5 | 200 | > 50000 | 0/5 | 0.039 | > 50000 | 0.047 | 0.047 | 1.00 / 0.43 |

qualifying fraction for rule (a) (none's censored median t_grok >= 4000): True

comparison T(a, b), paired by seed; PASS on either co-primary, each at p < 0.025 (t_grok: a earlier; AUC: a higher by >= 0.05 in the medians); wins need >= ceil(0.8 n)

| test | n | t_grok median a / b | wins | p | AUC median a / b | wins | p | verdict |
|---|---|---|---|---|---|---|---|---|
| expo < specrand | 5 | 4800 / > budget | 4/5 | 0.024 | 0.932 / 0.042 | 5/5 | 0.004 | PASS (t_grok + AUC) |
| onehot < specrand | 5 | > budget / > budget | 0/5 | 1.000 | 0.603 / 0.042 | 5/5 | 0.004 | PASS (AUC) |
| quot < specrand | 5 | > budget / > budget | 0/5 | 1.000 | 0.093 / 0.042 | 5/5 | 0.004 | PASS (AUC) |
| expo < none | 5 | 4800 / > budget | 4/5 | 0.024 | 0.932 / 0.038 | 5/5 | 0.004 | PASS (t_grok + AUC) |
| onehot < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.603 / 0.038 | 5/5 | 0.004 | PASS (AUC) |
| specrand < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.042 / 0.038 | 3/5 | 0.345 | floor: uninformative |
| shuf < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.032 / 0.038 | 3/5 | 0.579 | floor: uninformative |
| expo < onehot | 5 | 4800 / > budget | 4/5 | 0.024 | 0.932 / 0.603 | 4/5 | 0.075 | PASS (t_grok) |
| onehot < expo | 5 | > budget / 4800 | 0/5 | 1.000 | 0.603 / 0.932 | 1/5 | 0.952 | not shown at n=5 |
| onehot < specrand_oh | 5 | > budget / > budget | 0/5 | 1.000 | 0.603 / 0.039 | 5/5 | 0.004 | PASS (AUC) |
| specrand_oh < none | 5 | > budget / > budget | 0/5 | 1.000 | 0.039 / 0.038 | 3/5 | 0.421 | floor: uninformative |
| expo < specrand_oh | 5 | 4800 / > budget | 4/5 | 0.024 | 0.932 / 0.039 | 5/5 | 0.004 | PASS (t_grok + AUC) |

rule (a): PASS: linear structure helps

rule (b): drop_share of wrong test answers pooled over evals in [1000, 2000] (uniform-error chance 0.077, approximate for dim: answers are not uniform); per seed (share, n_wrong); a seed counts only with n_wrong >= 20

- none: s0 0.09 (5130), s1 0.08 (5245), s2 0.10 (5066), s3 0.06 (5158), s4 0.15 (5096)
- expo: s0 0.18 (4769), s1 0.37 (801), s2 0.17 (4041), s3 0.38 (2088), s4 0.35 (3161)
- quot: s0 0.33 (4859), s1 0.84 (4476), s2 0.34 (4908), s3 0.60 (4362), s4 0.48 (4800)
- onehot: s0 0.26 (3017), s1 0.25 (2473), s2 0.11 (4101), s3 0.19 (2777), s4 0.22 (2832)
- specrand: s0 0.13 (5204), s1 0.10 (5255), s2 0.11 (5118), s3 0.05 (5188), s4 0.11 (5227)
- shuf: s0 0.11 (5195), s1 0.10 (5171), s2 0.10 (5087), s3 0.10 (5171), s4 0.10 (5110)
- specrand_oh: s0 0.10 (5154), s1 0.09 (5235), s2 0.08 (5154), s3 0.09 (5086), s4 0.10 (5230)

rule (b) quot: share > 0.5 in 2/5 determined seeds -> False
control: none's share > 0.5 in 0/5 determined seeds (>= 4 -> the error signature is generic, not evidence that the prior was read)
rule (c) (read at the lowest fraction of the design only): none failed 5/5, expo grokked 4/5: RESCUE (without the prior it is not learned within the budget)
rule (d) (read only if expo or onehot passes vs none or vs specrand): expo < onehot = the rank-3 additive code beats its rank-6 superset (onehot's centred span contains expo); onehot < expo = the rank-6 superset is faster, which rank alone can produce (F59) -- with specrand_oh, onehot < specrand_oh separates onehot's structure from its rank; neither = not separated at n=5

no-twin subset (held-out questions whose negation twin is not in training; twin counts per seed s0 130, s1 142, s2 128, s3 154, s4 130): median AUC per arm: none 0.021, expo 0.934, quot 0.085, onehot 0.575, specrand 0.024, shuf 0.016, specrand_oh 0.017
- AUC expo > specrand on the no-twin subset: 0.934 / 0.024, wins 5/5, p 0.004 -> PASS
- AUC expo > none on the no-twin subset: 0.934 / 0.021, wins 5/5, p 0.004 -> PASS
- AUC onehot > specrand on the no-twin subset: 0.575 / 0.024, wins 5/5, p 0.004 -> PASS
