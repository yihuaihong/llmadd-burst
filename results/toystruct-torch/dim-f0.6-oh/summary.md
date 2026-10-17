# toy_struct dim, frac 0.6, class split, budget 25000 steps, lam 1.0, cell windows ['None', '[0, 2000]']

chance (max of 1/n_out and the majority-of-train baseline): 0.072; floor line = chance + 0.05 = 0.122

| arm | seeds | t_mem median | t_grok censored median | grokked | AUC median (co-primary) | t_0.5 median | final test mean | final test_fresh mean | CKA own target @2000 / final |
|---|---|---|---|---|---|---|---|---|---|
| none | 5 | 200 | 2400 | 5/5 | 0.955 | 1000 | 0.991 | 0.991 |  |
| expo | 5 | 200 | 200 | 5/5 | 0.992 | 200 | 0.994 | 0.994 | 1.00 / 0.99 |
| quot | 5 | 200 | 400 | 5/5 | 0.989 | 200 | 0.993 | 0.993 | 1.00 / 0.83 |
| onehot | 5 | 200 | 300 | 5/5 | 0.987 | 200 | 0.998 | 0.998 | 1.00 / 0.90 |
| specrand | 5 | 200 | 2100 | 5/5 | 0.946 | 1500 | 0.989 | 0.989 | 1.00 / 0.16 |
| shuf | 5 | 200 | 3700 | 5/5 | 0.937 | 1700 | 0.991 | 0.991 | 1.00 / 0.18 |
| specrand_oh | 5 | 200 | 2100 | 5/5 | 0.959 | 1000 | 0.996 | 0.996 | 1.00 / 0.22 |

qualifying fraction for rule (a) (none's censored median t_grok >= 4000): False

comparison T(a, b), paired by seed; PASS on either co-primary, each at p < 0.025 (t_grok: a earlier; AUC: a higher by >= 0.05 in the medians); wins need >= ceil(0.8 n)

| test | n | t_grok median a / b | wins | p | AUC median a / b | wins | p | verdict |
|---|---|---|---|---|---|---|---|---|
| expo < specrand | 5 | 200 / 2100 | 5/5 | 0.004 | 0.992 / 0.946 | 5/5 | 0.004 | PASS (t_grok) |
| onehot < specrand | 5 | 300 / 2100 | 5/5 | 0.004 | 0.987 / 0.946 | 5/5 | 0.004 | PASS (t_grok) |
| quot < specrand | 5 | 400 / 2100 | 5/5 | 0.004 | 0.989 / 0.946 | 5/5 | 0.004 | PASS (t_grok) |
| expo < none | 5 | 200 / 2400 | 5/5 | 0.004 | 0.992 / 0.955 | 5/5 | 0.004 | PASS (t_grok) |
| onehot < none | 5 | 300 / 2400 | 5/5 | 0.004 | 0.987 / 0.955 | 5/5 | 0.004 | PASS (t_grok) |
| specrand < none | 5 | 2100 / 2400 | 3/5 | 0.556 | 0.946 / 0.955 | 2/5 | 0.655 | not shown at n=5 |
| shuf < none | 5 | 3700 / 2400 | 0/5 | 0.833 | 0.937 / 0.955 | 0/5 | 0.925 | not shown at n=5 |
| expo < onehot | 5 | 200 / 300 | 3/5 | 0.060 | 0.992 / 0.987 | 4/5 | 0.155 | not shown at n=5 |
| onehot < expo | 5 | 300 / 200 | 0/5 | 1.000 | 0.987 / 0.992 | 1/5 | 0.889 | not shown at n=5 |
| onehot < specrand_oh | 5 | 300 / 2100 | 5/5 | 0.004 | 0.987 / 0.959 | 5/5 | 0.004 | PASS (t_grok) |
| specrand_oh < none | 5 | 2100 / 2400 | 2/5 | 0.627 | 0.959 / 0.955 | 3/5 | 0.579 | not shown at n=5 |
| expo < specrand_oh | 5 | 200 / 2100 | 5/5 | 0.004 | 0.992 / 0.959 | 5/5 | 0.004 | PASS (t_grok) |

rule (a): untestable at this fraction (none's censored median t_grok < 4000: none groks early); not a negative

rule (b): drop_share of wrong test answers pooled over evals in [1000, 2000] (uniform-error chance 0.077, approximate for dim: answers are not uniform); per seed (share, n_wrong); a seed counts only with n_wrong >= 20

- none: s0 0.33 (2170), s1 0.13 (968), s2 0.54 (100), s3 0.17 (109), s4 0.33 (1276)
- expo: s0 - (0), s1 - (0), s2 0.00 (1), s3 - (0), s4 0.36 (14)
- quot: s0 - (0), s1 0.08 (88), s2 0.66 (41), s3 - (0), s4 0.00 (3)
- onehot: s0 0.00 (36), s1 - (0), s2 - (0), s3 0.00 (5), s4 - (0)
- specrand: s0 0.43 (2343), s1 0.27 (1318), s2 0.49 (213), s3 0.10 (163), s4 0.06 (2666)
- shuf: s0 0.25 (2679), s1 0.11 (2035), s2 0.24 (207), s3 0.18 (784), s4 0.25 (1529)
- specrand_oh: s0 0.48 (2167), s1 0.17 (757), s2 0.23 (411), s3 0.05 (241), s4 0.10 (2577)

rule (b) quot: share > 0.5 in 1/2 determined seeds -> undetermined (< 4 seeds with >= 20 wrong answers)
control: none's share > 0.5 in 1/5 determined seeds (>= 4 -> the error signature is generic, not evidence that the prior was read)
rule (c) (read at the lowest fraction of the design only): none failed 0/5, expo grokked 5/5: none learns here: not the (c) regime
rule (d) (read only if expo or onehot passes vs none or vs specrand): expo < onehot = the rank-3 additive code beats its rank-6 superset (onehot's centred span contains expo); onehot < expo = the rank-6 superset is faster, which rank alone can produce (F59) -- with specrand_oh, onehot < specrand_oh separates onehot's structure from its rank; neither = not separated at n=5

no-twin subset (held-out questions whose negation twin is not in training; twin counts per seed s0 152, s1 160, s2 156, s3 176, s4 152): median AUC per arm: none 0.946, expo 0.991, quot 0.988, onehot 0.991, specrand 0.941, shuf 0.905, specrand_oh 0.938
- AUC expo > specrand on the no-twin subset: 0.991 / 0.941, wins 5/5, p 0.004 -> not shown
- AUC expo > none on the no-twin subset: 0.991 / 0.946, wins 5/5, p 0.004 -> not shown
- AUC onehot > specrand on the no-twin subset: 0.991 / 0.941, wins 5/5, p 0.004 -> not shown
