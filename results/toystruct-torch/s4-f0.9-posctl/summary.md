# toy_struct s4, frac 0.9, question split, budget 25000 steps, lam 1.0, cell windows ['None']

chance (max of 1/n_out and the majority-of-train baseline): 0.042; floor line = chance + 0.05 = 0.092

| arm | seeds | t_mem median | t_grok censored median | grokked | AUC median (co-primary) | t_0.5 median | final test mean | final test_fresh mean | CKA own target @2000 / final |
|---|---|---|---|---|---|---|---|---|---|
| none | 5 | 200 | 20800 | 3/5 | 0.642 | 7100 | 0.910 | 0.901 |  |

qualifying fraction for rule (a) (none's censored median t_grok >= 4000): True

comparison T(a, b), paired by seed; PASS on either co-primary, each at p < 0.025 (t_grok: a earlier; AUC: a higher by >= 0.05 in the medians); wins need >= ceil(0.8 n)

| test | n | t_grok median a / b | wins | p | AUC median a / b | wins | p | verdict |
|---|---|---|---|---|---|---|---|---|

rule (a): arms missing

rule (b): coset_share of wrong test answers pooled over evals in [1000, 2000] (uniform-error chance 0.130, approximate for dim: answers are not uniform); per seed (share, n_wrong); a seed counts only with n_wrong >= 20

- none: s0 0.50 (570), s1 0.72 (494), s2 0.75 (589), s3 0.17 (567), s4 0.14 (595)
rule (c) (read at the lowest fraction of the design only): arms missing
