# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| none | 3 | 0.973 ± 0.034 | 0.626 ± 0.102 | 0.131 ± 0.002 | 0.86 / 0.52 / 0.59; test prompts: sum_helix@L16 0.26 out_circle@L24 0.28 |
| out_circle | 3 | 1.000 ± 0.000 | 0.798 ± 0.105 | 0.152 ± 0.030 | 0.89 / 0.48 / 0.57; test prompts: sum_helix@L16 0.35 out_circle@L24 0.81 |
| sum_helix | 3 | 0.985 ± 0.021 | 0.424 ± 0.016 | 0.141 ± 0.022 | 0.88 / 0.52 / 0.58; test prompts: sum_helix@L16 0.70 out_circle@L24 0.13 |
| sum_helix_shuf | 3 | 0.979 ± 0.019 | 0.173 ± 0.052 | 0.134 ± 0.019 | 0.80 / 0.52 / 0.56; test prompts: sum_helix@L16 0.22 out_circle@L24 0.08 |
