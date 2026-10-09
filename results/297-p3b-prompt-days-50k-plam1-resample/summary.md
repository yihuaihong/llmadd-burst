# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| none | 3 | 0.973 ± 0.034 | 0.626 ± 0.102 | 0.131 ± 0.002 | 0.86 / 0.52 / 0.59; test prompts: sum_helix@L16 0.26 out_circle@L24 0.28 |
| out_circle | 3 | 0.996 ± 0.005 | 0.912 ± 0.047 | 0.148 ± 0.012 | 0.93 / 0.51 / 0.58; test prompts: sum_helix@L16 0.24 out_circle@L24 0.86 |
| sum_helix | 3 | 0.996 ± 0.005 | 0.600 ± 0.007 | 0.136 ± 0.010 | 0.88 / 0.51 / 0.59; test prompts: sum_helix@L16 0.67 out_circle@L24 0.24 |
| sum_helix_shuf | 3 | 1.000 ± 0.000 | 0.516 ± 0.040 | 0.139 ± 0.005 | 0.86 / 0.52 / 0.58; test prompts: sum_helix@L16 0.42 out_circle@L24 0.28 |
