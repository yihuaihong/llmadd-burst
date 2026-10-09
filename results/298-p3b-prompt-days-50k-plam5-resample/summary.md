# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| none | 3 | 0.973 ± 0.034 | 0.626 ± 0.102 | 0.131 ± 0.002 | 0.86 / 0.52 / 0.59; test prompts: sum_helix@L16 0.26 out_circle@L24 0.28 |
| out_circle | 3 | 1.000 ± 0.000 | 0.969 ± 0.008 | 0.162 ± 0.008 | 0.94 / 0.47 / 0.57; test prompts: sum_helix@L16 0.27 out_circle@L24 0.94 |
| sum_helix | 3 | 0.987 ± 0.017 | 0.522 ± 0.031 | 0.134 ± 0.011 | 0.89 / 0.51 / 0.58; test prompts: sum_helix@L16 0.70 out_circle@L24 0.18 |
| sum_helix_shuf | 3 | 0.988 ± 0.008 | 0.354 ± 0.049 | 0.139 ± 0.020 | 0.84 / 0.54 / 0.55; test prompts: sum_helix@L16 0.25 out_circle@L24 0.15 |
