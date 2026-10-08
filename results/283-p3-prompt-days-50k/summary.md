# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| circle | 3 | 1.000 ± 0.000 | 0.868 ± 0.030 | 0.153 ± 0.003 | 1.00 / 0.45 / 0.48; test prompts: sum_helix@L16 0.24 out_circle@L24 0.56 |
| none | 3 | 0.973 ± 0.034 | 0.626 ± 0.102 | 0.131 ± 0.002 | 0.86 / 0.52 / 0.59; test prompts: sum_helix@L16 0.26 out_circle@L24 0.28 |
| out_circle | 3 | 0.922 ± 0.025 | 0.251 ± 0.027 | 0.130 ± 0.003 | 0.82 / 0.51 / 0.57; test prompts: sum_helix@L16 0.29 out_circle@L24 0.08 |
| sum_helix | 3 | 0.999 ± 0.002 | 0.359 ± 0.009 | 0.137 ± 0.013 | 0.86 / 0.51 / 0.58; test prompts: sum_helix@L16 0.60 out_circle@L24 0.13 |
| sum_helix_shuf | 3 | 0.996 ± 0.005 | 0.210 ± 0.019 | 0.133 ± 0.022 | 0.82 / 0.51 / 0.57; test prompts: sum_helix@L16 0.21 out_circle@L24 0.07 |
