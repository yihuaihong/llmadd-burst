# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.71, L16 CKA(circle) 0.83

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.83 / 0.46 / 0.62 |
| circle | 3 | 0.933 ± 0.022 | 0.621 ± 0.028 | 0.119 ± 0.004 | 1.00 / 0.45 / 0.48; test prompts: sum_helix@L16 0.24 out_circle@L24 0.32 |
| none | 3 | 0.878 ± 0.040 | 0.510 ± 0.009 | 0.113 ± 0.012 | 0.88 / 0.49 / 0.60; test prompts: sum_helix@L16 0.21 out_circle@L24 0.17 |
| out_circle | 3 | 0.926 ± 0.024 | 0.250 ± 0.038 | 0.133 ± 0.012 | 0.84 / 0.48 / 0.59; test prompts: sum_helix@L16 0.20 out_circle@L24 0.09 |
| sum_helix | 3 | 1.000 ± 0.000 | 0.213 ± 0.018 | 0.119 ± 0.015 | 0.88 / 0.48 / 0.59; test prompts: sum_helix@L16 0.58 out_circle@L24 0.09 |
| sum_helix_shuf | 3 | 0.998 ± 0.002 | 0.143 ± 0.034 | 0.117 ± 0.006 | 0.84 / 0.50 / 0.61; test prompts: sum_helix@L16 0.24 out_circle@L24 0.07 |
