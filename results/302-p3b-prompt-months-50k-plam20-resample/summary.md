# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| none | 3 | 0.990 ± 0.008 | 0.641 ± 0.060 | 0.180 ± 0.003 | 0.71 / 0.33 / 0.41; test prompts: sum_helix@L16 0.34 out_circle@L24 0.15 |
| out_circle | 3 | 1.000 ± 0.000 | 0.986 ± 0.004 | 0.136 ± 0.018 | 0.80 / 0.29 / 0.44; test prompts: sum_helix@L16 0.37 out_circle@L24 0.97 |
| sum_helix | 3 | 0.968 ± 0.024 | 0.465 ± 0.039 | 0.102 ± 0.007 | 0.72 / 0.33 / 0.42; test prompts: sum_helix@L16 0.71 out_circle@L24 0.09 |
| sum_helix_shuf | 3 | 0.997 ± 0.002 | 0.284 ± 0.054 | 0.187 ± 0.023 | 0.63 / 0.35 / 0.36; test prompts: sum_helix@L16 0.20 out_circle@L24 0.07 |
