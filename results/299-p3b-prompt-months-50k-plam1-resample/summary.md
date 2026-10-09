# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| none | 3 | 0.990 ± 0.008 | 0.641 ± 0.060 | 0.180 ± 0.003 | 0.71 / 0.33 / 0.41; test prompts: sum_helix@L16 0.34 out_circle@L24 0.15 |
| out_circle | 3 | 0.990 ± 0.011 | 0.872 ± 0.085 | 0.133 ± 0.017 | 0.78 / 0.31 / 0.46; test prompts: sum_helix@L16 0.31 out_circle@L24 0.56 |
| sum_helix | 3 | 0.976 ± 0.017 | 0.694 ± 0.100 | 0.130 ± 0.032 | 0.74 / 0.32 / 0.43; test prompts: sum_helix@L16 0.68 out_circle@L24 0.22 |
| sum_helix_shuf | 3 | 0.996 ± 0.004 | 0.639 ± 0.115 | 0.187 ± 0.015 | 0.70 / 0.33 / 0.41; test prompts: sum_helix@L16 0.34 out_circle@L24 0.17 |
