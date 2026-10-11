# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| none | 2 | 0.992 ± 0.009 | 0.579 ± 0.186 | 0.175 ± 0.026 | 0.70 / 0.32 / 0.40; test prompts: sum_helix@L16 0.36 out_circle@L24 0.14 |
| sum_helix | 2 | 0.949 ± 0.014 | 0.707 ± 0.068 | 0.147 ± 0.047 | 0.76 / 0.31 / 0.44; test prompts: sum_helix@L16 0.70 out_circle@L24 0.21 |
| sum_helix_shuf | 2 | 0.985 ± 0.006 | 0.556 ± 0.064 | 0.180 ± 0.024 | 0.68 / 0.34 / 0.40; test prompts: sum_helix@L16 0.35 out_circle@L24 0.13 |
