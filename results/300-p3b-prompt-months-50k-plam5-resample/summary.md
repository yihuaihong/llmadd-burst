# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| none | 3 | 0.990 ± 0.008 | 0.641 ± 0.060 | 0.180 ± 0.003 | 0.71 / 0.33 / 0.41; test prompts: sum_helix@L16 0.34 out_circle@L24 0.15 |
| out_circle | 3 | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.087 ± 0.018 | 0.85 / 0.28 / 0.48; test prompts: sum_helix@L16 0.28 out_circle@L24 0.98 |
| sum_helix | 3 | 0.976 ± 0.006 | 0.748 ± 0.054 | 0.107 ± 0.020 | 0.77 / 0.32 / 0.44; test prompts: sum_helix@L16 0.71 out_circle@L24 0.20 |
| sum_helix_shuf | 3 | 0.981 ± 0.008 | 0.518 ± 0.084 | 0.152 ± 0.027 | 0.67 / 0.34 / 0.39; test prompts: sum_helix@L16 0.37 out_circle@L24 0.13 |
