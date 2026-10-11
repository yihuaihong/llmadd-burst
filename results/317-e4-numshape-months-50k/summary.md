# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| none | 3 | 0.990 ± 0.008 | 0.641 ± 0.060 | 0.180 ± 0.003 | 0.71 / 0.33 / 0.41; test prompts: sum_helix@L16 0.34 out_circle@L24 0.15 |
| num_helix | 3 | 0.995 ± 0.003 | 0.755 ± 0.057 | 0.078 ± 0.005 | 0.75 / 0.30 / 0.45; test prompts: sum_helix@L16 0.34 out_circle@L24 0.19 |
| num_helix_shuf | 3 | 0.995 ± 0.006 | 0.595 ± 0.099 | 0.220 ± 0.033 | 0.72 / 0.32 / 0.40; test prompts: sum_helix@L16 0.45 out_circle@L24 0.13 |
