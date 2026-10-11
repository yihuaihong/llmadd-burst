# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| out_circle_shuf | 5 | 0.987 ± 0.011 | 0.413 ± 0.043 | 0.165 ± 0.042 | 0.68 / 0.33 / 0.41; test prompts: sum_helix@L16 0.38 out_circle@L24 0.13 |
| out_onehot | 5 | 0.992 ± 0.005 | 0.733 ± 0.095 | 0.205 ± 0.030 | 0.73 / 0.32 / 0.43; test prompts: sum_helix@L16 0.37 out_circle@L24 0.35 |
