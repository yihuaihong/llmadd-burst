# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| out_circle | 2 | 1.000 ± 0.000 | 0.964 ± 0.044 | 0.162 ± 0.016 | 0.93 / 0.48 / 0.57; test prompts: sum_helix@L16 0.28 out_circle@L24 0.95 |
