# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.64, L16 CKA(circle) 0.59

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.003 | 0.000 | 0.59 / 0.29 / 0.30 |
| none | 3 | 0.986 ± 0.008 | 0.790 ± 0.045 | 0.204 ± 0.030 | 0.69 / 0.33 / 0.42; test prompts: sum_helix@L16 0.46 out_circle@L24 0.23 |
