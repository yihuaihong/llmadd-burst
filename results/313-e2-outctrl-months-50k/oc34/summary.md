# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| out_circle | 2 | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.100 ± 0.009 | 0.88 / 0.26 / 0.50; test prompts: sum_helix@L16 0.29 out_circle@L24 0.97 |
