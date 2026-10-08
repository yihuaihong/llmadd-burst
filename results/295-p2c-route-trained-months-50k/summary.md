# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| circle | 2 | 0.961 ± 0.031 | 0.876 ± 0.061 | 0.143 ± 0.028 | 1.00 / 0.14 / 0.44; test prompts: sum_helix@L16 0.31 out_circle@L24 0.38 |
| none | 2 | 0.995 ± 0.002 | 0.660 ± 0.072 | 0.180 ± 0.005 | 0.71 / 0.33 / 0.42; test prompts: sum_helix@L16 0.35 out_circle@L24 0.16 |
