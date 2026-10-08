# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| circle | 2 | 1.000 ± 0.000 | 0.877 ± 0.035 | 0.153 ± 0.005 | 1.00 / 0.44 / 0.48; test prompts: sum_helix@L16 0.24 out_circle@L24 0.56 |
| none | 2 | 0.993 ± 0.011 | 0.672 ± 0.092 | 0.132 ± 0.002 | 0.86 / 0.52 / 0.59; test prompts: sum_helix@L16 0.26 out_circle@L24 0.29 |
