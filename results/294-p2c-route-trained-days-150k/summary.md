# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.77, L16 CKA(circle) 0.80

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.013 | 0.000 | 0.80 / 0.51 / 0.58 |
| circle | 2 | 0.999 ± 0.002 | 0.960 ± 0.006 | 0.123 ± 0.009 | 1.00 / 0.45 / 0.48; test prompts: sum_helix@L16 0.25 out_circle@L24 0.49 |
| none | 2 | 0.973 ± 0.021 | 0.815 ± 0.016 | 0.142 ± 0.054 | 0.83 / 0.53 / 0.60; test prompts: sum_helix@L16 0.22 out_circle@L24 0.29 |
