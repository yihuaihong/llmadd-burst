# concept shaping: months (n = {'train': 603, 'test': 357, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| circle | 3 | 0.979 ± 0.009 | 0.594 ± 0.032 | 0.141 ± 0.032 | 1.00 / 0.14 / 0.45; test prompts: sum_helix@L16 0.38 out_circle@L24 0.37 |
| circle_shuf | 3 | 0.662 ± 0.113 | 0.033 ± 0.013 | 0.152 ± 0.016 | 0.16 / 0.99 / 0.07; test prompts: sum_helix@L16 0.31 out_circle@L24 0.06 |
| line | 3 | 0.928 ± 0.014 | 0.444 ± 0.089 | 0.180 ± 0.010 | 0.50 / 0.07 / 1.00; test prompts: sum_helix@L16 0.37 out_circle@L24 0.26 |
| none | 3 | 0.980 ± 0.004 | 0.287 ± 0.068 | 0.197 ± 0.006 | 0.68 / 0.32 / 0.39; test prompts: sum_helix@L16 0.36 out_circle@L24 0.16 |
