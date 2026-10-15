# concept shaping: months (n = {'train': 603, 'test': 357, 'ood': 300})

base geometry: emb CKA(circle) 0.54, L16 CKA(circle) 0.57

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.57 / 0.26 / 0.29 |
| circle | 3 | 0.827 ± 0.016 | 0.183 ± 0.082 | 0.099 ± 0.011 | 1.00 / 0.13 / 0.45; test prompts: sum_helix@L16 0.30 out_circle@L24 0.45 |
| circle_shuf | 3 | 0.507 ± 0.020 | 0.011 ± 0.005 | 0.146 ± 0.013 | 0.15 / 1.00 / 0.07; test prompts: sum_helix@L16 0.14 out_circle@L24 0.04 |
| line | 3 | 0.664 ± 0.041 | 0.102 ± 0.017 | 0.106 ± 0.002 | 0.49 / 0.07 / 1.00; test prompts: sum_helix@L16 0.31 out_circle@L24 0.26 |
| none | 3 | 0.811 ± 0.019 | 0.029 ± 0.015 | 0.102 ± 0.024 | 0.69 / 0.32 / 0.40; test prompts: sum_helix@L16 0.15 out_circle@L24 0.15 |
