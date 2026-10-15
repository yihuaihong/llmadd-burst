# concept shaping: days (n = {'train': 363, 'test': 197, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.003 | 0.020 | 0.000 | 0.81 / 0.50 / 0.58 |
| circle | 3 | 0.976 ± 0.021 | 0.102 ± 0.074 | 0.170 ± 0.009 | 1.00 / 0.45 / 0.47; test prompts: sum_helix@L16 0.29 out_circle@L24 0.33 |
| circle_shuf | 3 | 0.884 ± 0.013 | 0.005 ± 0.009 | 0.137 ± 0.007 | 0.45 / 1.00 / 0.15; test prompts: sum_helix@L16 0.24 out_circle@L24 0.14 |
| line | 3 | 0.904 ± 0.015 | 0.000 ± 0.000 | 0.124 ± 0.016 | 0.51 / 0.17 / 1.00; test prompts: sum_helix@L16 0.24 out_circle@L24 0.14 |
| none | 3 | 0.980 ± 0.014 | 0.139 ± 0.073 | 0.137 ± 0.009 | 0.85 / 0.52 / 0.57; test prompts: sum_helix@L16 0.23 out_circle@L24 0.20 |
