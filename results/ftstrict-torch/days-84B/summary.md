# concept shaping: days (n = {'train': 363, 'test': 197, 'ood': 300})

base geometry: emb CKA(circle) 0.71, L16 CKA(circle) 0.83

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.83 / 0.46 / 0.62 |
| circle | 3 | 0.938 ± 0.061 | 0.002 ± 0.003 | 0.131 ± 0.005 | 1.00 / 0.45 / 0.47; test prompts: sum_helix@L16 0.24 out_circle@L24 0.20 |
| circle_shuf | 3 | 0.730 ± 0.013 | 0.000 ± 0.000 | 0.129 ± 0.015 | 0.45 / 1.00 / 0.14; test prompts: sum_helix@L16 0.20 out_circle@L24 0.11 |
| line | 3 | 0.776 ± 0.041 | 0.000 ± 0.000 | 0.131 ± 0.005 | 0.51 / 0.14 / 1.00; test prompts: sum_helix@L16 0.22 out_circle@L24 0.13 |
| none | 3 | 0.873 ± 0.025 | 0.000 ± 0.000 | 0.148 ± 0.008 | 0.84 / 0.48 / 0.52; test prompts: sum_helix@L16 0.18 out_circle@L24 0.14 |
