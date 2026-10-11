# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| none | 3 | 0.973 ± 0.034 | 0.626 ± 0.102 | 0.131 ± 0.002 | 0.86 / 0.52 / 0.59; test prompts: sum_helix@L16 0.26 out_circle@L24 0.28 |
| num_helix | 3 | 0.997 ± 0.005 | 0.751 ± 0.053 | 0.121 ± 0.013 | 0.90 / 0.51 / 0.58; test prompts: sum_helix@L16 0.23 out_circle@L24 0.37 |
| num_helix_shuf | 3 | 0.999 ± 0.002 | 0.629 ± 0.025 | 0.176 ± 0.020 | 0.88 / 0.52 / 0.57; test prompts: sum_helix@L16 0.33 out_circle@L24 0.33 |
