# concept shaping: days (n = {'train': 363, 'test': 197, 'ood': 300})

base geometry: emb CKA(circle) 0.77, L16 CKA(circle) 0.80

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.014 | 0.005 | 0.000 | 0.80 / 0.51 / 0.58 |
| circle | 3 | 0.983 ± 0.010 | 0.124 ± 0.031 | 0.137 ± 0.013 | 1.00 / 0.45 / 0.48; test prompts: sum_helix@L16 0.31 out_circle@L24 0.37 |
| circle_shuf | 3 | 0.830 ± 0.057 | 0.002 ± 0.003 | 0.118 ± 0.014 | 0.45 / 1.00 / 0.14; test prompts: sum_helix@L16 0.29 out_circle@L24 0.09 |
| line | 3 | 0.915 ± 0.034 | 0.134 ± 0.031 | 0.143 ± 0.009 | 0.51 / 0.16 / 1.00; test prompts: sum_helix@L16 0.38 out_circle@L24 0.19 |
| none | 3 | 0.935 ± 0.059 | 0.191 ± 0.087 | 0.116 ± 0.012 | 0.83 / 0.53 / 0.59; test prompts: sum_helix@L16 0.27 out_circle@L24 0.24 |
