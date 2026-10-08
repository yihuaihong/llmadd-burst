# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.77, L16 CKA(circle) 0.80

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.013 | 0.000 | 0.80 / 0.51 / 0.58 |
| circle | 3 | 0.999 ± 0.002 | 0.958 ± 0.005 | 0.131 ± 0.015 | 1.00 / 0.45 / 0.48; test prompts: sum_helix@L16 0.24 out_circle@L24 0.49 |
| none | 3 | 0.980 ± 0.019 | 0.795 ± 0.037 | 0.126 ± 0.047 | 0.83 / 0.53 / 0.59; test prompts: sum_helix@L16 0.22 out_circle@L24 0.27 |
| out_circle | 3 | 0.964 ± 0.015 | 0.304 ± 0.037 | 0.139 ± 0.021 | 0.81 / 0.51 / 0.58; test prompts: sum_helix@L16 0.25 out_circle@L24 0.06 |
| sum_helix | 3 | 0.985 ± 0.012 | 0.418 ± 0.037 | 0.141 ± 0.012 | 0.84 / 0.53 / 0.57; test prompts: sum_helix@L16 0.61 out_circle@L24 0.15 |
| sum_helix_shuf | 3 | 0.943 ± 0.083 | 0.323 ± 0.027 | 0.133 ± 0.015 | 0.81 / 0.53 / 0.58; test prompts: sum_helix@L16 0.24 out_circle@L24 0.11 |
