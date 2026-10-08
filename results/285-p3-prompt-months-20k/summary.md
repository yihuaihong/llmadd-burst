# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.54, L16 CKA(circle) 0.57

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.57 / 0.26 / 0.30 |
| circle | 3 | 0.791 ± 0.040 | 0.553 ± 0.027 | 0.074 ± 0.024 | 1.00 / 0.13 / 0.45; test prompts: sum_helix@L16 0.33 out_circle@L24 0.52 |
| none | 3 | 0.737 ± 0.020 | 0.478 ± 0.029 | 0.066 ± 0.007 | 0.79 / 0.29 / 0.44; test prompts: sum_helix@L16 0.27 out_circle@L24 0.23 |
| out_circle | 3 | 0.656 ± 0.018 | 0.154 ± 0.019 | 0.108 ± 0.020 | 0.66 / 0.28 / 0.36; test prompts: sum_helix@L16 0.23 out_circle@L24 0.04 |
| sum_helix | 3 | 0.935 ± 0.037 | 0.257 ± 0.011 | 0.110 ± 0.010 | 0.80 / 0.29 / 0.46; test prompts: sum_helix@L16 0.59 out_circle@L24 0.21 |
| sum_helix_shuf | 3 | 0.929 ± 0.019 | 0.235 ± 0.032 | 0.061 ± 0.022 | 0.70 / 0.30 / 0.38; test prompts: sum_helix@L16 0.20 out_circle@L24 0.14 |
