# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.64, L16 CKA(circle) 0.59

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.003 | 0.000 | 0.59 / 0.29 / 0.30 |
| circle | 3 | 0.969 ± 0.018 | 0.734 ± 0.044 | 0.192 ± 0.013 | 1.00 / 0.14 / 0.46; test prompts: sum_helix@L16 0.44 out_circle@L24 0.23 |
| none | 3 | 0.986 ± 0.008 | 0.790 ± 0.045 | 0.204 ± 0.030 | 0.69 / 0.33 / 0.42; test prompts: sum_helix@L16 0.46 out_circle@L24 0.23 |
| out_circle | 3 | 0.942 ± 0.017 | 0.477 ± 0.025 | 0.140 ± 0.022 | 0.64 / 0.29 / 0.33; test prompts: sum_helix@L16 0.19 out_circle@L24 0.07 |
| sum_helix | 3 | 0.968 ± 0.025 | 0.568 ± 0.019 | 0.162 ± 0.008 | 0.67 / 0.33 / 0.41; test prompts: sum_helix@L16 0.61 out_circle@L24 0.12 |
| sum_helix_shuf | 3 | 0.978 ± 0.004 | 0.536 ± 0.010 | 0.163 ± 0.035 | 0.65 / 0.32 / 0.36; test prompts: sum_helix@L16 0.17 out_circle@L24 0.11 |
