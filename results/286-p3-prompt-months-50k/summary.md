# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.57, L16 CKA(circle) 0.65

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.65 / 0.32 / 0.34 |
| circle | 3 | 0.955 ± 0.024 | 0.869 ± 0.045 | 0.158 ± 0.032 | 1.00 / 0.14 / 0.44; test prompts: sum_helix@L16 0.31 out_circle@L24 0.41 |
| none | 3 | 0.990 ± 0.008 | 0.641 ± 0.060 | 0.180 ± 0.003 | 0.71 / 0.33 / 0.41; test prompts: sum_helix@L16 0.34 out_circle@L24 0.15 |
| out_circle | 3 | 0.951 ± 0.010 | 0.321 ± 0.047 | 0.130 ± 0.007 | 0.66 / 0.32 / 0.36; test prompts: sum_helix@L16 0.26 out_circle@L24 0.05 |
| sum_helix | 3 | 0.981 ± 0.003 | 0.545 ± 0.068 | 0.130 ± 0.010 | 0.74 / 0.32 / 0.42; test prompts: sum_helix@L16 0.60 out_circle@L24 0.15 |
| sum_helix_shuf | 3 | 0.974 ± 0.011 | 0.418 ± 0.053 | 0.144 ± 0.032 | 0.71 / 0.33 / 0.41; test prompts: sum_helix@L16 0.15 out_circle@L24 0.11 |
