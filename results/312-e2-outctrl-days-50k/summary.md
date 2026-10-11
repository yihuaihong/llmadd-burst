# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| out_circle_shuf | 5 | 0.996 ± 0.004 | 0.338 ± 0.049 | 0.126 ± 0.013 | 0.83 / 0.53 / 0.55; test prompts: sum_helix@L16 0.33 out_circle@L24 0.22 |
| out_onehot | 5 | 0.999 ± 0.001 | 0.650 ± 0.034 | 0.126 ± 0.016 | 0.91 / 0.52 / 0.56; test prompts: sum_helix@L16 0.29 out_circle@L24 0.65 |
