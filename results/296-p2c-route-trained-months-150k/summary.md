# concept shaping: months (n = {'train': 576, 'test': 384, 'ood': 300})

base geometry: emb CKA(circle) 0.64, L16 CKA(circle) 0.59

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.003 | 0.000 | 0.59 / 0.29 / 0.30 |
| circle | 2 | 0.964 ± 0.022 | 0.741 ± 0.061 | 0.185 ± 0.002 | 0.99 / 0.14 / 0.46; test prompts: sum_helix@L16 0.44 out_circle@L24 0.23 |
| none | 2 | 0.982 ± 0.001 | 0.801 ± 0.057 | 0.207 ± 0.042 | 0.69 / 0.33 / 0.43; test prompts: sum_helix@L16 0.45 out_circle@L24 0.24 |
