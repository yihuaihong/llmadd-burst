# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.77, L16 CKA(circle) 0.80

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.013 | 0.000 | 0.80 / 0.51 / 0.58 |
| none | 3 | 0.980 ± 0.019 | 0.795 ± 0.037 | 0.126 ± 0.047 | 0.83 / 0.53 / 0.59; test prompts: sum_helix@L16 0.22 out_circle@L24 0.27 |
