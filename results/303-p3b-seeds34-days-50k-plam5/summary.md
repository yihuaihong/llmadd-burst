# concept shaping: days (n = {'train': 336, 'test': 224, 'ood': 300})

base geometry: emb CKA(circle) 0.70, L16 CKA(circle) 0.81

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.000 | 0.81 / 0.50 / 0.58 |
| none | 2 | 0.981 ± 0.023 | 0.663 ± 0.022 | 0.132 ± 0.002 | 0.86 / 0.51 / 0.59; test prompts: sum_helix@L16 0.24 out_circle@L24 0.30 |
| sum_helix | 2 | 0.996 ± 0.006 | 0.578 ± 0.009 | 0.148 ± 0.016 | 0.91 / 0.52 / 0.58; test prompts: sum_helix@L16 0.70 out_circle@L24 0.20 |
| sum_helix_shuf | 2 | 0.963 ± 0.053 | 0.342 ± 0.009 | 0.153 ± 0.019 | 0.84 / 0.53 / 0.57; test prompts: sum_helix@L16 0.28 out_circle@L24 0.16 |
