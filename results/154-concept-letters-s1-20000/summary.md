# concept shaping: letters (n = {'train': 280, 'test': 188})

base geometry: emb CKA(line) 0.25, L16 CKA(line) 0.27

| geom | seeds | train | test | CKA@L16 line / line_shuf / circle |
|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.27 / 0.16 / 0.28 |
| circle | 3 | 0.833 ± 0.032 | 0.071 ± 0.012 | 0.44 / 0.01 / 0.99 |
| line | 3 | 0.708 ± 0.070 | 0.085 ± 0.024 | 1.00 / 0.01 / 0.47 |
| line_shuf | 3 | 0.882 ± 0.038 | 0.025 ± 0.008 | 0.02 / 1.00 / 0.02 |
| none | 3 | 0.988 ± 0.011 | 0.048 ± 0.009 | 0.27 / 0.16 / 0.29 |
