# concept shaping: letters (n = {'train': 280, 'test': 188})

base geometry: emb CKA(line) 0.26, L16 CKA(line) 0.27

| geom | seeds | train | test | CKA@L16 line / line_shuf / circle |
|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.27 / 0.16 / 0.27 |
| circle | 3 | 0.976 ± 0.005 | 0.039 ± 0.013 | 0.44 / 0.02 / 0.99 |
| line | 3 | 0.988 ± 0.018 | 0.046 ± 0.011 | 0.99 / 0.01 / 0.49 |
| line_shuf | 3 | 0.995 ± 0.005 | 0.034 ± 0.012 | 0.02 / 1.00 / 0.02 |
| none | 3 | 1.000 ± 0.000 | 0.048 ± 0.009 | 0.29 / 0.17 / 0.29 |
