# concept shaping: letters (n = {'train': 280, 'test': 188})

base geometry: emb CKA(line) 0.24, L16 CKA(line) 0.27

| geom | seeds | train | test | CKA@L16 line / line_shuf / circle |
|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.27 / 0.17 / 0.29 |
| circle | 3 | 0.888 ± 0.028 | 0.044 ± 0.006 | 0.45 / 0.01 / 0.99 |
| line | 3 | 0.724 ± 0.080 | 0.044 ± 0.011 | 1.00 / 0.01 / 0.47 |
| line_shuf | 3 | 0.973 ± 0.005 | 0.002 ± 0.003 | 0.02 / 1.00 / 0.02 |
| none | 3 | 0.999 ± 0.002 | 0.028 ± 0.006 | 0.31 / 0.17 / 0.33 |
