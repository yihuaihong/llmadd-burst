# concept shaping: letters (n = {'train': 280, 'test': 188})

base geometry: emb CKA(line) 0.22, L16 CKA(line) 0.27

| geom | seeds | train | test | CKA@L16 line / line_shuf / circle |
|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.27 / 0.17 / 0.30 |
| circle | 3 | 0.954 ± 0.012 | 0.005 ± 0.000 | 0.44 / 0.01 / 1.00 |
| line | 3 | 0.739 ± 0.049 | 0.020 ± 0.006 | 1.00 / 0.01 / 0.47 |
| line_shuf | 3 | 0.943 ± 0.042 | 0.002 ± 0.003 | 0.01 / 1.00 / 0.02 |
| none | 3 | 1.000 ± 0.000 | 0.002 ± 0.003 | 0.28 / 0.17 / 0.32 |
