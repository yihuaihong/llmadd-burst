# concept shaping: letters (n = {'train': 280, 'test': 188})

base geometry: emb CKA(line) 0.20, L16 CKA(line) 0.26

| geom | seeds | train | test | CKA@L16 line / line_shuf / circle |
|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.26 / 0.16 / 0.30 |
| circle | 3 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.44 / 0.01 / 1.00 |
| line | 3 | 0.994 ± 0.010 | 0.004 ± 0.003 | 1.00 / 0.01 / 0.46 |
| line_shuf | 3 | 0.999 ± 0.002 | 0.000 ± 0.000 | 0.01 / 1.00 / 0.02 |
| none | 3 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.28 / 0.17 / 0.33 |
