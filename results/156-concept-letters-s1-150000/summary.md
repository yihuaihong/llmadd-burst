# concept shaping: letters (n = {'train': 280, 'test': 188})

base geometry: emb CKA(line) 0.26, L16 CKA(line) 0.27

| geom | seeds | train | test | CKA@L16 line / line_shuf / circle |
|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.27 / 0.16 / 0.26 |
| circle | 3 | 0.982 ± 0.018 | 0.059 ± 0.000 | 0.43 / 0.02 / 1.00 |
| line | 3 | 0.993 ± 0.006 | 0.059 ± 0.005 | 1.00 / 0.01 / 0.48 |
| line_shuf | 3 | 0.993 ± 0.007 | 0.044 ± 0.003 | 0.02 / 1.00 / 0.03 |
| none | 3 | 0.998 ± 0.004 | 0.053 ± 0.000 | 0.30 / 0.17 / 0.29 |
