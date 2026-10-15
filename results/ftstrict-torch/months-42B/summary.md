# concept shaping: months (n = {'train': 603, 'test': 357, 'ood': 300})

base geometry: emb CKA(circle) 0.52, L16 CKA(circle) 0.59

| geom | seeds | train | test | ood | CKA@L16 circle / circle_shuf / line |
|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.59 / 0.28 / 0.30 |
| circle | 3 | 0.779 ± 0.004 | 0.041 ± 0.015 | 0.090 ± 0.013 | 1.00 / 0.13 / 0.45; test prompts: sum_helix@L16 0.25 out_circle@L24 0.25 |
| circle_shuf | 3 | 0.363 ± 0.017 | 0.000 ± 0.000 | 0.133 ± 0.009 | 0.14 / 1.00 / 0.07; test prompts: sum_helix@L16 0.20 out_circle@L24 0.06 |
| line | 3 | 0.580 ± 0.014 | 0.012 ± 0.004 | 0.088 ± 0.013 | 0.48 / 0.07 / 1.00; test prompts: sum_helix@L16 0.27 out_circle@L24 0.12 |
| none | 3 | 0.709 ± 0.028 | 0.002 ± 0.002 | 0.082 ± 0.005 | 0.76 / 0.31 / 0.43; test prompts: sum_helix@L16 0.16 out_circle@L24 0.13 |
