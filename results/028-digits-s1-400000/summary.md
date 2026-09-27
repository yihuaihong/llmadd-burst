# exp1b: where is the tens digit?

base acc 0.626

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.14 | 0.16 | 0.20 | 0.23 | 0.25 | 0.27 | 0.28 | 0.29 | 0.32 | 0.31 | 0.32 |
| digit_1hot | 0.14 | 0.18 | 0.23 | 0.29 | 0.32 | 0.34 | 0.39 | 0.40 | 0.45 | 0.42 | 0.44 |
| digit_circ | 0.10 | 0.11 | 0.13 | 0.14 | 0.16 | 0.16 | 0.16 | 0.16 | 0.19 | 0.19 | 0.20 |
| both | 0.12 | 0.16 | 0.21 | 0.27 | 0.31 | 0.33 | 0.37 | 0.38 | 0.43 | 0.41 | 0.43 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.69 (L0) | 0.51 (L0) |
| both:random_orth | 0.00 (L14) | 0.00 (L2) |
| both:tens | 0.00 (L8) | 0.51 (L0) |
| both:units | 0.68 (L2) | 0.00 (L0) |
| digit_1hot:all | 0.68 (L2) | 0.49 (L0) |
| digit_1hot:random_orth | 0.00 (L8) | 0.00 (L2) |
| digit_1hot:tens | 0.00 (L0) | 0.49 (L0) |
| digit_1hot:units | 0.68 (L2) | 0.00 (L0) |
| digit_circ:all | 0.03 (L16) | 0.05 (L0) |
| digit_circ:random_orth | 0.00 (L16) | 0.00 (L0) |
| digit_circ:tens | 0.00 (L0) | 0.05 (L0) |
| digit_circ:units | 0.03 (L16) | 0.00 (L0) |
| full_swap | 0.75 (L10) | 0.72 (L0) |
| helix:all | 0.55 (L16) | 0.11 (L0) |
| helix:random_orth | 0.01 (L2) | 0.00 (L16) |
| helix:tens | 0.00 (L10) | 0.11 (L0) |
| helix:units | 0.55 (L16) | 0.00 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.70 (L0) | 0.55 (L0) |
| both:random_orth | 0.00 (L12) | 0.01 (L6) |
| both:tens | 0.00 (L14) | 0.55 (L0) |
| both:units | 0.69 (L0) | 0.00 (L0) |
| digit_1hot:all | 0.70 (L0) | 0.54 (L0) |
| digit_1hot:random_orth | 0.01 (L6) | 0.01 (L4) |
| digit_1hot:tens | 0.00 (L0) | 0.54 (L0) |
| digit_1hot:units | 0.70 (L0) | 0.00 (L0) |
| digit_circ:all | 0.02 (L16) | 0.09 (L4) |
| digit_circ:random_orth | 0.00 (L6) | 0.01 (L16) |
| digit_circ:tens | 0.00 (L0) | 0.09 (L4) |
| digit_circ:units | 0.02 (L16) | 0.00 (L0) |
| full_swap | 0.76 (L16) | 0.72 (L8) |
| helix:all | 0.56 (L12) | 0.17 (L2) |
| helix:random_orth | 0.00 (L10) | 0.00 (L14) |
| helix:tens | 0.00 (L0) | 0.17 (L2) |
| helix:units | 0.55 (L12) | 0.00 (L0) |
