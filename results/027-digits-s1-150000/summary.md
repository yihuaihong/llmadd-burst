# exp1b: where is the tens digit?

base acc 0.823

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.13 | 0.16 | 0.20 | 0.23 | 0.23 | 0.24 | 0.25 | 0.25 | 0.28 | 0.29 | 0.30 |
| digit_1hot | 0.14 | 0.19 | 0.24 | 0.27 | 0.28 | 0.29 | 0.32 | 0.33 | 0.37 | 0.38 | 0.39 |
| digit_circ | 0.10 | 0.11 | 0.13 | 0.14 | 0.14 | 0.14 | 0.14 | 0.13 | 0.17 | 0.17 | 0.18 |
| both | 0.12 | 0.17 | 0.23 | 0.26 | 0.27 | 0.27 | 0.31 | 0.31 | 0.36 | 0.36 | 0.38 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.77 (L0) | 0.58 (L0) |
| both:random_orth | 0.00 (L2) | 0.01 (L16) |
| both:tens | 0.00 (L16) | 0.58 (L0) |
| both:units | 0.76 (L0) | 0.00 (L0) |
| digit_1hot:all | 0.76 (L10) | 0.57 (L0) |
| digit_1hot:random_orth | 0.00 (L12) | 0.01 (L16) |
| digit_1hot:tens | 0.00 (L0) | 0.57 (L0) |
| digit_1hot:units | 0.76 (L10) | 0.00 (L0) |
| digit_circ:all | 0.02 (L12) | 0.07 (L2) |
| digit_circ:random_orth | 0.00 (L6) | 0.00 (L4) |
| digit_circ:tens | 0.00 (L0) | 0.07 (L2) |
| digit_circ:units | 0.02 (L12) | 0.00 (L0) |
| full_swap | 0.86 (L2) | 0.85 (L0) |
| helix:all | 0.67 (L12) | 0.12 (L0) |
| helix:random_orth | 0.00 (L8) | 0.00 (L0) |
| helix:tens | 0.00 (L0) | 0.12 (L0) |
| helix:units | 0.66 (L12) | 0.00 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.79 (L8) | 0.59 (L0) |
| both:random_orth | 0.00 (L2) | 0.01 (L6) |
| both:tens | 0.00 (L0) | 0.59 (L0) |
| both:units | 0.78 (L10) | 0.00 (L0) |
| digit_1hot:all | 0.78 (L10) | 0.57 (L0) |
| digit_1hot:random_orth | 0.00 (L0) | 0.00 (L0) |
| digit_1hot:tens | 0.00 (L0) | 0.57 (L0) |
| digit_1hot:units | 0.78 (L10) | 0.00 (L0) |
| digit_circ:all | 0.03 (L8) | 0.10 (L10) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L8) |
| digit_circ:tens | 0.00 (L0) | 0.10 (L10) |
| digit_circ:units | 0.03 (L8) | 0.00 (L0) |
| full_swap | 0.88 (L0) | 0.86 (L0) |
| helix:all | 0.71 (L10) | 0.19 (L4) |
| helix:random_orth | 0.00 (L0) | 0.00 (L4) |
| helix:tens | 0.00 (L0) | 0.19 (L4) |
| helix:units | 0.69 (L10) | 0.00 (L0) |
