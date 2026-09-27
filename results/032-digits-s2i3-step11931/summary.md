# exp1b: where is the tens digit?

base acc 0.880

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.14 | 0.18 | 0.21 | 0.24 | 0.25 | 0.24 | 0.25 | 0.25 | 0.28 | 0.28 | 0.30 |
| digit_1hot | 0.15 | 0.21 | 0.26 | 0.29 | 0.32 | 0.31 | 0.34 | 0.36 | 0.40 | 0.40 | 0.41 |
| digit_circ | 0.10 | 0.12 | 0.14 | 0.15 | 0.16 | 0.15 | 0.15 | 0.15 | 0.18 | 0.19 | 0.20 |
| both | 0.13 | 0.19 | 0.24 | 0.28 | 0.31 | 0.31 | 0.34 | 0.35 | 0.39 | 0.39 | 0.41 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.84 (L0) | 0.67 (L0) |
| both:random_orth | 0.00 (L16) | 0.00 (L0) |
| both:tens | 0.00 (L0) | 0.67 (L0) |
| both:units | 0.83 (L0) | 0.00 (L0) |
| digit_1hot:all | 0.83 (L0) | 0.65 (L0) |
| digit_1hot:random_orth | 0.00 (L2) | 0.00 (L4) |
| digit_1hot:tens | 0.00 (L0) | 0.65 (L0) |
| digit_1hot:units | 0.83 (L0) | 0.00 (L0) |
| digit_circ:all | 0.01 (L12) | 0.03 (L4) |
| digit_circ:random_orth | 0.00 (L18) | 0.00 (L0) |
| digit_circ:tens | 0.00 (L0) | 0.03 (L4) |
| digit_circ:units | 0.01 (L12) | 0.00 (L0) |
| full_swap | 0.93 (L2) | 0.92 (L0) |
| helix:all | 0.72 (L16) | 0.08 (L0) |
| helix:random_orth | 0.00 (L6) | 0.00 (L0) |
| helix:tens | 0.00 (L0) | 0.08 (L0) |
| helix:units | 0.71 (L16) | 0.00 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.80 (L8) | 0.62 (L0) |
| both:random_orth | 0.00 (L0) | 0.00 (L0) |
| both:tens | 0.00 (L0) | 0.62 (L0) |
| both:units | 0.78 (L6) | 0.00 (L0) |
| digit_1hot:all | 0.78 (L8) | 0.60 (L0) |
| digit_1hot:random_orth | 0.00 (L0) | 0.00 (L4) |
| digit_1hot:tens | 0.00 (L0) | 0.60 (L0) |
| digit_1hot:units | 0.78 (L8) | 0.00 (L0) |
| digit_circ:all | 0.01 (L16) | 0.09 (L2) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L16) |
| digit_circ:tens | 0.00 (L0) | 0.09 (L2) |
| digit_circ:units | 0.01 (L16) | 0.00 (L0) |
| full_swap | 0.93 (L16) | 0.87 (L0) |
| helix:all | 0.71 (L16) | 0.18 (L0) |
| helix:random_orth | 0.00 (L18) | 0.00 (L0) |
| helix:tens | 0.00 (L0) | 0.18 (L0) |
| helix:units | 0.69 (L16) | 0.00 (L0) |
