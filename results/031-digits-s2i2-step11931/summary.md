# exp1b: where is the tens digit?

base acc 0.926

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.14 | 0.18 | 0.21 | 0.25 | 0.26 | 0.25 | 0.25 | 0.26 | 0.29 | 0.29 | 0.31 |
| digit_1hot | 0.15 | 0.21 | 0.25 | 0.30 | 0.32 | 0.32 | 0.34 | 0.36 | 0.41 | 0.40 | 0.43 |
| digit_circ | 0.10 | 0.12 | 0.14 | 0.16 | 0.17 | 0.16 | 0.16 | 0.18 | 0.21 | 0.21 | 0.23 |
| both | 0.14 | 0.19 | 0.24 | 0.28 | 0.31 | 0.31 | 0.33 | 0.35 | 0.40 | 0.39 | 0.42 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.86 (L0) | 0.74 (L0) |
| both:random_orth | 0.00 (L0) | 0.00 (L4) |
| both:tens | 0.00 (L0) | 0.74 (L0) |
| both:units | 0.85 (L0) | 0.00 (L0) |
| digit_1hot:all | 0.85 (L0) | 0.71 (L0) |
| digit_1hot:random_orth | 0.00 (L0) | 0.00 (L14) |
| digit_1hot:tens | 0.00 (L0) | 0.71 (L0) |
| digit_1hot:units | 0.85 (L0) | 0.00 (L0) |
| digit_circ:all | 0.01 (L16) | 0.01 (L4) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L0) |
| digit_circ:tens | 0.00 (L0) | 0.01 (L4) |
| digit_circ:units | 0.01 (L16) | 0.00 (L0) |
| full_swap | 0.95 (L2) | 0.95 (L0) |
| helix:all | 0.71 (L16) | 0.07 (L0) |
| helix:random_orth | 0.00 (L0) | 0.00 (L2) |
| helix:tens | 0.00 (L0) | 0.07 (L0) |
| helix:units | 0.70 (L16) | 0.00 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.85 (L12) | 0.67 (L0) |
| both:random_orth | 0.00 (L0) | 0.00 (L0) |
| both:tens | 0.00 (L0) | 0.67 (L0) |
| both:units | 0.83 (L8) | 0.00 (L0) |
| digit_1hot:all | 0.83 (L8) | 0.64 (L6) |
| digit_1hot:random_orth | 0.00 (L0) | 0.00 (L0) |
| digit_1hot:tens | 0.00 (L0) | 0.64 (L6) |
| digit_1hot:units | 0.83 (L8) | 0.00 (L0) |
| digit_circ:all | 0.01 (L14) | 0.08 (L6) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L0) |
| digit_circ:tens | 0.00 (L0) | 0.08 (L6) |
| digit_circ:units | 0.01 (L14) | 0.00 (L0) |
| full_swap | 0.95 (L16) | 0.94 (L10) |
| helix:all | 0.74 (L16) | 0.19 (L6) |
| helix:random_orth | 0.00 (L0) | 0.00 (L0) |
| helix:tens | 0.00 (L0) | 0.19 (L6) |
| helix:units | 0.71 (L16) | 0.00 (L0) |
