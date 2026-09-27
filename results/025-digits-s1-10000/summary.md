# exp1b: where is the tens digit?

base acc 0.037

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.16 | 0.20 | 0.23 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 | 0.23 | 0.23 |
| digit_1hot | 0.20 | 0.25 | 0.30 | 0.31 | 0.30 | 0.30 | 0.29 | 0.29 | 0.29 | 0.28 | 0.28 |
| digit_circ | 0.11 | 0.13 | 0.15 | 0.16 | 0.16 | 0.16 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| both | 0.18 | 0.23 | 0.28 | 0.29 | 0.28 | 0.28 | 0.27 | 0.27 | 0.26 | 0.26 | 0.26 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.07 (L0) | 0.05 (L0) |
| both:random_orth | 0.02 (L2) | 0.01 (L0) |
| both:tens | 0.01 (L0) | 0.05 (L0) |
| both:units | 0.07 (L0) | 0.01 (L0) |
| digit_1hot:all | 0.07 (L0) | 0.05 (L0) |
| digit_1hot:random_orth | 0.04 (L0) | 0.01 (L0) |
| digit_1hot:tens | 0.00 (L0) | 0.05 (L0) |
| digit_1hot:units | 0.07 (L0) | 0.01 (L0) |
| digit_circ:all | 0.03 (L0) | 0.01 (L0) |
| digit_circ:random_orth | 0.01 (L2) | 0.01 (L0) |
| digit_circ:tens | 0.00 (L0) | 0.01 (L0) |
| digit_circ:units | 0.03 (L0) | 0.01 (L0) |
| full_swap | 0.24 (L0) | 0.11 (L6) |
| helix:all | 0.05 (L0) | 0.02 (L2) |
| helix:random_orth | 0.01 (L0) | 0.01 (L0) |
| helix:tens | 0.00 (L0) | 0.02 (L2) |
| helix:units | 0.05 (L0) | 0.01 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.23 (L2) | 0.24 (L2) |
| both:random_orth | 0.01 (L4) | 0.00 (L2) |
| both:tens | 0.01 (L2) | 0.24 (L2) |
| both:units | 0.21 (L4) | 0.00 (L0) |
| digit_1hot:all | 0.21 (L2) | 0.21 (L2) |
| digit_1hot:random_orth | 0.01 (L10) | 0.00 (L0) |
| digit_1hot:tens | 0.00 (L0) | 0.21 (L2) |
| digit_1hot:units | 0.21 (L2) | 0.00 (L0) |
| digit_circ:all | 0.02 (L10) | 0.03 (L2) |
| digit_circ:random_orth | 0.00 (L2) | 0.00 (L8) |
| digit_circ:tens | 0.00 (L0) | 0.03 (L2) |
| digit_circ:units | 0.02 (L10) | 0.00 (L0) |
| full_swap | 0.65 (L6) | 0.69 (L6) |
| helix:all | 0.13 (L2) | 0.07 (L0) |
| helix:random_orth | 0.01 (L0) | 0.00 (L2) |
| helix:tens | 0.00 (L0) | 0.07 (L0) |
| helix:units | 0.12 (L6) | 0.00 (L0) |
