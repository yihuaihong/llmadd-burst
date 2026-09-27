# exp1b: where is the tens digit?

base acc 0.868

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.14 | 0.17 | 0.22 | 0.25 | 0.25 | 0.25 | 0.26 | 0.27 | 0.29 | 0.29 | 0.32 |
| digit_1hot | 0.15 | 0.20 | 0.26 | 0.31 | 0.31 | 0.32 | 0.37 | 0.40 | 0.44 | 0.43 | 0.45 |
| digit_circ | 0.10 | 0.12 | 0.15 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | 0.19 | 0.19 | 0.21 |
| both | 0.13 | 0.19 | 0.25 | 0.29 | 0.30 | 0.31 | 0.35 | 0.38 | 0.42 | 0.41 | 0.44 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.87 (L4) | 0.71 (L0) |
| both:random_orth | 0.00 (L0) | 0.00 (L16) |
| both:tens | 0.00 (L0) | 0.71 (L0) |
| both:units | 0.86 (L2) | 0.00 (L0) |
| digit_1hot:all | 0.86 (L2) | 0.69 (L0) |
| digit_1hot:random_orth | 0.00 (L18) | 0.00 (L10) |
| digit_1hot:tens | 0.00 (L0) | 0.69 (L0) |
| digit_1hot:units | 0.86 (L2) | 0.00 (L0) |
| digit_circ:all | 0.01 (L2) | 0.03 (L4) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L12) |
| digit_circ:tens | 0.00 (L0) | 0.03 (L4) |
| digit_circ:units | 0.01 (L2) | 0.00 (L0) |
| full_swap | 0.94 (L2) | 0.93 (L2) |
| helix:all | 0.72 (L2) | 0.10 (L0) |
| helix:random_orth | 0.00 (L18) | 0.00 (L6) |
| helix:tens | 0.00 (L0) | 0.10 (L0) |
| helix:units | 0.71 (L2) | 0.00 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.85 (L8) | 0.69 (L0) |
| both:random_orth | 0.00 (L4) | 0.00 (L2) |
| both:tens | 0.00 (L0) | 0.69 (L0) |
| both:units | 0.83 (L12) | 0.00 (L0) |
| digit_1hot:all | 0.83 (L10) | 0.67 (L0) |
| digit_1hot:random_orth | 0.00 (L2) | 0.00 (L4) |
| digit_1hot:tens | 0.00 (L0) | 0.67 (L0) |
| digit_1hot:units | 0.83 (L10) | 0.00 (L0) |
| digit_circ:all | 0.02 (L4) | 0.11 (L0) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L16) |
| digit_circ:tens | 0.00 (L0) | 0.11 (L0) |
| digit_circ:units | 0.02 (L4) | 0.00 (L0) |
| full_swap | 0.92 (L16) | 0.89 (L8) |
| helix:all | 0.76 (L16) | 0.23 (L6) |
| helix:random_orth | 0.00 (L6) | 0.00 (L12) |
| helix:tens | 0.00 (L0) | 0.23 (L6) |
| helix:units | 0.74 (L16) | 0.00 (L0) |
