# exp1b: where is the tens digit?

base acc 0.383

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.16 | 0.21 | 0.25 | 0.26 | 0.26 | 0.26 | 0.25 | 0.26 | 0.28 | 0.29 | 0.29 |
| digit_1hot | 0.20 | 0.27 | 0.32 | 0.33 | 0.33 | 0.33 | 0.34 | 0.35 | 0.37 | 0.38 | 0.38 |
| digit_circ | 0.11 | 0.13 | 0.15 | 0.16 | 0.15 | 0.15 | 0.14 | 0.15 | 0.16 | 0.17 | 0.17 |
| both | 0.18 | 0.26 | 0.31 | 0.32 | 0.32 | 0.31 | 0.32 | 0.33 | 0.35 | 0.36 | 0.36 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.46 (L2) | 0.40 (L0) |
| both:random_orth | 0.01 (L10) | 0.01 (L2) |
| both:tens | 0.00 (L4) | 0.40 (L0) |
| both:units | 0.45 (L2) | 0.00 (L0) |
| digit_1hot:all | 0.46 (L2) | 0.40 (L0) |
| digit_1hot:random_orth | 0.01 (L8) | 0.01 (L2) |
| digit_1hot:tens | 0.00 (L0) | 0.40 (L0) |
| digit_1hot:units | 0.46 (L2) | 0.00 (L0) |
| digit_circ:all | 0.04 (L2) | 0.05 (L0) |
| digit_circ:random_orth | 0.00 (L2) | 0.00 (L4) |
| digit_circ:tens | 0.00 (L0) | 0.05 (L0) |
| digit_circ:units | 0.04 (L2) | 0.00 (L0) |
| full_swap | 0.64 (L2) | 0.55 (L0) |
| helix:all | 0.38 (L12) | 0.13 (L0) |
| helix:random_orth | 0.01 (L12) | 0.00 (L6) |
| helix:tens | 0.00 (L0) | 0.13 (L0) |
| helix:units | 0.37 (L12) | 0.00 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.56 (L8) | 0.44 (L0) |
| both:random_orth | 0.00 (L0) | 0.01 (L4) |
| both:tens | 0.00 (L6) | 0.44 (L0) |
| both:units | 0.55 (L6) | 0.00 (L0) |
| digit_1hot:all | 0.56 (L6) | 0.43 (L0) |
| digit_1hot:random_orth | 0.00 (L10) | 0.01 (L4) |
| digit_1hot:tens | 0.00 (L0) | 0.43 (L0) |
| digit_1hot:units | 0.56 (L6) | 0.00 (L0) |
| digit_circ:all | 0.04 (L10) | 0.13 (L0) |
| digit_circ:random_orth | 0.00 (L8) | 0.00 (L8) |
| digit_circ:tens | 0.00 (L0) | 0.13 (L0) |
| digit_circ:units | 0.04 (L10) | 0.00 (L0) |
| full_swap | 0.65 (L6) | 0.60 (L2) |
| helix:all | 0.49 (L10) | 0.18 (L2) |
| helix:random_orth | 0.00 (L2) | 0.01 (L8) |
| helix:tens | 0.00 (L0) | 0.18 (L2) |
| helix:units | 0.47 (L10) | 0.00 (L0) |
