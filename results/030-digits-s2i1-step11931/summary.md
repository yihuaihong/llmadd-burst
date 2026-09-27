# exp1b: where is the tens digit?

base acc 0.975

## held-out R^2 (slot A), by layer

| basis | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helix | 0.14 | 0.18 | 0.21 | 0.23 | 0.24 | 0.23 | 0.24 | 0.26 | 0.29 | 0.29 | 0.32 |
| digit_1hot | 0.15 | 0.21 | 0.25 | 0.29 | 0.30 | 0.30 | 0.33 | 0.36 | 0.41 | 0.42 | 0.44 |
| digit_circ | 0.10 | 0.12 | 0.14 | 0.15 | 0.16 | 0.15 | 0.16 | 0.18 | 0.20 | 0.21 | 0.23 |
| both | 0.13 | 0.19 | 0.24 | 0.27 | 0.29 | 0.30 | 0.33 | 0.36 | 0.41 | 0.41 | 0.44 |

## slot A: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.93 (L0) | 0.77 (L0) |
| both:random_orth | 0.00 (L2) | 0.00 (L16) |
| both:tens | 0.00 (L0) | 0.77 (L0) |
| both:units | 0.92 (L0) | 0.00 (L0) |
| digit_1hot:all | 0.93 (L0) | 0.75 (L0) |
| digit_1hot:random_orth | 0.00 (L8) | 0.00 (L14) |
| digit_1hot:tens | 0.00 (L0) | 0.75 (L0) |
| digit_1hot:units | 0.93 (L0) | 0.00 (L0) |
| digit_circ:all | 0.01 (L16) | 0.02 (L4) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L2) |
| digit_circ:tens | 0.00 (L0) | 0.02 (L4) |
| digit_circ:units | 0.01 (L16) | 0.00 (L0) |
| full_swap | 0.99 (L2) | 0.98 (L0) |
| helix:all | 0.79 (L16) | 0.09 (L0) |
| helix:random_orth | 0.00 (L0) | 0.00 (L0) |
| helix:tens | 0.00 (L0) | 0.09 (L0) |
| helix:units | 0.77 (L16) | 0.00 (L0) |

## slot B: success, best layer per condition (mean over the Deltas of each kind)

| cond | units shifts | tens shifts |
|---|---|---|
| both:all | 0.92 (L4) | 0.77 (L0) |
| both:random_orth | 0.00 (L0) | 0.00 (L16) |
| both:tens | 0.00 (L0) | 0.77 (L0) |
| both:units | 0.90 (L2) | 0.00 (L0) |
| digit_1hot:all | 0.91 (L2) | 0.76 (L0) |
| digit_1hot:random_orth | 0.00 (L4) | 0.00 (L18) |
| digit_1hot:tens | 0.00 (L0) | 0.76 (L0) |
| digit_1hot:units | 0.91 (L2) | 0.00 (L0) |
| digit_circ:all | 0.01 (L16) | 0.08 (L6) |
| digit_circ:random_orth | 0.00 (L0) | 0.00 (L0) |
| digit_circ:tens | 0.00 (L0) | 0.08 (L6) |
| digit_circ:units | 0.01 (L16) | 0.00 (L0) |
| full_swap | 0.99 (L16) | 0.98 (L0) |
| helix:all | 0.84 (L16) | 0.20 (L6) |
| helix:random_orth | 0.00 (L0) | 0.00 (L10) |
| helix:tens | 0.00 (L0) | 0.20 (L6) |
| helix:units | 0.80 (L16) | 0.00 (L0) |
