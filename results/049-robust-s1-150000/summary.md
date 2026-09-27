# exp3: robustness of the three-term tens readout error

| format | acc | P(probe right \| model wrong) | probe acc (L) | wrong by tens of total (3x..9x) |
|---|---|---|---|---|
| terse | 0.07 | 0.42 | 0.43 (L26) | 3x:0.40 4x:0.79 5x:0.98 6x:0.92 7x:0.76 8x:0.97 9x:0.97 |
| spaced | 0.12 | 0.41 | 0.41 (L29) | 3x:0.20 4x:0.50 5x:0.89 6x:0.87 7x:0.76 8x:0.92 9x:0.94 |
| two_shot | 0.06 | 0.65 | 0.65 (L27) | 3x:0.80 4x:0.88 5x:0.94 6x:0.86 7x:0.85 8x:0.98 9x:0.98 |

## two-term: wrong by tens of the sum

- terse (acc 0.770): 2x:0.11 3x:0.15 4x:0.07 5x:0.17 6x:0.25 7x:0.19 8x:0.24 9x:0.35
- spaced (acc 0.953): 2x:0.00 3x:0.02 4x:0.01 5x:0.02 6x:0.03 7x:0.04 8x:0.07 9x:0.08

## unembedding: cos(readout(n), readout(n-10)) by tens of n

1x:0.50 2x:0.59 3x:0.56 4x:0.57 5x:0.57 6x:0.53 7x:0.56 8x:0.56 9x:0.55

cos(n, n-1) mean 0.65
