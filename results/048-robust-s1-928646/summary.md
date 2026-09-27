# exp3: robustness of the three-term tens readout error

| format | acc | P(probe right \| model wrong) | probe acc (L) | wrong by tens of total (3x..9x) |
|---|---|---|---|---|
| terse | 0.11 | 0.74 | 0.75 (L27) | 3x:0.40 4x:0.46 5x:0.97 6x:0.99 7x:0.87 8x:0.84 9x:0.91 |
| spaced | 0.43 | 0.93 | 0.94 (L31) | 3x:0.20 4x:0.08 5x:0.34 6x:0.54 7x:0.53 8x:0.59 9x:0.65 |
| two_shot | 0.34 | 0.93 | 0.93 (L28) | 3x:0.00 4x:0.04 5x:0.31 6x:0.58 7x:0.51 8x:0.70 9x:0.82 |

## two-term: wrong by tens of the sum

- terse (acc 0.818): 2x:0.04 3x:0.08 4x:0.11 5x:0.09 6x:0.15 7x:0.19 8x:0.20 9x:0.27
- spaced (acc 0.997): 2x:0.00 3x:0.01 4x:0.02 5x:0.00 6x:0.00 7x:0.00 8x:0.00 9x:0.00

## unembedding: cos(readout(n), readout(n-10)) by tens of n

1x:0.61 2x:0.70 3x:0.67 4x:0.66 5x:0.65 6x:0.62 7x:0.63 8x:0.63 9x:0.63

cos(n, n-1) mean 0.71
