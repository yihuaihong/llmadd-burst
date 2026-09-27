# exp3: robustness of the three-term tens readout error

| format | acc | P(probe right \| model wrong) | probe acc (L) | wrong by tens of total (3x..9x) |
|---|---|---|---|---|
| terse | 0.76 | 0.99 | 0.99 (L29) | 3x:0.00 4x:0.04 5x:0.51 6x:0.57 7x:0.19 8x:0.16 9x:0.16 |
| spaced | 0.99 | 1.00 | 1.00 (L27) | 3x:0.00 4x:0.00 5x:0.00 6x:0.00 7x:0.02 8x:0.01 9x:0.01 |
| two_shot | 0.92 | 1.00 | 1.00 (L29) | 3x:0.00 4x:0.00 5x:0.13 6x:0.05 7x:0.03 8x:0.06 9x:0.12 |

## two-term: wrong by tens of the sum

- terse (acc 0.928): 2x:0.02 3x:0.06 4x:0.07 5x:0.06 6x:0.10 7x:0.06 8x:0.05 9x:0.09
- spaced (acc 1.000): 2x:0.00 3x:0.00 4x:0.00 5x:0.00 6x:0.00 7x:0.00 8x:0.00 9x:0.00

## unembedding: cos(readout(n), readout(n-10)) by tens of n

1x:0.61 2x:0.70 3x:0.66 4x:0.66 5x:0.65 6x:0.62 7x:0.63 8x:0.63 9x:0.63

cos(n, n-1) mean 0.71
