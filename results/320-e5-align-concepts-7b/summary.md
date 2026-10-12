# P2: number route (shots 0)

## sum route (layer with the highest consistency; null = k permuted within item)

| ckpt | domain | model acc | layer | probe acc (numeric) | slope | r2 (null p95) | consistency (null p95) | line acc (null p95) | mod acc (null p95) | offsets |
|---|---|---|---|---|---|---|---|---|---|---|

## operand route ("Q: item" decoded on the number line; layer with the highest |spearman|)

| ckpt | domain | layer | probe acc (numeric) | spearman | decoded |
|---|---|---|---|---|---|
| main | planets | L4 | 0.98 | 0.94 | [100, 96, 100, 101, 101, 102] |
| main | elements | L18 | 0.98 | -0.78 | [93, 91, 111, 80, 59, 80, 89, 79, 49] |
| main | presidents | L6 | 0.99 | -0.78 | [99, 100, 102, 105, 96, 101, 93, 96, 96, 95, 95, 90] |
| main | greek | L12 | 0.98 | 0.87 | [87, 93, 91, 87, 93, 97, 97, 103, 97, 93, 103] |
| main | ordinals | L28 | 0.97 | 1.00 | [82, 82, 83, 83, 83, 86, 87, 88, 89, 89] |
| main | solfege | L26 | 0.97 | 0.79 | [80, 72, 70, 88, 82, 118, 93] |
| stage1-step10000-tokens42B | planets | L16 | 0.97 | 0.94 | [106, 93, 108, 110, 110, 110] |
| stage1-step10000-tokens42B | elements | L22 | 0.96 | 0.55 | [102, 102, 112, 108, 108, 98, 108, 113, 112] |
| stage1-step10000-tokens42B | presidents | L28 | 0.95 | 0.49 | [98, 103, 92, 99, 98, 99, 113, 104, 105, 108, 98, 102] |
| stage1-step10000-tokens42B | greek | L18 | 0.97 | 0.69 | [108, 108, 108, 103, 109, 103, 108, 118, 108, 113, 110] |
| stage1-step10000-tokens42B | ordinals | L22 | 0.96 | 1.00 | [99, 102, 103, 104, 104, 108, 108, 108, 109, 109] |
| stage1-step10000-tokens42B | solfege | L8 | 0.97 | -0.79 | [101, 101, 101, 101, 96, 95, 90] |
| stage1-step150000-tokens630B | planets | L30 | 0.96 | 0.94 | [90, 68, 98, 98, 98, 98] |
| stage1-step150000-tokens630B | elements | L18 | 0.97 | 0.55 | [108, 106, 114, 60, 129, 80, 119, 124, 130] |
| stage1-step150000-tokens630B | presidents | L2 | 0.99 | -0.60 | [100, 92, 102, 105, 102, 101, 103, 12, 6, 12, 92, 91] |
| stage1-step150000-tokens630B | greek | L10 | 0.99 | 0.57 | [92, 98, 104, 94, 88, 84, 99, 104, 108, 104, 102] |
| stage1-step150000-tokens630B | ordinals | L20 | 0.97 | 1.00 | [100, 102, 103, 104, 105, 106, 107, 108, 109, 110] |
| stage1-step150000-tokens630B | solfege | L4 | 0.99 | 0.79 | [2, 2, 91, 92, 92, 100, 91] |
| stage1-step2000-tokens9B | planets | L20 | 0.32 | -0.66 | [112, 111, 21, 21, 111, 11] |
| stage1-step2000-tokens9B | elements | L20 | 0.32 | 0.57 | [12, 10, 10, 12, 10, 11, 10, 14, 20] |
| stage1-step2000-tokens9B | presidents | L12 | 0.33 | -0.83 | [109, 108, 103, 110, 102, 99, 102, 108, 90, 98, 88, 93] |
| stage1-step2000-tokens9B | greek | L30 | 0.26 | -0.78 | [118, 108, 108, 108, 108, 114, 102, 12, 102, 12, 12] |
| stage1-step2000-tokens9B | ordinals | L18 | 0.33 | 0.99 | [8, 108, 98, 108, 108, 108, 108, 108, 108, 108] |
| stage1-step2000-tokens9B | solfege | L26 | 0.27 | 0.46 | [10, 11, 10, 113, 11, 102, 10] |
| stage1-step400000-tokens1678B | planets | L6 | 0.99 | 0.83 | [94, 86, 92, 94, 94, 98] |
| stage1-step400000-tokens1678B | elements | L12 | 0.98 | 0.37 | [92, 92, 104, 94, 93, 92, 88, 104, 141] |
| stage1-step400000-tokens1678B | presidents | L14 | 0.98 | -0.40 | [88, 83, 92, 95, 96, 111, 93, 94, 86, 83, 78, 84] |
| stage1-step400000-tokens1678B | greek | L30 | 0.97 | -0.71 | [123, 129, 133, 123, 73, 59, 123, 113, 33, 93, 33] |
| stage1-step400000-tokens1678B | ordinals | L8 | 0.99 | 1.00 | [92, 92, 93, 94, 95, 96, 97, 98, 99, 104] |
| stage1-step400000-tokens1678B | solfege | L2 | 0.98 | 1.00 | [0, 1, 91, 100, 100, 100, 101] |
| stage1-step50000-tokens210B | planets | L8 | 0.98 | 0.94 | [96, 86, 99, 110, 110, 110] |
| stage1-step50000-tokens210B | elements | L2 | 0.99 | 0.38 | [101, 101, 20, 90, 90, 91, 100, 101, 111] |
| stage1-step50000-tokens210B | presidents | L18 | 0.97 | -0.48 | [99, 83, 92, 105, 102, 111, 103, 96, 86, 77, 77, 92] |
| stage1-step50000-tokens210B | greek | L20 | 0.98 | 0.53 | [88, 88, 104, 92, 88, 93, 88, 103, 108, 94, 92] |
| stage1-step50000-tokens210B | ordinals | L12 | 0.98 | 1.00 | [99, 102, 103, 104, 105, 106, 107, 108, 109, 109] |
| stage1-step50000-tokens210B | solfege | L8 | 0.98 | -0.96 | [102, 98, 91, 90, 90, 89, 89] |

## alignment: item i vs number i + o (centred cosine, p vs 2000 permutations; CKA, p vs 300)

| ckpt | domain | site | o | cos (null) p | CKA (null) p |
|---|---|---|---|---|---|
| main | planets | emb_in | 0 | -0.001 (0.000) 0.494 | 0.98 (0.96) 0.020 |
| main | planets | emb_in | 1 | 0.009 (-0.000) 0.181 | 0.97 (0.96) 0.126 |
| main | planets | emb_out | 0 | 0.007 (0.000) 0.169 | 0.94 (0.93) 0.256 |
| main | planets | emb_out | 1 | 0.010 (-0.000) 0.096 | 0.95 (0.94) 0.229 |
| main | planets | L14 | 0 | 0.030 (0.000) 0.054 | 0.89 (0.83) 0.086 |
| main | planets | L14 | 1 | 0.023 (0.000) 0.121 | 0.92 (0.86) 0.096 |
| main | elements | emb_in | 0 | 0.000 (0.000) 0.512 | 0.93 (0.90) 0.136 |
| main | elements | emb_in | 1 | 0.016 (0.000) 0.018 | 0.92 (0.88) 0.116 |
| main | elements | emb_out | 0 | 0.001 (0.000) 0.432 | 0.91 (0.90) 0.419 |
| main | elements | emb_out | 1 | 0.015 (-0.000) 0.001 | 0.91 (0.89) 0.056 |
| main | elements | L4 | 0 | 0.008 (0.000) 0.199 | 0.86 (0.85) 0.355 |
| main | elements | L4 | 1 | 0.030 (0.000) 0.000 | 0.87 (0.85) 0.206 |
| main | presidents | emb_in | 0 | 0.007 (-0.000) 0.115 | 0.88 (0.85) 0.186 |
| main | presidents | emb_in | 1 | 0.004 (0.000) 0.250 | 0.89 (0.86) 0.133 |
| main | presidents | emb_out | 0 | 0.002 (0.000) 0.323 | 0.86 (0.86) 0.292 |
| main | presidents | emb_out | 1 | 0.002 (0.000) 0.332 | 0.87 (0.86) 0.106 |
| main | presidents | L8 | 0 | 0.010 (0.000) 0.138 | 0.78 (0.74) 0.053 |
| main | presidents | L8 | 1 | 0.002 (-0.000) 0.394 | 0.77 (0.75) 0.130 |
| main | greek | emb_in | 0 | -0.000 (-0.000) 0.510 | 0.92 (0.89) 0.063 |
| main | greek | emb_in | 1 | 0.009 (0.000) 0.031 | 0.93 (0.88) 0.020 |
| main | greek | emb_out | 0 | 0.008 (0.000) 0.044 | 0.91 (0.87) 0.020 |
| main | greek | emb_out | 1 | 0.013 (0.000) 0.006 | 0.91 (0.86) 0.003 |
| main | greek | L10 | 0 | 0.012 (0.000) 0.141 | 0.70 (0.73) 0.674 |
| main | greek | L10 | 1 | 0.046 (0.000) 0.000 | 0.78 (0.75) 0.203 |
| main | ordinals | emb_in | 0 | 0.057 (0.000) 0.100 | 0.96 (0.84) 0.003 |
| main | ordinals | emb_in | 1 | 0.334 (0.001) 0.000 | 0.97 (0.83) 0.003 |
| main | ordinals | emb_out | 0 | 0.114 (0.001) 0.053 | 0.96 (0.73) 0.003 |
| main | ordinals | emb_out | 1 | 0.524 (0.002) 0.000 | 0.94 (0.74) 0.003 |
| main | ordinals | L16 | 0 | 0.090 (0.001) 0.120 | 0.94 (0.65) 0.003 |
| main | ordinals | L16 | 1 | 0.627 (0.002) 0.000 | 0.92 (0.68) 0.003 |
| main | solfege | emb_in | 0 | -0.008 (0.000) 0.907 | 0.87 (0.88) 0.601 |
| main | solfege | emb_in | 1 | 0.010 (0.000) 0.075 | 0.89 (0.87) 0.312 |
| main | solfege | emb_out | 0 | 0.002 (-0.000) 0.384 | 0.87 (0.91) 0.980 |
| main | solfege | emb_out | 1 | 0.017 (-0.000) 0.004 | 0.90 (0.92) 0.937 |
| main | solfege | L14 | 0 | 0.008 (0.001) 0.312 | 0.83 (0.79) 0.199 |
| main | solfege | L14 | 1 | 0.024 (0.000) 0.071 | 0.87 (0.82) 0.150 |
| stage1-step10000-tokens42B | planets | emb_in | 0 | 0.014 (-0.000) 0.132 | 0.99 (0.99) 0.515 |
| stage1-step10000-tokens42B | planets | emb_in | 1 | -0.007 (0.000) 0.635 | 0.99 (0.99) 0.233 |
| stage1-step10000-tokens42B | planets | emb_out | 0 | 0.002 (0.000) 0.401 | 0.96 (0.96) 0.256 |
| stage1-step10000-tokens42B | planets | emb_out | 1 | 0.001 (-0.000) 0.470 | 0.97 (0.96) 0.259 |
| stage1-step10000-tokens42B | planets | L2 | 0 | 0.014 (0.000) 0.147 | 0.92 (0.90) 0.213 |
| stage1-step10000-tokens42B | planets | L2 | 1 | -0.009 (0.000) 0.719 | 0.96 (0.94) 0.123 |
| stage1-step10000-tokens42B | elements | emb_in | 0 | -0.011 (0.000) 0.809 | 0.96 (0.96) 0.382 |
| stage1-step10000-tokens42B | elements | emb_in | 1 | 0.016 (0.000) 0.155 | 0.97 (0.96) 0.073 |
| stage1-step10000-tokens42B | elements | emb_out | 0 | 0.003 (-0.000) 0.212 | 0.94 (0.93) 0.229 |
| stage1-step10000-tokens42B | elements | emb_out | 1 | 0.006 (0.000) 0.131 | 0.94 (0.93) 0.070 |
| stage1-step10000-tokens42B | elements | L26 | 0 | 0.005 (-0.000) 0.267 | 0.76 (0.78) 0.701 |
| stage1-step10000-tokens42B | elements | L26 | 1 | 0.026 (-0.000) 0.001 | 0.80 (0.78) 0.296 |
| stage1-step10000-tokens42B | presidents | emb_in | 0 | 0.008 (-0.000) 0.234 | 0.95 (0.95) 0.399 |
| stage1-step10000-tokens42B | presidents | emb_in | 1 | 0.000 (-0.000) 0.487 | 0.95 (0.95) 0.126 |
| stage1-step10000-tokens42B | presidents | emb_out | 0 | 0.003 (0.000) 0.299 | 0.92 (0.91) 0.066 |
| stage1-step10000-tokens42B | presidents | emb_out | 1 | -0.002 (-0.000) 0.639 | 0.93 (0.91) 0.040 |
| stage1-step10000-tokens42B | presidents | L10 | 0 | -0.003 (-0.000) 0.672 | 0.83 (0.78) 0.013 |
| stage1-step10000-tokens42B | presidents | L10 | 1 | 0.010 (0.000) 0.112 | 0.86 (0.81) 0.023 |
| stage1-step10000-tokens42B | greek | emb_in | 0 | 0.020 (0.000) 0.032 | 0.97 (0.96) 0.056 |
| stage1-step10000-tokens42B | greek | emb_in | 1 | 0.008 (-0.001) 0.213 | 0.97 (0.96) 0.060 |
| stage1-step10000-tokens42B | greek | emb_out | 0 | 0.010 (0.000) 0.024 | 0.95 (0.92) 0.003 |
| stage1-step10000-tokens42B | greek | emb_out | 1 | 0.008 (0.000) 0.052 | 0.95 (0.91) 0.003 |
| stage1-step10000-tokens42B | greek | L16 | 0 | 0.025 (0.000) 0.000 | 0.84 (0.76) 0.037 |
| stage1-step10000-tokens42B | greek | L16 | 1 | 0.013 (-0.000) 0.066 | 0.88 (0.79) 0.003 |
| stage1-step10000-tokens42B | ordinals | emb_in | 0 | 0.027 (-0.000) 0.152 | 0.97 (0.96) 0.003 |
| stage1-step10000-tokens42B | ordinals | emb_in | 1 | 0.208 (-0.000) 0.000 | 0.98 (0.96) 0.003 |
| stage1-step10000-tokens42B | ordinals | emb_out | 0 | 0.077 (-0.000) 0.036 | 0.96 (0.81) 0.003 |
| stage1-step10000-tokens42B | ordinals | emb_out | 1 | 0.300 (0.000) 0.000 | 0.95 (0.82) 0.003 |
| stage1-step10000-tokens42B | ordinals | L4 | 0 | 0.104 (-0.002) 0.023 | 0.97 (0.71) 0.003 |
| stage1-step10000-tokens42B | ordinals | L4 | 1 | 0.415 (0.001) 0.000 | 0.97 (0.74) 0.003 |
| stage1-step10000-tokens42B | solfege | emb_in | 0 | -0.016 (0.000) 0.835 | 0.98 (0.98) 0.960 |
| stage1-step10000-tokens42B | solfege | emb_in | 1 | 0.012 (0.000) 0.251 | 0.97 (0.98) 1.000 |
| stage1-step10000-tokens42B | solfege | emb_out | 0 | -0.001 (0.000) 0.606 | 0.93 (0.95) 0.973 |
| stage1-step10000-tokens42B | solfege | emb_out | 1 | 0.003 (0.000) 0.313 | 0.95 (0.96) 0.867 |
| stage1-step10000-tokens42B | solfege | L18 | 0 | -0.018 (-0.000) 0.970 | 0.90 (0.85) 0.066 |
| stage1-step10000-tokens42B | solfege | L18 | 1 | 0.011 (0.000) 0.100 | 0.92 (0.87) 0.013 |
| stage1-step150000-tokens630B | planets | emb_in | 0 | 0.001 (0.000) 0.492 | 0.99 (0.97) 0.013 |
| stage1-step150000-tokens630B | planets | emb_in | 1 | 0.003 (0.000) 0.404 | 0.98 (0.97) 0.056 |
| stage1-step150000-tokens630B | planets | emb_out | 0 | 0.012 (0.000) 0.080 | 0.95 (0.93) 0.223 |
| stage1-step150000-tokens630B | planets | emb_out | 1 | 0.011 (-0.000) 0.070 | 0.95 (0.94) 0.246 |
| stage1-step150000-tokens630B | planets | L8 | 0 | 0.013 (0.000) 0.195 | 0.90 (0.86) 0.130 |
| stage1-step150000-tokens630B | planets | L8 | 1 | 0.032 (-0.000) 0.013 | 0.91 (0.87) 0.140 |
| stage1-step150000-tokens630B | elements | emb_in | 0 | 0.007 (0.000) 0.181 | 0.93 (0.91) 0.146 |
| stage1-step150000-tokens630B | elements | emb_in | 1 | 0.018 (0.000) 0.010 | 0.92 (0.89) 0.136 |
| stage1-step150000-tokens630B | elements | emb_out | 0 | 0.002 (0.000) 0.358 | 0.91 (0.91) 0.429 |
| stage1-step150000-tokens630B | elements | emb_out | 1 | 0.010 (-0.000) 0.007 | 0.92 (0.90) 0.047 |
| stage1-step150000-tokens630B | elements | L18 | 0 | -0.004 (-0.000) 0.641 | 0.79 (0.80) 0.575 |
| stage1-step150000-tokens630B | elements | L18 | 1 | 0.020 (-0.000) 0.026 | 0.85 (0.82) 0.163 |
| stage1-step150000-tokens630B | presidents | emb_in | 0 | 0.008 (-0.000) 0.090 | 0.88 (0.86) 0.066 |
| stage1-step150000-tokens630B | presidents | emb_in | 1 | 0.000 (-0.000) 0.454 | 0.89 (0.87) 0.047 |
| stage1-step150000-tokens630B | presidents | emb_out | 0 | -0.000 (0.000) 0.548 | 0.87 (0.86) 0.140 |
| stage1-step150000-tokens630B | presidents | emb_out | 1 | 0.003 (-0.000) 0.270 | 0.88 (0.86) 0.050 |
| stage1-step150000-tokens630B | presidents | L28 | 0 | 0.005 (0.000) 0.175 | 0.81 (0.69) 0.003 |
| stage1-step150000-tokens630B | presidents | L28 | 1 | -0.006 (0.000) 0.854 | 0.78 (0.71) 0.010 |
| stage1-step150000-tokens630B | greek | emb_in | 0 | 0.002 (-0.000) 0.397 | 0.94 (0.91) 0.013 |
| stage1-step150000-tokens630B | greek | emb_in | 1 | 0.018 (-0.000) 0.002 | 0.93 (0.89) 0.007 |
| stage1-step150000-tokens630B | greek | emb_out | 0 | 0.014 (0.000) 0.002 | 0.92 (0.88) 0.003 |
| stage1-step150000-tokens630B | greek | emb_out | 1 | 0.009 (0.000) 0.041 | 0.93 (0.87) 0.003 |
| stage1-step150000-tokens630B | greek | L10 | 0 | 0.008 (0.000) 0.192 | 0.77 (0.76) 0.332 |
| stage1-step150000-tokens630B | greek | L10 | 1 | 0.028 (0.000) 0.000 | 0.84 (0.78) 0.053 |
| stage1-step150000-tokens630B | ordinals | emb_in | 0 | 0.062 (-0.000) 0.078 | 0.97 (0.86) 0.003 |
| stage1-step150000-tokens630B | ordinals | emb_in | 1 | 0.333 (0.001) 0.000 | 0.98 (0.85) 0.003 |
| stage1-step150000-tokens630B | ordinals | emb_out | 0 | 0.117 (0.001) 0.044 | 0.97 (0.77) 0.003 |
| stage1-step150000-tokens630B | ordinals | emb_out | 1 | 0.480 (0.001) 0.000 | 0.95 (0.78) 0.003 |
| stage1-step150000-tokens630B | ordinals | L16 | 0 | 0.124 (0.001) 0.052 | 0.93 (0.62) 0.003 |
| stage1-step150000-tokens630B | ordinals | L16 | 1 | 0.579 (0.003) 0.000 | 0.93 (0.65) 0.003 |
| stage1-step150000-tokens630B | solfege | emb_in | 0 | -0.023 (0.000) 0.999 | 0.87 (0.89) 0.787 |
| stage1-step150000-tokens630B | solfege | emb_in | 1 | 0.010 (0.000) 0.107 | 0.90 (0.89) 0.385 |
| stage1-step150000-tokens630B | solfege | emb_out | 0 | 0.000 (0.000) 0.478 | 0.85 (0.90) 0.987 |
| stage1-step150000-tokens630B | solfege | emb_out | 1 | 0.010 (-0.000) 0.033 | 0.88 (0.91) 0.950 |
| stage1-step150000-tokens630B | solfege | L12 | 0 | 0.005 (-0.001) 0.379 | 0.89 (0.82) 0.120 |
| stage1-step150000-tokens630B | solfege | L12 | 1 | 0.032 (0.000) 0.022 | 0.88 (0.84) 0.189 |
| stage1-step2000-tokens9B | planets | emb_in | 0 | 0.011 (0.000) 0.111 | 1.00 (1.00) 0.153 |
| stage1-step2000-tokens9B | planets | emb_in | 1 | -0.016 (0.000) 0.963 | 1.00 (1.00) 0.551 |
| stage1-step2000-tokens9B | planets | emb_out | 0 | -0.001 (0.000) 0.575 | 0.99 (0.99) 0.369 |
| stage1-step2000-tokens9B | planets | emb_out | 1 | 0.002 (-0.000) 0.417 | 1.00 (0.99) 0.209 |
| stage1-step2000-tokens9B | planets | L8 | 0 | 0.011 (-0.000) 0.266 | 0.88 (0.91) 0.977 |
| stage1-step2000-tokens9B | planets | L8 | 1 | -0.014 (-0.000) 0.786 | 0.92 (0.94) 0.814 |
| stage1-step2000-tokens9B | elements | emb_in | 0 | 0.004 (0.000) 0.305 | 0.93 (0.93) 0.455 |
| stage1-step2000-tokens9B | elements | emb_in | 1 | 0.004 (0.000) 0.315 | 0.93 (0.93) 0.917 |
| stage1-step2000-tokens9B | elements | emb_out | 0 | 0.002 (-0.000) 0.303 | 0.94 (0.94) 0.545 |
| stage1-step2000-tokens9B | elements | emb_out | 1 | 0.009 (0.000) 0.039 | 0.93 (0.94) 0.645 |
| stage1-step2000-tokens9B | elements | L2 | 0 | -0.012 (0.000) 0.754 | 0.90 (0.90) 0.339 |
| stage1-step2000-tokens9B | elements | L2 | 1 | 0.022 (0.000) 0.105 | 0.92 (0.91) 0.226 |
| stage1-step2000-tokens9B | presidents | emb_in | 0 | 0.001 (-0.000) 0.432 | 0.94 (0.94) 0.322 |
| stage1-step2000-tokens9B | presidents | emb_in | 1 | 0.000 (-0.000) 0.463 | 0.94 (0.94) 0.030 |
| stage1-step2000-tokens9B | presidents | emb_out | 0 | 0.009 (0.000) 0.041 | 0.95 (0.94) 0.316 |
| stage1-step2000-tokens9B | presidents | emb_out | 1 | -0.001 (0.000) 0.561 | 0.95 (0.95) 0.316 |
| stage1-step2000-tokens9B | presidents | L30 | 0 | 0.012 (-0.000) 0.122 | 0.84 (0.78) 0.020 |
| stage1-step2000-tokens9B | presidents | L30 | 1 | 0.020 (-0.000) 0.019 | 0.83 (0.79) 0.066 |
| stage1-step2000-tokens9B | greek | emb_in | 0 | 0.004 (0.000) 0.241 | 0.94 (0.94) 0.206 |
| stage1-step2000-tokens9B | greek | emb_in | 1 | -0.004 (-0.000) 0.728 | 0.94 (0.94) 0.405 |
| stage1-step2000-tokens9B | greek | emb_out | 0 | 0.002 (-0.000) 0.388 | 0.95 (0.94) 0.037 |
| stage1-step2000-tokens9B | greek | emb_out | 1 | 0.002 (0.000) 0.378 | 0.95 (0.94) 0.033 |
| stage1-step2000-tokens9B | greek | L30 | 0 | 0.000 (0.000) 0.481 | 0.82 (0.80) 0.219 |
| stage1-step2000-tokens9B | greek | L30 | 1 | 0.006 (-0.000) 0.315 | 0.86 (0.83) 0.159 |
| stage1-step2000-tokens9B | ordinals | emb_in | 0 | 0.002 (0.000) 0.380 | 0.99 (0.99) 0.229 |
| stage1-step2000-tokens9B | ordinals | emb_in | 1 | 0.006 (-0.000) 0.205 | 0.99 (0.99) 0.329 |
| stage1-step2000-tokens9B | ordinals | emb_out | 0 | 0.011 (-0.000) 0.042 | 0.99 (0.97) 0.003 |
| stage1-step2000-tokens9B | ordinals | emb_out | 1 | 0.022 (0.000) 0.001 | 0.99 (0.97) 0.003 |
| stage1-step2000-tokens9B | ordinals | L8 | 0 | 0.050 (0.000) 0.002 | 0.96 (0.79) 0.003 |
| stage1-step2000-tokens9B | ordinals | L8 | 1 | 0.054 (-0.000) 0.000 | 0.95 (0.81) 0.003 |
| stage1-step2000-tokens9B | solfege | emb_in | 0 | -0.009 (-0.000) 0.811 | 1.00 (1.00) 0.877 |
| stage1-step2000-tokens9B | solfege | emb_in | 1 | 0.001 (0.000) 0.486 | 1.00 (1.00) 0.751 |
| stage1-step2000-tokens9B | solfege | emb_out | 0 | -0.001 (0.000) 0.560 | 0.99 (0.99) 0.877 |
| stage1-step2000-tokens9B | solfege | emb_out | 1 | -0.003 (0.000) 0.648 | 0.99 (0.99) 0.940 |
| stage1-step2000-tokens9B | solfege | L16 | 0 | -0.007 (0.000) 0.683 | 0.97 (0.92) 0.037 |
| stage1-step2000-tokens9B | solfege | L16 | 1 | -0.019 (0.000) 0.946 | 0.97 (0.94) 0.043 |
| stage1-step400000-tokens1678B | planets | emb_in | 0 | -0.001 (0.000) 0.519 | 0.99 (0.96) 0.017 |
| stage1-step400000-tokens1678B | planets | emb_in | 1 | 0.011 (0.000) 0.190 | 0.98 (0.96) 0.093 |
| stage1-step400000-tokens1678B | planets | emb_out | 0 | 0.012 (0.000) 0.077 | 0.96 (0.93) 0.203 |
| stage1-step400000-tokens1678B | planets | emb_out | 1 | 0.008 (0.000) 0.144 | 0.96 (0.94) 0.169 |
| stage1-step400000-tokens1678B | planets | L16 | 0 | 0.033 (-0.000) 0.096 | 0.93 (0.91) 0.213 |
| stage1-step400000-tokens1678B | planets | L16 | 1 | 0.009 (0.001) 0.341 | 0.97 (0.94) 0.093 |
| stage1-step400000-tokens1678B | elements | emb_in | 0 | 0.002 (0.000) 0.408 | 0.93 (0.90) 0.136 |
| stage1-step400000-tokens1678B | elements | emb_in | 1 | 0.016 (0.000) 0.018 | 0.92 (0.88) 0.123 |
| stage1-step400000-tokens1678B | elements | emb_out | 0 | 0.001 (0.000) 0.430 | 0.91 (0.91) 0.409 |
| stage1-step400000-tokens1678B | elements | emb_out | 1 | 0.009 (-0.000) 0.020 | 0.92 (0.89) 0.050 |
| stage1-step400000-tokens1678B | elements | L16 | 0 | -0.003 (0.000) 0.582 | 0.86 (0.86) 0.585 |
| stage1-step400000-tokens1678B | elements | L16 | 1 | 0.028 (-0.000) 0.025 | 0.88 (0.87) 0.309 |
| stage1-step400000-tokens1678B | presidents | emb_in | 0 | 0.005 (-0.000) 0.188 | 0.88 (0.85) 0.140 |
| stage1-step400000-tokens1678B | presidents | emb_in | 1 | 0.006 (-0.000) 0.149 | 0.89 (0.86) 0.093 |
| stage1-step400000-tokens1678B | presidents | emb_out | 0 | -0.001 (0.000) 0.615 | 0.87 (0.86) 0.193 |
| stage1-step400000-tokens1678B | presidents | emb_out | 1 | -0.000 (0.000) 0.505 | 0.88 (0.87) 0.037 |
| stage1-step400000-tokens1678B | presidents | L20 | 0 | 0.011 (-0.000) 0.115 | 0.84 (0.80) 0.047 |
| stage1-step400000-tokens1678B | presidents | L20 | 1 | 0.004 (-0.000) 0.307 | 0.85 (0.83) 0.113 |
| stage1-step400000-tokens1678B | greek | emb_in | 0 | -0.002 (0.000) 0.682 | 0.93 (0.89) 0.030 |
| stage1-step400000-tokens1678B | greek | emb_in | 1 | 0.013 (-0.000) 0.004 | 0.93 (0.88) 0.017 |
| stage1-step400000-tokens1678B | greek | emb_out | 0 | 0.012 (0.000) 0.007 | 0.92 (0.88) 0.007 |
| stage1-step400000-tokens1678B | greek | emb_out | 1 | 0.010 (0.000) 0.015 | 0.92 (0.87) 0.003 |
| stage1-step400000-tokens1678B | greek | L12 | 0 | 0.010 (0.000) 0.171 | 0.78 (0.79) 0.631 |
| stage1-step400000-tokens1678B | greek | L12 | 1 | 0.037 (-0.000) 0.001 | 0.83 (0.81) 0.229 |
| stage1-step400000-tokens1678B | ordinals | emb_in | 0 | 0.067 (-0.000) 0.072 | 0.96 (0.84) 0.003 |
| stage1-step400000-tokens1678B | ordinals | emb_in | 1 | 0.339 (0.001) 0.000 | 0.97 (0.83) 0.003 |
| stage1-step400000-tokens1678B | ordinals | emb_out | 0 | 0.115 (0.001) 0.049 | 0.97 (0.77) 0.003 |
| stage1-step400000-tokens1678B | ordinals | emb_out | 1 | 0.496 (0.001) 0.000 | 0.95 (0.77) 0.003 |
| stage1-step400000-tokens1678B | ordinals | L16 | 0 | 0.106 (0.000) 0.106 | 0.94 (0.70) 0.003 |
| stage1-step400000-tokens1678B | ordinals | L16 | 1 | 0.698 (0.001) 0.000 | 0.92 (0.73) 0.003 |
| stage1-step400000-tokens1678B | solfege | emb_in | 0 | -0.014 (0.000) 0.991 | 0.87 (0.88) 0.645 |
| stage1-step400000-tokens1678B | solfege | emb_in | 1 | 0.004 (0.000) 0.287 | 0.89 (0.87) 0.319 |
| stage1-step400000-tokens1678B | solfege | emb_out | 0 | -0.004 (0.000) 0.770 | 0.86 (0.91) 0.980 |
| stage1-step400000-tokens1678B | solfege | emb_out | 1 | 0.011 (-0.000) 0.028 | 0.89 (0.91) 0.953 |
| stage1-step400000-tokens1678B | solfege | L14 | 0 | -0.000 (0.001) 0.521 | 0.88 (0.78) 0.013 |
| stage1-step400000-tokens1678B | solfege | L14 | 1 | 0.037 (0.001) 0.012 | 0.90 (0.82) 0.013 |
| stage1-step50000-tokens210B | planets | emb_in | 0 | 0.001 (0.000) 0.483 | 0.99 (0.98) 0.126 |
| stage1-step50000-tokens210B | planets | emb_in | 1 | -0.009 (0.000) 0.749 | 0.99 (0.99) 0.023 |
| stage1-step50000-tokens210B | planets | emb_out | 0 | 0.008 (0.000) 0.201 | 0.94 (0.92) 0.229 |
| stage1-step50000-tokens210B | planets | emb_out | 1 | 0.011 (-0.000) 0.135 | 0.94 (0.94) 0.385 |
| stage1-step50000-tokens210B | planets | L12 | 0 | 0.020 (-0.000) 0.107 | 0.91 (0.84) 0.120 |
| stage1-step50000-tokens210B | planets | L12 | 1 | 0.006 (-0.001) 0.331 | 0.93 (0.89) 0.106 |
| stage1-step50000-tokens210B | elements | emb_in | 0 | -0.006 (0.000) 0.715 | 0.97 (0.96) 0.063 |
| stage1-step50000-tokens210B | elements | emb_in | 1 | 0.023 (0.000) 0.018 | 0.97 (0.95) 0.013 |
| stage1-step50000-tokens210B | elements | emb_out | 0 | 0.001 (0.000) 0.433 | 0.91 (0.91) 0.425 |
| stage1-step50000-tokens210B | elements | emb_out | 1 | 0.008 (-0.000) 0.042 | 0.92 (0.90) 0.056 |
| stage1-step50000-tokens210B | elements | L2 | 0 | 0.001 (0.000) 0.500 | 0.90 (0.88) 0.199 |
| stage1-step50000-tokens210B | elements | L2 | 1 | 0.024 (0.000) 0.005 | 0.91 (0.88) 0.056 |
| stage1-step50000-tokens210B | presidents | emb_in | 0 | 0.008 (-0.000) 0.206 | 0.95 (0.93) 0.003 |
| stage1-step50000-tokens210B | presidents | emb_in | 1 | 0.006 (-0.000) 0.285 | 0.95 (0.93) 0.013 |
| stage1-step50000-tokens210B | presidents | emb_out | 0 | -0.003 (0.000) 0.728 | 0.88 (0.87) 0.080 |
| stage1-step50000-tokens210B | presidents | emb_out | 1 | -0.001 (-0.000) 0.534 | 0.88 (0.87) 0.066 |
| stage1-step50000-tokens210B | presidents | L12 | 0 | 0.006 (-0.000) 0.216 | 0.82 (0.78) 0.020 |
| stage1-step50000-tokens210B | presidents | L12 | 1 | -0.004 (-0.000) 0.696 | 0.84 (0.80) 0.037 |
| stage1-step50000-tokens210B | greek | emb_in | 0 | 0.011 (0.000) 0.135 | 0.97 (0.95) 0.003 |
| stage1-step50000-tokens210B | greek | emb_in | 1 | 0.020 (-0.001) 0.011 | 0.96 (0.94) 0.013 |
| stage1-step50000-tokens210B | greek | emb_out | 0 | 0.010 (0.000) 0.023 | 0.93 (0.88) 0.003 |
| stage1-step50000-tokens210B | greek | emb_out | 1 | 0.001 (0.000) 0.452 | 0.93 (0.87) 0.003 |
| stage1-step50000-tokens210B | greek | L4 | 0 | 0.002 (0.000) 0.409 | 0.89 (0.82) 0.003 |
| stage1-step50000-tokens210B | greek | L4 | 1 | 0.013 (0.000) 0.050 | 0.92 (0.84) 0.003 |
| stage1-step50000-tokens210B | ordinals | emb_in | 0 | 0.037 (-0.001) 0.151 | 0.98 (0.93) 0.003 |
| stage1-step50000-tokens210B | ordinals | emb_in | 1 | 0.303 (0.000) 0.000 | 0.99 (0.93) 0.003 |
| stage1-step50000-tokens210B | ordinals | emb_out | 0 | 0.121 (0.001) 0.037 | 0.97 (0.78) 0.003 |
| stage1-step50000-tokens210B | ordinals | emb_out | 1 | 0.468 (0.001) 0.000 | 0.95 (0.79) 0.003 |
| stage1-step50000-tokens210B | ordinals | L16 | 0 | 0.134 (0.001) 0.039 | 0.94 (0.65) 0.003 |
| stage1-step50000-tokens210B | ordinals | L16 | 1 | 0.552 (0.002) 0.000 | 0.93 (0.68) 0.003 |
| stage1-step50000-tokens210B | solfege | emb_in | 0 | -0.026 (0.000) 0.991 | 0.96 (0.97) 0.970 |
| stage1-step50000-tokens210B | solfege | emb_in | 1 | 0.008 (0.000) 0.270 | 0.96 (0.97) 0.970 |
| stage1-step50000-tokens210B | solfege | emb_out | 0 | -0.003 (0.000) 0.741 | 0.87 (0.92) 0.987 |
| stage1-step50000-tokens210B | solfege | emb_out | 1 | 0.007 (-0.000) 0.059 | 0.91 (0.93) 0.924 |
| stage1-step50000-tokens210B | solfege | L10 | 0 | -0.015 (-0.000) 0.855 | 0.85 (0.81) 0.113 |
| stage1-step50000-tokens210B | solfege | L10 | 1 | 0.019 (0.000) 0.057 | 0.88 (0.86) 0.133 |
