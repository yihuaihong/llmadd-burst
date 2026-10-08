# P2: number route

## sum route (layer with the highest consistency; null = k permuted within item)

| ckpt | domain | model acc | layer | probe acc (numeric) | slope | r2 (null p95) | consistency (null p95) | line acc (null p95) | mod acc (null p95) | offsets |
|---|---|---|---|---|---|---|---|---|---|---|
| stage1-step10000-tokens42B | days | 0.00 | L10 | 0.13 | 0.47 | 0.66 (0.06) | 0.10 (0.08) | 0.03 (0.04) | 0.16 (0.18) | [20, 15, 25, 30, 21, 22, 20] |
| stage1-step10000-tokens42B | months | 0.00 | L18 | 0.14 | 0.44 | 0.34 (0.04) | 0.10 (0.08) | 0.03 (0.03) | 0.10 (0.11) | [24, 24, 14, 21, 49, 23, 19, 29, 19, 23, 11, 29] |
| stage1-step10000-tokens42B | letters | 0.00 | L2 | 0.07 | 1.20 | 0.69 (0.13) | 0.20 (0.16) | 0.05 (0.06) | - | [18, 16, 29, 30, 28, 24, 18, 30, 11, 30, 28, 30] |
| stage1-step10000-tokens42B | numwords | - | L20 | 0.21 | 0.88 | 0.62 (0.08) | 0.17 (0.13) | 0.09 (0.05) | - | [-1, 0, 2, 4, -1, 4, 7, 8, 16, 10, 12, 10] |
| stage1-step2000-tokens9B | days | 0.00 | L8 | 0.06 | 0.84 | 0.75 (0.05) | 0.10 (0.07) | 0.00 (0.02) | 0.16 (0.19) | [33, 47, 49, 52, 28, 30, 33] |
| stage1-step2000-tokens9B | months | 0.00 | L26 | 0.07 | 0.45 | 0.36 (0.04) | 0.09 (0.08) | 0.04 (0.04) | 0.11 (0.11) | [18, 31, 39, 21, 19, 22, 29, 19, 28, 35, 41, 15] |
| stage1-step2000-tokens9B | letters | 0.00 | L4 | 0.06 | 1.16 | 0.64 (0.13) | 0.20 (0.16) | 0.05 (0.06) | - | [17, 9, 4, 23, 17, 14, 14, 14, 11, 14, 25, 19] |
| stage1-step2000-tokens9B | numwords | - | L2 | 0.04 | 0.90 | 0.75 (0.08) | 0.19 (0.13) | 0.07 (0.06) | - | [23, 28, 21, 30, 25, 30, 21, 30, 24, 20, 33, 29] |
| stage1-step5000-tokens21B | days | 0.00 | L18 | 0.10 | 1.17 | 0.71 (0.05) | 0.10 (0.08) | 0.04 (0.04) | 0.15 (0.19) | [13, 27, 15, 6, 11, -1, -5] |
| stage1-step5000-tokens21B | months | 0.00 | L16 | 0.11 | 0.47 | 0.61 (0.05) | 0.08 (0.07) | 0.02 (0.03) | 0.09 (0.11) | [23, 22, 25, 25, 29, 23, 23, 30, 20, 23, 23, 30] |
| stage1-step5000-tokens21B | letters | 0.00 | L6 | 0.14 | 0.68 | 0.37 (0.13) | 0.20 (0.15) | 0.07 (0.07) | - | [2, 6, 6, 6, 11, 6, 15, 15, 1, 6, 16, 11] |
| stage1-step5000-tokens21B | numwords | - | L16 | 0.19 | 1.00 | 0.73 (0.08) | 0.16 (0.12) | 0.08 (0.06) | - | [-7, -1, 5, 14, 19, 8, 15, 18, 20, 20, 11, 12] |

## operand route ("Q: item" decoded on the number line; layer with the highest |spearman|)

| ckpt | domain | layer | probe acc (numeric) | spearman | decoded |
|---|---|---|---|---|---|
| stage1-step10000-tokens42B | days | L2 | 0.96 | -0.57 | [25, 30, 29, 21, 21, 28, 18] |
| stage1-step10000-tokens42B | months | L22 | 0.96 | 0.36 | [103, 103, 104, 99, 109, 104, 103, 98, 99, 109, 109, 104] |
| stage1-step10000-tokens42B | letters | L24 | 0.95 | 0.75 | [98, 98, 94, 98, 98, 98, 100, 103, 98, 100, 104, 98] |
| stage1-step10000-tokens42B | numwords | L20 | 0.95 | 0.64 | [92, 102, 93, 104, 95, 96, 97, 98, 99, 10, 101, 12] |
| stage1-step2000-tokens9B | days | L28 | 0.27 | 0.89 | [17, 17, 17, 17, 27, 27, 17] |
| stage1-step2000-tokens9B | months | L24 | 0.29 | 0.74 | [15, 19, 99, 99, 21, 81, 89, 97, 89, 101, 109, 109] |
| stage1-step2000-tokens9B | letters | L4 | 0.28 | -0.30 | [9, 98, 98, 98, 92, 88, 110, 101, 102, 101, 118, 98] |
| stage1-step2000-tokens9B | numwords | L24 | 0.29 | 0.32 | [13, 4, 94, 4, 4, 94, 94, 94, 94, 14, 100, 24] |
| stage1-step5000-tokens21B | days | L10 | 0.88 | 0.89 | [14, 24, 18, 22, 24, 24, 108] |
| stage1-step5000-tokens21B | months | L26 | 0.84 | 0.66 | [110, 20, 20, 109, 10, 16, 111, 20, 111, 111, 110, 116] |
| stage1-step5000-tokens21B | letters | L8 | 0.85 | -0.41 | [100, 106, 8, 110, 100, 100, 110, 101, 110, 110, 20, 100] |
| stage1-step5000-tokens21B | numwords | L22 | 0.87 | 0.75 | [8, 102, 103, 104, 104, 106, 107, 98, 109, 10, 109, 108] |

## alignment: item i vs number i + o (centred cosine, p vs 2000 permutations; CKA, p vs 300)

| ckpt | domain | site | o | cos (null) p | CKA (null) p |
|---|---|---|---|---|---|
| stage1-step10000-tokens42B | days | emb_in | 0 | 0.009 (0.000) 0.287 | 0.97 (0.97) 0.502 |
| stage1-step10000-tokens42B | days | emb_in | 1 | 0.008 (-0.000) 0.343 | 0.98 (0.97) 0.150 |
| stage1-step10000-tokens42B | days | emb_out | 0 | -0.010 (-0.000) 0.857 | 0.93 (0.86) 0.020 |
| stage1-step10000-tokens42B | days | emb_out | 1 | 0.015 (-0.000) 0.077 | 0.95 (0.87) 0.003 |
| stage1-step10000-tokens42B | days | L6 | 0 | 0.012 (-0.000) 0.286 | 0.82 (0.75) 0.053 |
| stage1-step10000-tokens42B | days | L6 | 1 | 0.048 (0.000) 0.025 | 0.87 (0.77) 0.020 |
| stage1-step10000-tokens42B | months | emb_in | 0 | -0.026 (0.000) 0.944 | 0.97 (0.96) 0.037 |
| stage1-step10000-tokens42B | months | emb_in | 1 | 0.096 (0.000) 0.000 | 0.97 (0.96) 0.007 |
| stage1-step10000-tokens42B | months | emb_out | 0 | -0.003 (-0.000) 0.513 | 0.90 (0.83) 0.003 |
| stage1-step10000-tokens42B | months | emb_out | 1 | 0.090 (-0.000) 0.000 | 0.91 (0.83) 0.003 |
| stage1-step10000-tokens42B | months | L4 | 0 | -0.011 (0.001) 0.676 | 0.87 (0.79) 0.013 |
| stage1-step10000-tokens42B | months | L4 | 1 | 0.169 (0.000) 0.000 | 0.91 (0.81) 0.003 |
| stage1-step10000-tokens42B | letters | emb_in | 0 | 0.010 (-0.000) 0.129 | 0.91 (0.91) 0.445 |
| stage1-step10000-tokens42B | letters | emb_in | 1 | -0.004 (-0.000) 0.675 | 0.92 (0.91) 0.159 |
| stage1-step10000-tokens42B | letters | emb_out | 0 | 0.002 (-0.000) 0.135 | 0.87 (0.85) 0.003 |
| stage1-step10000-tokens42B | letters | emb_out | 1 | 0.004 (0.000) 0.022 | 0.87 (0.85) 0.003 |
| stage1-step10000-tokens42B | letters | L28 | 0 | -0.001 (0.000) 0.546 | 0.65 (0.62) 0.083 |
| stage1-step10000-tokens42B | letters | L28 | 1 | 0.013 (0.000) 0.061 | 0.65 (0.61) 0.030 |
| stage1-step10000-tokens42B | numwords | emb_in | 0 | 0.090 (0.000) 0.001 | 0.95 (0.89) 0.003 |
| stage1-step10000-tokens42B | numwords | emb_in | 1 | 0.341 (-0.000) 0.000 | 0.96 (0.89) 0.003 |
| stage1-step10000-tokens42B | numwords | emb_out | 0 | 0.121 (-0.000) 0.000 | 0.93 (0.65) 0.003 |
| stage1-step10000-tokens42B | numwords | emb_out | 1 | 0.411 (-0.001) 0.000 | 0.94 (0.65) 0.003 |
| stage1-step10000-tokens42B | numwords | L4 | 0 | 0.203 (-0.000) 0.000 | 0.94 (0.59) 0.003 |
| stage1-step10000-tokens42B | numwords | L4 | 1 | 0.621 (-0.001) 0.000 | 0.97 (0.61) 0.003 |
| stage1-step2000-tokens9B | days | emb_in | 0 | 0.021 (0.000) 0.013 | 1.00 (1.00) 0.455 |
| stage1-step2000-tokens9B | days | emb_in | 1 | -0.009 (-0.000) 0.836 | 1.00 (1.00) 0.711 |
| stage1-step2000-tokens9B | days | emb_out | 0 | 0.004 (0.000) 0.288 | 0.99 (0.98) 0.047 |
| stage1-step2000-tokens9B | days | emb_out | 1 | -0.018 (0.000) 0.998 | 0.99 (0.98) 0.013 |
| stage1-step2000-tokens9B | days | L4 | 0 | 0.018 (-0.001) 0.103 | 0.95 (0.93) 0.150 |
| stage1-step2000-tokens9B | days | L4 | 1 | -0.015 (-0.000) 0.828 | 0.98 (0.96) 0.003 |
| stage1-step2000-tokens9B | months | emb_in | 0 | -0.014 (0.001) 0.919 | 0.94 (0.94) 0.983 |
| stage1-step2000-tokens9B | months | emb_in | 1 | 0.006 (0.000) 0.124 | 0.94 (0.94) 0.551 |
| stage1-step2000-tokens9B | months | emb_out | 0 | -0.004 (-0.000) 0.357 | 0.95 (0.94) 0.080 |
| stage1-step2000-tokens9B | months | emb_out | 1 | 0.001 (-0.000) 0.399 | 0.94 (0.93) 0.123 |
| stage1-step2000-tokens9B | months | L20 | 0 | -0.010 (-0.000) 0.848 | 0.83 (0.82) 0.339 |
| stage1-step2000-tokens9B | months | L20 | 1 | 0.013 (-0.000) 0.053 | 0.87 (0.84) 0.126 |
| stage1-step2000-tokens9B | letters | emb_in | 0 | 0.003 (-0.000) 0.269 | 0.93 (0.93) 0.738 |
| stage1-step2000-tokens9B | letters | emb_in | 1 | -0.004 (-0.000) 0.749 | 0.93 (0.93) 0.807 |
| stage1-step2000-tokens9B | letters | emb_out | 0 | -0.000 (0.000) 0.565 | 0.93 (0.92) 0.063 |
| stage1-step2000-tokens9B | letters | emb_out | 1 | -0.002 (0.000) 0.775 | 0.93 (0.92) 0.053 |
| stage1-step2000-tokens9B | letters | L4 | 0 | 0.005 (-0.000) 0.297 | 0.77 (0.75) 0.030 |
| stage1-step2000-tokens9B | letters | L4 | 1 | 0.004 (0.000) 0.358 | 0.80 (0.76) 0.007 |
| stage1-step2000-tokens9B | numwords | emb_in | 0 | 0.044 (0.000) 0.045 | 0.92 (0.92) 0.123 |
| stage1-step2000-tokens9B | numwords | emb_in | 1 | 0.054 (-0.000) 0.038 | 0.91 (0.92) 0.950 |
| stage1-step2000-tokens9B | numwords | emb_out | 0 | 0.054 (0.000) 0.047 | 0.92 (0.87) 0.007 |
| stage1-step2000-tokens9B | numwords | emb_out | 1 | 0.076 (-0.001) 0.027 | 0.92 (0.88) 0.007 |
| stage1-step2000-tokens9B | numwords | L8 | 0 | 0.099 (-0.000) 0.000 | 0.92 (0.66) 0.003 |
| stage1-step2000-tokens9B | numwords | L8 | 1 | 0.206 (0.000) 0.000 | 0.91 (0.67) 0.003 |
| stage1-step5000-tokens21B | days | emb_in | 0 | 0.029 (0.000) 0.005 | 0.99 (0.99) 0.741 |
| stage1-step5000-tokens21B | days | emb_in | 1 | -0.002 (-0.000) 0.560 | 0.99 (0.99) 0.249 |
| stage1-step5000-tokens21B | days | emb_out | 0 | -0.002 (0.000) 0.593 | 0.95 (0.90) 0.020 |
| stage1-step5000-tokens21B | days | emb_out | 1 | -0.007 (-0.000) 0.833 | 0.96 (0.90) 0.007 |
| stage1-step5000-tokens21B | days | L8 | 0 | 0.011 (0.000) 0.320 | 0.89 (0.80) 0.003 |
| stage1-step5000-tokens21B | days | L8 | 1 | 0.033 (0.000) 0.072 | 0.92 (0.82) 0.007 |
| stage1-step5000-tokens21B | months | emb_in | 0 | -0.019 (0.001) 0.925 | 0.96 (0.96) 0.342 |
| stage1-step5000-tokens21B | months | emb_in | 1 | 0.031 (0.000) 0.002 | 0.97 (0.96) 0.043 |
| stage1-step5000-tokens21B | months | emb_out | 0 | -0.001 (-0.000) 0.394 | 0.92 (0.87) 0.003 |
| stage1-step5000-tokens21B | months | emb_out | 1 | 0.041 (-0.000) 0.000 | 0.92 (0.87) 0.003 |
| stage1-step5000-tokens21B | months | L6 | 0 | -0.012 (-0.000) 0.753 | 0.85 (0.78) 0.063 |
| stage1-step5000-tokens21B | months | L6 | 1 | 0.084 (0.000) 0.000 | 0.90 (0.80) 0.003 |
| stage1-step5000-tokens21B | letters | emb_in | 0 | 0.008 (-0.000) 0.162 | 0.93 (0.93) 0.535 |
| stage1-step5000-tokens21B | letters | emb_in | 1 | -0.004 (-0.000) 0.682 | 0.93 (0.93) 0.548 |
| stage1-step5000-tokens21B | letters | emb_out | 0 | 0.003 (-0.000) 0.109 | 0.89 (0.88) 0.003 |
| stage1-step5000-tokens21B | letters | emb_out | 1 | 0.002 (0.000) 0.183 | 0.89 (0.88) 0.010 |
| stage1-step5000-tokens21B | letters | L30 | 0 | -0.000 (0.000) 0.524 | 0.66 (0.61) 0.066 |
| stage1-step5000-tokens21B | letters | L30 | 1 | 0.009 (0.000) 0.138 | 0.65 (0.60) 0.037 |
| stage1-step5000-tokens21B | numwords | emb_in | 0 | 0.076 (0.000) 0.009 | 0.95 (0.92) 0.007 |
| stage1-step5000-tokens21B | numwords | emb_in | 1 | 0.204 (-0.000) 0.000 | 0.95 (0.92) 0.007 |
| stage1-step5000-tokens21B | numwords | emb_out | 0 | 0.096 (-0.000) 0.001 | 0.93 (0.73) 0.003 |
| stage1-step5000-tokens21B | numwords | emb_out | 1 | 0.267 (-0.001) 0.000 | 0.94 (0.73) 0.003 |
| stage1-step5000-tokens21B | numwords | L4 | 0 | 0.180 (-0.000) 0.000 | 0.94 (0.59) 0.003 |
| stage1-step5000-tokens21B | numwords | L4 | 1 | 0.532 (-0.001) 0.000 | 0.97 (0.60) 0.003 |
