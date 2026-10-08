# P2: number route

## sum route (layer with the highest consistency; null = k permuted within item)

| ckpt | domain | model acc | layer | probe acc (numeric) | slope | r2 (null p95) | consistency (null p95) | line acc (null p95) | mod acc (null p95) | offsets |
|---|---|---|---|---|---|---|---|---|---|---|
| stage1-step10000-tokens21B | days | 0.00 | L10 | 0.13 | 0.62 | 0.75 (0.05) | 0.09 (0.07) | 0.02 (0.04) | 0.16 (0.19) | [24, 19, 27, 22, 16, 21, 11] |
| stage1-step10000-tokens21B | months | 0.00 | L8 | 0.11 | 1.08 | 0.62 (0.05) | 0.08 (0.06) | 0.03 (0.03) | 0.11 (0.11) | [27, 29, 47, 9, 18, 33, 30, -11, 20, 17, 27, 17] |
| stage1-step10000-tokens21B | letters | 0.00 | L2 | 0.04 | 1.47 | 0.73 (0.13) | 0.20 (0.16) | 0.06 (0.06) | - | [11, 12, 12, 22, 22, 11, 13, 12, 11, 22, 32, 21] |
| stage1-step10000-tokens21B | numwords | - | L6 | 0.14 | 0.65 | 0.62 (0.08) | 0.16 (0.12) | 0.09 (0.06) | - | [51, 20, 16, 11, 11, 11, 11, 11, 11, 21, 13, 21] |
| stage1-step100000-tokens210B | days | 0.00 | L10 | 0.13 | 0.40 | 0.59 (0.05) | 0.10 (0.09) | 0.00 (0.03) | 0.17 (0.20) | [29, 36, 21, 30, 29, 40, 30] |
| stage1-step100000-tokens210B | months | 0.00 | L10 | 0.13 | 0.25 | 0.42 (0.04) | 0.08 (0.07) | 0.02 (0.03) | 0.09 (0.11) | [50, 21, 20, 26, 20, 34, 21, 29, 29, 21, 26, 26] |
| stage1-step100000-tokens210B | letters | 0.00 | L4 | 0.09 | 2.17 | 0.63 (0.13) | 0.19 (0.16) | 0.04 (0.04) | - | [42, 12, 0, 12, 1, 2, 1, 52, 11, 6, 5, 2] |
| stage1-step100000-tokens210B | numwords | - | L14 | 0.28 | 0.50 | 0.45 (0.08) | 0.20 (0.13) | 0.06 (0.05) | - | [1, 0, 0, 0, -1, 0, -1, -2, -1, 0, 0, 0] |
| stage1-step20000-tokens42B | days | 0.00 | L8 | 0.13 | 1.46 | 0.51 (0.05) | 0.07 (0.07) | 0.03 (0.03) | 0.15 (0.19) | [-7, -18, 5, -17, -11, -23, -11] |
| stage1-step20000-tokens42B | months | 0.00 | L10 | 0.16 | 1.16 | 0.68 (0.04) | 0.09 (0.07) | 0.05 (0.03) | 0.09 (0.11) | [-1, -1, 4, 0, -3, 8, 2, 1, -13, 4, 2, 1] |
| stage1-step20000-tokens42B | letters | 0.00 | L2 | 0.07 | 0.84 | 0.48 (0.12) | 0.20 (0.16) | 0.06 (0.06) | - | [7, 21, 11, 19, 21, 11, 11, 13, 9, 21, 20, 20] |
| stage1-step20000-tokens42B | numwords | - | L14 | 0.23 | 1.23 | 0.83 (0.09) | 0.16 (0.13) | 0.10 (0.06) | - | [11, 2, 11, 12, 7, 12, 4, 17, 18, 15, 12, 18] |
| stage1-step300000-tokens630B | days | 0.00 | L8 | 0.14 | 0.61 | 0.57 (0.05) | 0.08 (0.07) | 0.00 (0.01) | 0.15 (0.19) | [43, 49, 52, 49, 49, 42, 61] |
| stage1-step300000-tokens630B | months | 0.00 | L6 | 0.11 | 0.89 | 0.58 (0.05) | 0.10 (0.08) | 0.07 (0.04) | 0.10 (0.11) | [5, 10, 4, 12, 6, 6, 8, 7, 6, 0, 13, 13] |
| stage1-step300000-tokens630B | letters | 0.00 | L4 | 0.12 | 1.69 | 0.45 (0.13) | 0.17 (0.16) | 0.02 (0.03) | - | [50, -6, 4, 1, 0, -6, 50, 4, 21, 0, 1, -2] |
| stage1-step300000-tokens630B | numwords | - | L12 | 0.25 | 0.90 | 0.68 (0.08) | 0.20 (0.13) | 0.13 (0.06) | - | [21, 12, 10, 8, 9, 12, 11, 8, 9, 10, 11, 13] |
| stage1-step40000-tokens84B | days | 0.00 | L8 | 0.14 | 1.43 | 0.67 (0.05) | 0.08 (0.07) | 0.06 (0.04) | 0.17 (0.19) | [-2, 4, 3, 2, 9, 3, 9] |
| stage1-step40000-tokens84B | months | 0.00 | L14 | 0.17 | 0.85 | 0.71 (0.04) | 0.11 (0.08) | 0.05 (0.03) | 0.11 (0.11) | [20, -5, 5, 2, 0, 4, 5, 4, 7, 12, 15, -11] |
| stage1-step40000-tokens84B | letters | 0.00 | L2 | 0.03 | 0.76 | 0.54 (0.12) | 0.24 (0.16) | 0.05 (0.06) | - | [6, 12, 2, 12, 12, 12, 12, 12, 18, 12, 12, 12] |
| stage1-step40000-tokens84B | numwords | - | L14 | 0.26 | 0.75 | 0.64 (0.08) | 0.18 (0.13) | 0.16 (0.06) | - | [0, -8, 3, -3, 5, 6, 2, 7, 9, 10, 11, 5] |
| stage1-step800000-tokens1678B | days | 0.00 | L10 | 0.16 | 0.33 | 0.34 (0.05) | 0.07 (0.07) | 0.00 (0.01) | 0.16 (0.19) | [36, 22, 30, 34, 38, 30, 32] |
| stage1-step800000-tokens1678B | months | 0.00 | L2 | 0.10 | 0.95 | 0.72 (0.04) | 0.09 (0.07) | 0.04 (0.03) | 0.09 (0.11) | [22, 20, 19, 25, 35, 23, 7, 23, 22, 12, 21, 23] |
| stage1-step800000-tokens1678B | letters | 0.00 | L4 | 0.17 | 2.53 | 0.41 (0.12) | 0.18 (0.16) | 0.01 (0.02) | - | [9, 1, -2, 0, -6, -1, 0, -1, 60, 43, -1, -7] |
| stage1-step800000-tokens1678B | numwords | - | L14 | 0.26 | 0.74 | 0.66 (0.08) | 0.25 (0.13) | 0.06 (0.05) | - | [1, -8, 3, -4, -1, -4, -1, 0, -1, 0, 1, 2] |

## operand route ("Q: item" decoded on the number line; layer with the highest |spearman|)

| ckpt | domain | layer | probe acc (numeric) | spearman | decoded |
|---|---|---|---|---|---|
| stage1-step10000-tokens21B | days | L12 | 0.94 | -0.54 | [99, 101, 99, 89, 89, 91, 91] |
| stage1-step10000-tokens21B | months | L12 | 0.94 | -0.53 | [93, 91, 89, 93, 89, 91, 79, 79, 79, 89, 89, 89] |
| stage1-step10000-tokens21B | letters | L8 | 0.96 | 0.17 | [100, 100, 100, 110, 100, 94, 104, 94, 100, 100, 100, 106] |
| stage1-step10000-tokens21B | numwords | L2 | 0.94 | 0.88 | [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| stage1-step100000-tokens210B | days | L8 | 0.98 | 0.96 | [8, 21, 22, 33, 30, 82, 102] |
| stage1-step100000-tokens210B | months | L2 | 0.98 | -0.56 | [101, 102, 103, 104, 5, 106, 97, 98, 99, 100, 91, 92] |
| stage1-step100000-tokens210B | letters | L14 | 0.96 | 0.44 | [98, 92, 92, 92, 92, 78, 98, 98, 117, 98, 98, 101] |
| stage1-step100000-tokens210B | numwords | L14 | 0.96 | 0.93 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 11, 12] |
| stage1-step20000-tokens42B | days | L4 | 0.97 | 0.71 | [16, 15, 14, 25, 24, 16, 106] |
| stage1-step20000-tokens42B | months | L12 | 0.96 | -0.66 | [101, 102, 103, 103, 113, 62, 43, 52, 59, 93, 82, 34] |
| stage1-step20000-tokens42B | letters | L4 | 0.97 | 0.30 | [91, 91, 92, 102, 100, 95, 94, 93, 101, 97, 101, 101] |
| stage1-step20000-tokens42B | numwords | L4 | 0.97 | 0.90 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22] |
| stage1-step300000-tokens630B | days | L14 | 0.94 | -0.79 | [98, 101, 110, 93, 95, 81, 83] |
| stage1-step300000-tokens630B | months | L4 | 1.00 | -0.62 | [101, 102, 103, 104, 105, 96, 97, 98, 99, 100, 101, 94] |
| stage1-step300000-tokens630B | letters | L12 | 0.96 | -0.40 | [98, 93, 102, 98, 99, 102, 98, 102, 100, 97, 99, 101] |
| stage1-step300000-tokens630B | numwords | L2 | 0.99 | 1.00 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| stage1-step40000-tokens84B | days | L14 | 0.95 | 0.39 | [82, 81, 62, 62, 66, 82, 82] |
| stage1-step40000-tokens84B | months | L6 | 0.98 | -0.45 | [101, 102, 104, 104, 115, 106, 115, 108, 99, 100, 91, 84] |
| stage1-step40000-tokens84B | letters | L14 | 0.95 | 0.16 | [4, 22, 46, 26, 82, 26, 46, 76, 110, 37, 116, 122] |
| stage1-step40000-tokens84B | numwords | L4 | 0.98 | 1.00 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| stage1-step800000-tokens1678B | days | L2 | 0.98 | 1.00 | [10, 21, 22, 33, 50, 80, 91] |
| stage1-step800000-tokens1678B | months | L4 | 0.99 | -0.19 | [101, 92, 93, 94, 95, 86, 87, 98, 89, 90, 101, 90] |
| stage1-step800000-tokens1678B | letters | L10 | 0.96 | 0.47 | [91, 91, 101, 101, 101, 101, 101, 101, 101, 111, 111, 101] |
| stage1-step800000-tokens1678B | numwords | L4 | 0.99 | 0.88 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22] |

## alignment: item i vs number i + o (centred cosine, p vs 2000 permutations; CKA, p vs 300)

| ckpt | domain | site | o | cos (null) p | CKA (null) p |
|---|---|---|---|---|---|
| stage1-step10000-tokens21B | days | emb_in | 0 | 0.026 (-0.001) 0.058 | 0.97 (0.97) 0.173 |
| stage1-step10000-tokens21B | days | emb_in | 1 | 0.031 (0.000) 0.028 | 0.97 (0.97) 0.206 |
| stage1-step10000-tokens21B | days | emb_out | 0 | 0.016 (-0.000) 0.071 | 0.91 (0.82) 0.023 |
| stage1-step10000-tokens21B | days | emb_out | 1 | 0.012 (0.000) 0.140 | 0.94 (0.83) 0.007 |
| stage1-step10000-tokens21B | days | L6 | 0 | 0.037 (0.000) 0.093 | 0.89 (0.78) 0.010 |
| stage1-step10000-tokens21B | days | L6 | 1 | 0.053 (0.000) 0.031 | 0.92 (0.80) 0.003 |
| stage1-step10000-tokens21B | months | emb_in | 0 | -0.021 (0.000) 0.902 | 0.96 (0.95) 0.033 |
| stage1-step10000-tokens21B | months | emb_in | 1 | 0.071 (-0.000) 0.000 | 0.97 (0.96) 0.010 |
| stage1-step10000-tokens21B | months | emb_out | 0 | 0.009 (0.000) 0.240 | 0.85 (0.77) 0.017 |
| stage1-step10000-tokens21B | months | emb_out | 1 | 0.097 (0.000) 0.000 | 0.87 (0.76) 0.003 |
| stage1-step10000-tokens21B | months | L8 | 0 | 0.001 (-0.000) 0.480 | 0.85 (0.73) 0.010 |
| stage1-step10000-tokens21B | months | L8 | 1 | 0.126 (-0.000) 0.000 | 0.89 (0.75) 0.003 |
| stage1-step10000-tokens21B | letters | emb_in | 0 | 0.021 (0.001) 0.050 | 0.88 (0.88) 0.478 |
| stage1-step10000-tokens21B | letters | emb_in | 1 | 0.008 (-0.000) 0.260 | 0.89 (0.88) 0.090 |
| stage1-step10000-tokens21B | letters | emb_out | 0 | 0.003 (-0.000) 0.215 | 0.78 (0.76) 0.003 |
| stage1-step10000-tokens21B | letters | emb_out | 1 | 0.009 (0.000) 0.012 | 0.79 (0.76) 0.003 |
| stage1-step10000-tokens21B | letters | L2 | 0 | 0.013 (0.000) 0.101 | 0.74 (0.71) 0.023 |
| stage1-step10000-tokens21B | letters | L2 | 1 | 0.008 (-0.000) 0.194 | 0.76 (0.71) 0.003 |
| stage1-step10000-tokens21B | numwords | emb_in | 0 | 0.084 (-0.000) 0.001 | 0.94 (0.85) 0.003 |
| stage1-step10000-tokens21B | numwords | emb_in | 1 | 0.373 (0.001) 0.000 | 0.96 (0.85) 0.003 |
| stage1-step10000-tokens21B | numwords | emb_out | 0 | 0.140 (0.001) 0.001 | 0.92 (0.52) 0.003 |
| stage1-step10000-tokens21B | numwords | emb_out | 1 | 0.420 (0.001) 0.000 | 0.93 (0.52) 0.003 |
| stage1-step10000-tokens21B | numwords | L4 | 0 | 0.237 (0.000) 0.000 | 0.94 (0.55) 0.003 |
| stage1-step10000-tokens21B | numwords | L4 | 1 | 0.637 (0.000) 0.000 | 0.95 (0.55) 0.003 |
| stage1-step100000-tokens210B | days | emb_in | 0 | 0.050 (-0.001) 0.030 | 0.94 (0.87) 0.017 |
| stage1-step100000-tokens210B | days | emb_in | 1 | 0.041 (0.000) 0.057 | 0.97 (0.88) 0.003 |
| stage1-step100000-tokens210B | days | emb_out | 0 | 0.033 (0.000) 0.038 | 0.91 (0.81) 0.020 |
| stage1-step100000-tokens210B | days | emb_out | 1 | 0.037 (0.001) 0.027 | 0.95 (0.81) 0.007 |
| stage1-step100000-tokens210B | days | L10 | 0 | 0.085 (-0.001) 0.007 | 0.83 (0.73) 0.027 |
| stage1-step100000-tokens210B | days | L10 | 1 | 0.079 (0.001) 0.031 | 0.91 (0.77) 0.003 |
| stage1-step100000-tokens210B | months | emb_in | 0 | 0.028 (-0.001) 0.160 | 0.95 (0.87) 0.003 |
| stage1-step100000-tokens210B | months | emb_in | 1 | 0.305 (0.000) 0.000 | 0.95 (0.86) 0.003 |
| stage1-step100000-tokens210B | months | emb_out | 0 | 0.010 (0.000) 0.300 | 0.86 (0.77) 0.007 |
| stage1-step100000-tokens210B | months | emb_out | 1 | 0.240 (0.000) 0.000 | 0.88 (0.76) 0.003 |
| stage1-step100000-tokens210B | months | L10 | 0 | 0.073 (-0.001) 0.047 | 0.80 (0.70) 0.040 |
| stage1-step100000-tokens210B | months | L10 | 1 | 0.417 (0.001) 0.000 | 0.87 (0.73) 0.003 |
| stage1-step100000-tokens210B | letters | emb_in | 0 | 0.007 (0.000) 0.142 | 0.82 (0.76) 0.003 |
| stage1-step100000-tokens210B | letters | emb_in | 1 | 0.011 (-0.000) 0.054 | 0.82 (0.77) 0.003 |
| stage1-step100000-tokens210B | letters | emb_out | 0 | 0.002 (0.000) 0.279 | 0.78 (0.75) 0.003 |
| stage1-step100000-tokens210B | letters | emb_out | 1 | 0.008 (0.000) 0.009 | 0.79 (0.75) 0.003 |
| stage1-step100000-tokens210B | letters | L8 | 0 | 0.005 (0.000) 0.327 | 0.64 (0.55) 0.023 |
| stage1-step100000-tokens210B | letters | L8 | 1 | 0.024 (0.000) 0.007 | 0.66 (0.55) 0.003 |
| stage1-step100000-tokens210B | numwords | emb_in | 0 | 0.165 (0.001) 0.000 | 0.95 (0.62) 0.003 |
| stage1-step100000-tokens210B | numwords | emb_in | 1 | 0.568 (0.002) 0.000 | 0.96 (0.62) 0.003 |
| stage1-step100000-tokens210B | numwords | emb_out | 0 | 0.190 (0.001) 0.001 | 0.94 (0.53) 0.003 |
| stage1-step100000-tokens210B | numwords | emb_out | 1 | 0.576 (0.001) 0.000 | 0.96 (0.53) 0.003 |
| stage1-step100000-tokens210B | numwords | L10 | 0 | 0.250 (-0.001) 0.000 | 0.94 (0.50) 0.003 |
| stage1-step100000-tokens210B | numwords | L10 | 1 | 0.704 (-0.001) 0.000 | 0.98 (0.51) 0.003 |
| stage1-step20000-tokens42B | days | emb_in | 0 | 0.029 (-0.001) 0.072 | 0.97 (0.96) 0.056 |
| stage1-step20000-tokens42B | days | emb_in | 1 | 0.033 (0.000) 0.055 | 0.98 (0.96) 0.037 |
| stage1-step20000-tokens42B | days | emb_out | 0 | 0.026 (-0.000) 0.029 | 0.92 (0.81) 0.017 |
| stage1-step20000-tokens42B | days | emb_out | 1 | 0.021 (0.000) 0.074 | 0.95 (0.82) 0.007 |
| stage1-step20000-tokens42B | days | L4 | 0 | 0.027 (0.000) 0.169 | 0.88 (0.77) 0.003 |
| stage1-step20000-tokens42B | days | L4 | 1 | 0.060 (0.000) 0.020 | 0.93 (0.81) 0.003 |
| stage1-step20000-tokens42B | months | emb_in | 0 | -0.022 (-0.000) 0.880 | 0.97 (0.95) 0.003 |
| stage1-step20000-tokens42B | months | emb_in | 1 | 0.138 (-0.000) 0.000 | 0.97 (0.95) 0.003 |
| stage1-step20000-tokens42B | months | emb_out | 0 | 0.005 (0.000) 0.350 | 0.85 (0.76) 0.013 |
| stage1-step20000-tokens42B | months | emb_out | 1 | 0.145 (0.000) 0.000 | 0.87 (0.75) 0.003 |
| stage1-step20000-tokens42B | months | L6 | 0 | 0.039 (-0.000) 0.076 | 0.86 (0.74) 0.007 |
| stage1-step20000-tokens42B | months | L6 | 1 | 0.233 (-0.000) 0.000 | 0.90 (0.76) 0.003 |
| stage1-step20000-tokens42B | letters | emb_in | 0 | 0.016 (0.000) 0.075 | 0.88 (0.87) 0.070 |
| stage1-step20000-tokens42B | letters | emb_in | 1 | 0.010 (-0.000) 0.197 | 0.89 (0.87) 0.017 |
| stage1-step20000-tokens42B | letters | emb_out | 0 | 0.004 (-0.000) 0.118 | 0.78 (0.74) 0.003 |
| stage1-step20000-tokens42B | letters | emb_out | 1 | 0.010 (0.000) 0.002 | 0.79 (0.74) 0.003 |
| stage1-step20000-tokens42B | letters | L2 | 0 | 0.013 (0.000) 0.072 | 0.75 (0.70) 0.003 |
| stage1-step20000-tokens42B | letters | L2 | 1 | 0.006 (0.000) 0.244 | 0.76 (0.71) 0.003 |
| stage1-step20000-tokens42B | numwords | emb_in | 0 | 0.093 (-0.000) 0.002 | 0.94 (0.81) 0.003 |
| stage1-step20000-tokens42B | numwords | emb_in | 1 | 0.463 (0.001) 0.000 | 0.96 (0.81) 0.003 |
| stage1-step20000-tokens42B | numwords | emb_out | 0 | 0.165 (0.001) 0.001 | 0.93 (0.52) 0.003 |
| stage1-step20000-tokens42B | numwords | emb_out | 1 | 0.498 (0.001) 0.000 | 0.94 (0.52) 0.003 |
| stage1-step20000-tokens42B | numwords | L4 | 0 | 0.252 (0.000) 0.000 | 0.95 (0.54) 0.003 |
| stage1-step20000-tokens42B | numwords | L4 | 1 | 0.671 (0.001) 0.000 | 0.97 (0.55) 0.003 |
| stage1-step300000-tokens630B | days | emb_in | 0 | 0.083 (-0.000) 0.003 | 0.94 (0.85) 0.007 |
| stage1-step300000-tokens630B | days | emb_in | 1 | 0.060 (0.000) 0.019 | 0.96 (0.86) 0.003 |
| stage1-step300000-tokens630B | days | emb_out | 0 | 0.053 (-0.000) 0.003 | 0.92 (0.83) 0.020 |
| stage1-step300000-tokens630B | days | emb_out | 1 | 0.040 (0.001) 0.033 | 0.95 (0.83) 0.003 |
| stage1-step300000-tokens630B | days | L10 | 0 | 0.093 (-0.001) 0.009 | 0.81 (0.73) 0.083 |
| stage1-step300000-tokens630B | days | L10 | 1 | 0.061 (0.000) 0.097 | 0.90 (0.80) 0.007 |
| stage1-step300000-tokens630B | months | emb_in | 0 | 0.030 (-0.001) 0.152 | 0.95 (0.86) 0.003 |
| stage1-step300000-tokens630B | months | emb_in | 1 | 0.325 (0.000) 0.000 | 0.95 (0.85) 0.003 |
| stage1-step300000-tokens630B | months | emb_out | 0 | 0.008 (0.001) 0.336 | 0.86 (0.78) 0.010 |
| stage1-step300000-tokens630B | months | emb_out | 1 | 0.273 (0.000) 0.000 | 0.88 (0.77) 0.003 |
| stage1-step300000-tokens630B | months | L10 | 0 | 0.057 (-0.001) 0.109 | 0.79 (0.72) 0.083 |
| stage1-step300000-tokens630B | months | L10 | 1 | 0.483 (0.002) 0.000 | 0.88 (0.76) 0.003 |
| stage1-step300000-tokens630B | letters | emb_in | 0 | 0.009 (0.000) 0.055 | 0.80 (0.74) 0.003 |
| stage1-step300000-tokens630B | letters | emb_in | 1 | 0.011 (-0.000) 0.017 | 0.81 (0.75) 0.003 |
| stage1-step300000-tokens630B | letters | emb_out | 0 | 0.004 (0.000) 0.113 | 0.79 (0.76) 0.003 |
| stage1-step300000-tokens630B | letters | emb_out | 1 | 0.006 (0.000) 0.033 | 0.80 (0.77) 0.003 |
| stage1-step300000-tokens630B | letters | L8 | 0 | 0.006 (0.000) 0.319 | 0.59 (0.50) 0.050 |
| stage1-step300000-tokens630B | letters | L8 | 1 | 0.025 (0.000) 0.022 | 0.60 (0.50) 0.003 |
| stage1-step300000-tokens630B | numwords | emb_in | 0 | 0.187 (0.002) 0.000 | 0.93 (0.56) 0.003 |
| stage1-step300000-tokens630B | numwords | emb_in | 1 | 0.587 (0.002) 0.000 | 0.95 (0.57) 0.003 |
| stage1-step300000-tokens630B | numwords | emb_out | 0 | 0.183 (0.001) 0.001 | 0.94 (0.54) 0.003 |
| stage1-step300000-tokens630B | numwords | emb_out | 1 | 0.593 (0.001) 0.000 | 0.96 (0.54) 0.003 |
| stage1-step300000-tokens630B | numwords | L10 | 0 | 0.234 (-0.001) 0.000 | 0.91 (0.53) 0.003 |
| stage1-step300000-tokens630B | numwords | L10 | 1 | 0.728 (-0.001) 0.000 | 0.97 (0.54) 0.003 |
| stage1-step40000-tokens84B | days | emb_in | 0 | 0.043 (-0.000) 0.029 | 0.97 (0.93) 0.007 |
| stage1-step40000-tokens84B | days | emb_in | 1 | 0.029 (0.000) 0.107 | 0.98 (0.94) 0.003 |
| stage1-step40000-tokens84B | days | emb_out | 0 | 0.022 (0.000) 0.072 | 0.91 (0.80) 0.020 |
| stage1-step40000-tokens84B | days | emb_out | 1 | 0.030 (0.001) 0.023 | 0.95 (0.81) 0.007 |
| stage1-step40000-tokens84B | days | L8 | 0 | 0.048 (0.001) 0.038 | 0.81 (0.68) 0.013 |
| stage1-step40000-tokens84B | days | L8 | 1 | 0.066 (-0.000) 0.010 | 0.89 (0.72) 0.003 |
| stage1-step40000-tokens84B | months | emb_in | 0 | -0.008 (-0.000) 0.596 | 0.97 (0.93) 0.003 |
| stage1-step40000-tokens84B | months | emb_in | 1 | 0.222 (0.000) 0.000 | 0.97 (0.93) 0.003 |
| stage1-step40000-tokens84B | months | emb_out | 0 | 0.024 (0.000) 0.138 | 0.85 (0.76) 0.013 |
| stage1-step40000-tokens84B | months | emb_out | 1 | 0.206 (0.000) 0.000 | 0.87 (0.75) 0.003 |
| stage1-step40000-tokens84B | months | L4 | 0 | 0.055 (0.001) 0.074 | 0.88 (0.76) 0.007 |
| stage1-step40000-tokens84B | months | L4 | 1 | 0.325 (0.001) 0.000 | 0.93 (0.78) 0.003 |
| stage1-step40000-tokens84B | letters | emb_in | 0 | 0.011 (0.000) 0.123 | 0.87 (0.85) 0.010 |
| stage1-step40000-tokens84B | letters | emb_in | 1 | 0.015 (-0.000) 0.058 | 0.88 (0.85) 0.003 |
| stage1-step40000-tokens84B | letters | emb_out | 0 | 0.006 (0.000) 0.032 | 0.78 (0.74) 0.003 |
| stage1-step40000-tokens84B | letters | emb_out | 1 | 0.012 (0.000) 0.000 | 0.79 (0.75) 0.003 |
| stage1-step40000-tokens84B | letters | L8 | 0 | 0.000 (0.001) 0.527 | 0.62 (0.55) 0.060 |
| stage1-step40000-tokens84B | letters | L8 | 1 | 0.017 (0.000) 0.029 | 0.63 (0.55) 0.003 |
| stage1-step40000-tokens84B | numwords | emb_in | 0 | 0.105 (-0.000) 0.002 | 0.95 (0.77) 0.003 |
| stage1-step40000-tokens84B | numwords | emb_in | 1 | 0.508 (0.001) 0.000 | 0.97 (0.77) 0.003 |
| stage1-step40000-tokens84B | numwords | emb_out | 0 | 0.183 (0.001) 0.001 | 0.94 (0.51) 0.003 |
| stage1-step40000-tokens84B | numwords | emb_out | 1 | 0.537 (0.001) 0.000 | 0.95 (0.51) 0.003 |
| stage1-step40000-tokens84B | numwords | L10 | 0 | 0.266 (-0.000) 0.000 | 0.95 (0.47) 0.003 |
| stage1-step40000-tokens84B | numwords | L10 | 1 | 0.719 (-0.000) 0.000 | 0.98 (0.48) 0.003 |
| stage1-step800000-tokens1678B | days | emb_in | 0 | 0.096 (-0.000) 0.003 | 0.94 (0.86) 0.007 |
| stage1-step800000-tokens1678B | days | emb_in | 1 | 0.056 (0.000) 0.030 | 0.96 (0.87) 0.003 |
| stage1-step800000-tokens1678B | days | emb_out | 0 | 0.076 (-0.000) 0.001 | 0.92 (0.82) 0.020 |
| stage1-step800000-tokens1678B | days | emb_out | 1 | 0.037 (0.001) 0.084 | 0.94 (0.82) 0.007 |
| stage1-step800000-tokens1678B | days | L10 | 0 | 0.118 (-0.001) 0.001 | 0.81 (0.72) 0.056 |
| stage1-step800000-tokens1678B | days | L10 | 1 | 0.034 (0.000) 0.211 | 0.89 (0.76) 0.007 |
| stage1-step800000-tokens1678B | months | emb_in | 0 | 0.026 (-0.001) 0.174 | 0.95 (0.86) 0.003 |
| stage1-step800000-tokens1678B | months | emb_in | 1 | 0.302 (0.000) 0.000 | 0.94 (0.84) 0.003 |
| stage1-step800000-tokens1678B | months | emb_out | 0 | 0.013 (0.001) 0.303 | 0.86 (0.76) 0.003 |
| stage1-step800000-tokens1678B | months | emb_out | 1 | 0.287 (0.000) 0.000 | 0.87 (0.74) 0.003 |
| stage1-step800000-tokens1678B | months | L10 | 0 | 0.045 (-0.001) 0.141 | 0.74 (0.66) 0.093 |
| stage1-step800000-tokens1678B | months | L10 | 1 | 0.452 (0.001) 0.000 | 0.81 (0.69) 0.017 |
| stage1-step800000-tokens1678B | letters | emb_in | 0 | 0.013 (0.000) 0.007 | 0.80 (0.74) 0.007 |
| stage1-step800000-tokens1678B | letters | emb_in | 1 | 0.010 (0.000) 0.033 | 0.80 (0.75) 0.003 |
| stage1-step800000-tokens1678B | letters | emb_out | 0 | 0.003 (0.000) 0.195 | 0.78 (0.76) 0.003 |
| stage1-step800000-tokens1678B | letters | emb_out | 1 | 0.002 (0.000) 0.230 | 0.79 (0.77) 0.003 |
| stage1-step800000-tokens1678B | letters | L6 | 0 | 0.012 (0.000) 0.109 | 0.60 (0.51) 0.027 |
| stage1-step800000-tokens1678B | letters | L6 | 1 | 0.020 (0.000) 0.021 | 0.63 (0.52) 0.003 |
| stage1-step800000-tokens1678B | numwords | emb_in | 0 | 0.190 (0.002) 0.000 | 0.93 (0.57) 0.003 |
| stage1-step800000-tokens1678B | numwords | emb_in | 1 | 0.592 (0.002) 0.000 | 0.95 (0.57) 0.003 |
| stage1-step800000-tokens1678B | numwords | emb_out | 0 | 0.183 (0.001) 0.001 | 0.94 (0.54) 0.003 |
| stage1-step800000-tokens1678B | numwords | emb_out | 1 | 0.611 (0.001) 0.000 | 0.96 (0.54) 0.003 |
| stage1-step800000-tokens1678B | numwords | L12 | 0 | 0.279 (0.003) 0.000 | 0.91 (0.43) 0.003 |
| stage1-step800000-tokens1678B | numwords | L12 | 1 | 0.742 (0.003) 0.000 | 0.97 (0.42) 0.003 |
