# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | test_terse | three_term | three_digit | word0 | word1 | word2 | gpt3 | gpt3_unseen | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| helix | 2 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.045 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.013 ± 0.008 | 0.007 ± 0.006 | 0.013 ± 0.003 | 0.128 ± 0.025 | 0.130 ± 0.027 | 0.85 / 0.83 CKA@L16 helix 1.00 digit 0.63; emb helix 0.30 digit 0.44 |
