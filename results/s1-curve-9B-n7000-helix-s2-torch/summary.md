# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | test_terse | three_term | three_digit | word0 | word1 | word2 | gpt3 | gpt3_unseen | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| helix | 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.019 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | 0.057 ± 0.000 | 0.058 ± 0.000 | 0.88 / 0.85 CKA@L16 helix 1.00 digit 0.62; emb helix 0.30 digit 0.44 |
