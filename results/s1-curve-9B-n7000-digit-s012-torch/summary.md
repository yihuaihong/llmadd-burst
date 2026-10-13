# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | test_terse | three_term | three_digit | word0 | word1 | word2 | gpt3 | gpt3_unseen | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| digit | 3 | 0.999 ± 0.002 | 1.000 ± 0.000 | 0.009 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.001 | 0.003 ± 0.002 | 0.004 ± 0.001 | 0.014 ± 0.009 | 0.011 ± 0.011 | 0.30 / 0.87 CKA@L16 helix 0.62 digit 1.00; emb helix 0.30 digit 0.44 |
