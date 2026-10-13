# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | test_terse | three_term | three_digit | word0 | word1 | word2 | gpt3 | gpt3_unseen | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.003 | 0.007 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| helix_shuf | 3 | 0.394 ± 0.222 | 0.741 ± 0.152 | 0.002 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.12 / -0.27 CKA@L16 helix 0.08 digit 0.13; emb helix 0.50 digit 0.60 |
