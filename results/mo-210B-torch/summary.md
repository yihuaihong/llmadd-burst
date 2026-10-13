# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mixed, split random)

| geom | n seeds | test | train | test_terse | test_sub | test_mul | gpt3_unseen_add | gpt3_unseen_sub | gpt3_unseen_mul | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.483 | 0.537 | 0.135 | 0.644 | 0.414 | 0.232 | 0.563 | 0.417 | - |
| none | 3 | 0.964 ± 0.003 | 0.997 ± 0.001 | 0.340 ± 0.018 | 0.970 ± 0.001 | 0.771 ± 0.013 | 0.708 ± 0.019 | 0.792 ± 0.014 | 0.661 ± 0.021 | 0.27 / 0.33 CKA@L16 helix 0.75 digit 0.63; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.929 ± 0.020 | 0.988 ± 0.009 | 0.420 ± 0.014 | 0.955 ± 0.037 | 0.683 ± 0.009 | 0.687 ± 0.013 | 0.739 ± 0.041 | 0.606 ± 0.024 | 0.73 / 0.73 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.881 ± 0.016 | 0.975 ± 0.013 | 0.410 ± 0.032 | 0.899 ± 0.035 | 0.715 ± 0.009 | 0.667 ± 0.010 | 0.669 ± 0.044 | 0.683 ± 0.030 | -0.08 / -0.21 CKA@L16 helix 0.12 digit 0.18; emb helix 0.65 digit 0.72 |
