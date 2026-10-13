# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mixed, split random)

| geom | n seeds | test | train | test_terse | test_sub | test_mul | gpt3_unseen_add | gpt3_unseen_sub | gpt3_unseen_mul | lm_val_loss | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.080 | 0.095 | 0.016 | 0.074 | 0.134 | 0.019 | 0.054 | 0.108 | 2.748 | - |
| none | 3 | 0.563 ± 0.024 | 0.936 ± 0.015 | 0.094 ± 0.005 | 0.566 ± 0.009 | 0.641 ± 0.007 | 0.127 ± 0.014 | 0.235 ± 0.004 | 0.283 ± 0.014 | 2.749 ± 0.000 | 0.29 / 0.36 CKA@L16 helix 0.73 digit 0.56; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.662 ± 0.068 | 0.932 ± 0.029 | 0.174 ± 0.007 | 0.753 ± 0.015 | 0.534 ± 0.007 | 0.219 ± 0.008 | 0.310 ± 0.024 | 0.314 ± 0.017 | 2.749 ± 0.000 | 0.73 / 0.73 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.71 |
| helix_digitf | 3 | 0.674 ± 0.030 | 0.935 ± 0.013 | 0.158 ± 0.006 | 0.683 ± 0.047 | 0.515 ± 0.017 | 0.219 ± 0.033 | 0.273 ± 0.029 | 0.328 ± 0.025 | 2.749 ± 0.000 | 0.62 / 0.71 CKA@L16 helix 0.95 digit 0.75; emb helix 0.65 digit 0.71 |
