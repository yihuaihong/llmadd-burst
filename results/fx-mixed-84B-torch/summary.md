# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mixed, split random)

| geom | n seeds | test | train | test_terse | test_sub | test_mul | gpt3_unseen_add | gpt3_unseen_sub | gpt3_unseen_mul | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.080 | 0.095 | 0.016 | 0.074 | 0.134 | 0.019 | 0.054 | 0.108 | - |
| none | 3 | 0.563 ± 0.024 | 0.936 ± 0.015 | 0.094 ± 0.005 | 0.566 ± 0.009 | 0.641 ± 0.007 | 0.127 ± 0.014 | 0.235 ± 0.004 | 0.283 ± 0.014 | 0.29 / 0.36 CKA@L16 helix 0.73 digit 0.56; emb helix 0.65 digit 0.71 |
| factor | 3 | 0.475 ± 0.051 | 0.863 ± 0.042 | 0.148 ± 0.002 | 0.556 ± 0.032 | 0.536 ± 0.027 | 0.223 ± 0.008 | 0.215 ± 0.018 | 0.292 ± 0.025 | 0.28 / 0.52 CKA@L16 helix 0.62 digit 0.71; emb helix 0.65 digit 0.71 |
| helix_factor | 3 | 0.604 ± 0.039 | 0.922 ± 0.004 | 0.174 ± 0.009 | 0.633 ± 0.024 | 0.569 ± 0.012 | 0.249 ± 0.021 | 0.222 ± 0.008 | 0.331 ± 0.013 | 0.45 / 0.60 CKA@L16 helix 0.85 digit 0.75; emb helix 0.65 digit 0.71 |
