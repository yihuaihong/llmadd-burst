# GPU usage (from tools/compute_usage.py, 2026-10-13; Burst records from 2026-09-27; Torch jobs finished by then)

| family | Burst A100-40GB | Torch A100-80GB | Torch H100 | total GPU-h |
|---|---|---|---|---|
| fine-tuning sweeps (sizes, checkpoints, seeds, methods) | 73.4 | 3.5 | 2.0 | 78.9 |
| concept domains (7-domain window, months / weekdays) | 39.2 | 0.0 | 0.0 | 39.2 |
| from-scratch pretraining | 0.0 | 13.4 | 6.9 | 20.3 |
| learning curves (Fig. 1) | 4.0 | 4.4 | 6.1 | 14.5 |
| multiplication / mixed ops / structure | 0.0 | 4.5 | 3.6 | 8.1 |
| data volume x checkpoint | 4.4 | 0.0 | 1.2 | 5.6 |
| toy grokking | 2.6 | 0.0 | 1.6 | 4.2 |
| regularity dose / self-timed prior | 0.0 | 3.8 | 0.4 | 4.2 |
| component ablations (sep / nonsep / coarse) | 3.0 | 0.0 | 0.0 | 3.0 |
| general-ability evals | 0.0 | 1.5 | 1.5 | 3.0 |
| geometry diagnostics (kappa, spike) | 0.9 | 0.0 | 0.0 | 0.9 |
| model downloads / setup / smoke tests | 0.2 | 0.0 | 0.0 | 0.2 |
| **total** | 127.7 | 30.9 | 23.3 | 181.9 |
