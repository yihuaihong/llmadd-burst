# Results report: code and data index

Summary for collaborators: the Google Doc "数字与结构化概念的表征流形：实验结果汇总" (2026-10-02).
Full chronological log with every number: [FINDINGS.md](FINDINGS.md) (entries F1 to F34).
Figures: [figures/](figures/).

Every run is a task file `tasks/NNN-name.sh` (the exact command), its outputs are in `results/NNN-name/`
(`summary.md`, one `run_<geometry>_seed<k>.json` per run, `base.json`, `log_tail.txt`) and its state is in
`status/NNN-name.json`. A `b` suffix (e.g. `105b`) is a re-run of a task that hit a node or memory failure.

| Script | What it does | Findings | Tasks |
|---|---|---|---|
| `tools/manifold_train.py` | LoRA / ReFT / full fine-tuning with the CKA geometry loss (helix, digit, shuffled; 0..999 variants; learned targets main / self; embedding site; hard transplant; holdout and carry splits; subtraction; `--save_preds`; `--device_map auto` for 32B) | F10–F23, F26, F27, F32, F34 | 064–126, 161–172, 199–202, 237b–240b, 258–262, 272–273 |
| `tools/manifold_train_fsdp.py` | full fine-tuning on 2 x 80 GB (FSDP) | F16 | Torch runs `v2fsdp_*` |
| `tools/concept_train.py` | the same protocol on number words, Roman numerals, days, months, letters | F24, F25, F29, F34 | 127–156, 173–192, 263–270 |
| `tools/pythia_train.py` | the same protocol on Pythia (GPT-NeoX tokenizer) | F30, F33 | 203–230, 241–254 |
| `tools/mse_control.py` | point-wise MSE pull towards main's states, with shuffled / self / random controls | F31 | 231–234, 255–257 |
| `tools/exp4_snap.py` | inference-time snap / shuffle of operand coordinates | F8, F28 | 050–055, 193–198 |
| `tools/emb_manifold.py` | embedding geometry over 19 checkpoints | F9, F13 | 076 |
| `tools/error_analysis.py` | error anatomy of saved predictions | F21 | 112, 113 |
| `tools/exp1_steer.py`, `exp1b_digits.py`, `exp2*.py`, `exp3_robust.py` | early probing and editing analyses | F1–F7 | 009–049 |
| `tools/fetch_olmo.py`, `tools/fetch_hf.py` | checkpoint downloads (OLMo-2 1B / 7B / 13B / 32B, Pythia) | | 002–008, 080, 157–160, 203–224, 235–236, 241, 248, 271 |
| `burst/` | the git-driven scheduler on NYU Burst (tick, retries, lock) | | |
