# Burst infrastructure for the 2%-main pretraining batch + xsite53 (2026-10-16, Batch 1 B0 / B1)

Intent spec, written before any of these tasks ran on Burst. Plan: `notes/next_phase_synthesis.md` (Batch 1 B0, B1;
section 4) and the burst design in `notes/next_phase_designs.json` (burst-setup-E0, xsite53).

## Questions
- **B0.** Can Burst rebuild the Torch pretraining inputs byte for byte (OLMo-2 tokenizer, `pretrain_data` 4e8 + 2e6
  tokens, `pretrain_data_3B` 3e9 + 2e6), run the trainer at a fixed commit whatever is pushed later, in a Python env
  with Torch's versions? Which accounts, partitions and limits apply (the 1006 account name, CPU billing)?
- **B1 xsite53.** Do 53M runs on Burst A100 40GB reproduce Torch L40S closely enough that Burst cells can be pooled as
  within-site paired effects? How many A100-hours does a 53M 3e8-token run take (this rescales every Burst estimate)?

## Files
| file | role |
|---|---|
| `tools/pretrain_data.py` | `--revision` (HF dataset commit; default None = main, as before); missing parquet files fetched with `huggingface_hub.hf_hub_download` (the `hf` CLI is not on PATH in Burst tasks); tokens streamed to disk (the old in-memory concatenation needs ~2 x 12 GB for 3B; a Burst CPU node has 23 GB, `results/000-hello`). Output bytes unchanged. |
| `tools/hash_ckpt.py` | bf16 tensor digest of a safetensors checkpoint (sha256 over name, shape, bf16 bytes in sorted name order); equal for Torch's fp32 snapshots and Burst's bf16 conversions iff the loaded weights are equal; `--expect` exits 3 on a mismatch, a missing checkpoint is reported and counts as one. |
| `burst/pin_run.sh` | `bash burst/pin_run.sh <commit> <script under tools/> [args...]` (shared interface): extracts `tools/` at the commit once into `$RUNS_DIR/pinned_<commit>` (default `$LLMADD_ROOT/llmadd_runs`; temp dir, checked there, then renamed into place, so concurrent tasks never see half a tree), checks on every call that all tracked files there are the commit's git blobs (a tree that is not, e.g. after a scratch purge or an edit, is moved aside to `pinned_<commit>.bad.<time>.<pid>` and extracted again; 3 attempts), appends `<full commit> <sha256 of the script> tools/<script> <UTC time>` to `$OUT/PINNED.txt`, runs the script with `$LLMADD_ROOT/llmadd_env311/bin/python` from `<pinned>/tools` (args must be absolute paths; no .pyc written). Exit 3 = the pin failed; otherwise the script's exit code. Rejects branch names (they move). |
| `tasks/330-probe.sh` | cpu, read-only: associations / QOS / GrpTRESMins with usage (`scontrol show assoc_mgr`), fairshare, `sbatch --test-only` per account x {CPU, 1-GPU, 2-GPU partition}, GPU-h and CPU-h used per account since 2026-08-01 (jobs of other users counted, never named), partition and node sizes, glibc / python / driver -> `probe.txt`; 3b: each GPU account accepted / refused on the GPU partition, in the tick's order. `probe.txt` goes to the public repo: only the accounts' own rows and this user's rows are queried or kept, any other `User=` / `UserName=` value is redacted. Always exits 0. |
| `tasks/331-env311.sh` | cpu: `/scratch/$USER/llmadd_env311` = CPython 3.11.13 (uv) + 60 pinned packages = the dependency closure, in Torch's open_r1 (+ `llmadd_torch/pylib`), of torch, transformers, tokenizers, numpy, scipy, pyarrow, peft, huggingface_hub, hf-xet, safetensors, accelerate, geonamescache, scikit-learn, matplotlib, tqdm. torch 2.10.0 = the same PyPI wheel as open_r1 (CUDA 12.8, sm_70..sm_120). Idempotent; exit 3 on any version difference; a completed env with other pins is never deleted (exit 3; running tasks may use it). |
| `tasks/332-data.sh` | cpu, after 331: tokenizer files (OLMo-2-1124-7B @ f872bfe9 = the `stage1-step2000-tokens9B` branch = the Torch `--tokenizer` dir) into `llmadd_data/olmo2_tok` + sha256 + token-id probe; `pretrain_data` and `pretrain_data_3B` with the Torch builds' arguments at fineweb-edu @87f09149; sha256 of parquet files and bins (parquet 000 / 001 of the 3B build are hard links to the verified 4e8 ones, no second download); CPU `--dry_inputs` of the pinned trainer vs the Torch log. Mismatch -> dir renamed `<dir>.bad`, exit 3. |
| `tasks/333-hash-olmo.sh` | cpu, after 331, nothing waits for it: bf16 digests of Burst's 42B / 210B checkpoints vs Torch, tokenizer.json of 9B, the other 5 ft36b checkpoints (9B 21B 84B 630B 3896B) complete (`.complete` names the revision, every shard of the index present and not empty), list of checkpoint dirs (for the LoRA batches). Exit 3 if any of this fails. |
| `tasks/340..347-xsite53-*.sh` | gpu, after 330 + 331 + 332: one run each, seeds 0-3 x {none, W}, pinned at 64abc19. |

Shared conventions used: env `/scratch/$USER/llmadd_env311`; data `/scratch/$USER/llmadd_data/{pretrain_data,
pretrain_data_3B,olmo2_tok}`; pin commit 64abc19 (its `tools/pretrain_range.py` = sha256 a3bce3ab63c1...).

## Torch references (computed on Torch 2026-10-16)
- fineweb-edu commit of both Torch builds: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (every
  `raw/.cache/huggingface/download/*.metadata`); HF main is still at this commit today.
- parquet sha256 = the HF LFS sha256 at that commit (`HfApi.get_paths_info`) = `sha256sum` of Torch's files:
  000 b1ba7b2c..., 001 3fcf2dc6..., 002 547ae182..., 003 22184e6e..., 004 33557ddd... (full values in 332).
- bins: `pretrain_data` train 4de2ed19093c37b3..., val d6f744a99b988ed6...; `pretrain_data_3B` train
  5e456d775670562d..., val 4b7c27cfd8613cac.... Consistency: the 4e8 train.bin is the prefix of the 3B train.bin and
  the 4e8 val.bin equals 3B train[4e8 : 4e8 + 2e6] (`cmp`).
- tokenizer: 7 files at OLMo-2-1124-7B f872bfe9 (sha256 in 332; tokenizer.json 73fd5254...); probe ids and the
  extended digest e278563e... computed with transformers 4.57.6 / tokenizers 0.22.2 (open_r1).
- `--dry_inputs` digest per seed: sha256 of the first 9 lines matching
  `^(line:|held_b|ood_b|\{'held_b11'|model:|train row)` in `logs/llmadd_pt999-7pct-none-s<seed>_*.out`:
  s0 af96f9c5..., s1 a261e357..., s2 52ab489c..., s3 01b0d252.... A seed's rows do not depend on the arm (W prints the
  same lines plus "target features").
- OLMo-2 7B bf16 digests: 42B 4133fa61ec69ece6, 210B 08e4c3a401069fd0 (355 tensors each).

Reference cells (seeds 0-3, 18310 steps, L40S, 3440-3476 s per run):
| seed | lm0 (both arms) | val_loss0 none / W | area none (pt999 7pct-none) | area W (ptrangeg w0.05-0.4) | W - none |
|---|---|---|---|---|---|
| 0 | 11.608092 | 11.607071 / 11.607063 | 59.90 | 88.99 | 29.09 |
| 1 | 11.599074 | 11.596082 / 11.596097 | 51.54 | 88.38 | 36.84 |
| 2 | 11.579556 | 11.569641 / 11.569629 | 52.29 | 89.14 | 36.85 |
| 3 | 11.597104 | 11.591120 / 11.591121 | 40.04 | 88.92 | 48.88 |
| mean | | | 50.94 | 88.86 | 37.92 |
b33_area none 46.86 / 22.76 / 30.96 / 16.22; W 80.64 / 79.40 / 80.68 / 83.51. Torch same-seed replicate spread
(pt999 7pct-gate-ctx minus ptrangeg whole-helix3-gate, same flags, two snapshots, recomputed here): area diffs
+0.39 / +0.33 / -1.38 / +0.52, SD 0.90, max 1.38; step-0 lm identical (difference 0.0).

## xsite53 design: flags identical to the Torch cells
Burst: `--data $D/pretrain_data --tokenizer $D/olmo2_tok --tokenization whole --out $OUT/<tag> <arm> --arith_lines 4
--seed s --tag _<tag> --save_model --ckpt_every 2000`, arm none = `--target none` (tag `x53-7pct-none`), W =
`--target helix3 --schedule window --window 0.05,0.4 --site ctx` (tag `x53-7pct-win-ctx`); everything else default.
Parsed with the pinned trainer's own argparse and compared with the `args` saved in the Torch run JSONs (seed 0):
- vs pt999 7pct-none: 28 keys equal; differ only `data`, `tokenizer`, `out`, `tag`; new in a3bce3ab63c1 with inert
  defaults: `gate_start 0`, `gate_hyst 0`, `n_prefix 256`.
- vs ptrangeg whole-helix3-w0.05-0.4: 21 keys equal; differ `data`, `tokenizer`, `out`, `tag`; absent in the older
  code (defaults = the old behaviour for this cell): `site ctx`, `ckpt_every 2000` (saving touches no RNG),
  `stop_after -1`, `text_min 16`, `grad_ckpt auto` (off at 53M), `lm_chunk_tokens -1` (= 0 at bs 32), `untie False`,
  `gate_start / gate_hyst 0`, `n_prefix 256`.
The tags are unique so no summary merges sites; `tools/pt999_summary.py results/34[0-7]-*` reads the Burst cells as
`x53-7pct-none` / `x53-7pct-win-ctx` (W paired with none inside Burst). Code: a3bce3ab63c1 here vs 79e9b410dcc3 (pt999)
and eeb9208b2f0e (ptrangeg); the ctx path is unchanged between them (2f99b06 regression tests, identical step-0 lm,
identical `--dry_inputs` rows, verified below). ptgw main-W (same flags, same code a3bce3ab63c1) is the same-code Torch
replicate when it finishes; pt999b 7pct-win-ctx another.

Each task: data sha256 -> `--dry_inputs` (rows must hash to the Torch log's, PINNED.txt must show a3bce3ab63c1; exit 3
before any GPU time otherwise) -> training (`--ckpt_every 2000`; a retry after an outside kill keeps `$OUT` and resumes;
run_guard takes over the stale claim) -> `offline_lm.py` (full val.bin) -> P0 checks -> `p0_check.{json,txt}`, exit 4
on a failed check (results still come back). `gpu.csv`: nvidia-smi utilisation and memory every 60 s (a retry
appends). `node.txt`: GPU, driver, CPUs per attempt. `# time: 02:00:00`: planning 0.75 A100-h (2.5 A100-h per 1e9
tokens), pessimistic 1.0 h; a TIMEOUT is not retried by the tick, so the limit is 2x the pessimistic value. The guard
reserves 2 h per task, 16 h while all 8 are in flight.

## Pass rule (pre-registered; synthesis B1 = burst design P0 / P1)
**P0 exactness**, all required, otherwise stop and debug and use no Burst pretraining result:
1. 332: tokenizer files, token probe, parquet files and bins equal Torch's; the CPU `--dry_inputs` equal the Torch log.
2. per task: `--dry_inputs` digest = Torch's for the seed; `code_sha` a3bce3ab63c1; last eval at step 18309.
3. per seed: |step-0 lm - Torch| <= 1e-3 (Torch same-seed replicates: 0.0); |step-0 val_loss - Torch| <= 2e-3.
4. W: `prior_events == [[916, 1], [7324, 0]]` (0.05 x 18310 = 915.5, 0.4 x 18310 = 7324) and geo > 0 at every eval
   with the prior on; none: never on.
5. offline_lm on the full val: |orig_minus_logged| < 0.002.
(2-5 are in each task's `p0_check.json`.)

**P1 pooling**, paired by seed, Burst minus Torch. Revised 2026-10-16 after the adversarial review, before any result
(finding 2: at 7% W sits at its ceiling and "W - none > 0" cannot fail on a site effect, and 2% is the batch's main
setting):
- **P1a, 7%** (the 8 xsite53 tasks; an exactness check near the ceiling, not a test of the 2% regime): W area
  |mean diff| <= 1.5 and every |diff| <= 3 (tolerances from the Torch replicate spread, SD 0.90, max 1.38).
  "W - none > 0 in 4/4" is kept only as a sanity check (Torch gaps 29-49).
- **P1b, 2%** (required; no extra GPU time): the pt2b cells at lr 1e-3 (tasks 350-353, tags `pt2b-lr1e-3-none` and
  `pt2b-lr1e-3-win-ctx`, same pin) run the flags of Torch pt999 `2pct-none` / `2pct-win-ctx`: parsed with each code's
  own argparse (seed 0), 28 keys equal; only data, tokenizer, out and tag differ, plus `gate_start`, `gate_hyst`,
  `n_prefix` that the pt999 code lacks, at inert defaults. Same seeds, data and step-0 batch (pt2b asserts step-0 lm
  within 1e-3 of pt999 2pct-none). Tolerance: no same-flag 2% replicate exists on Torch, so the paired-difference SD is
  taken as 0.64 x the cell's 8-seed SD (0.64 = the 7% ratio of replicate-difference SD to seed SD, gate-ctx
  0.90 / 1.41), with the 7% multipliers (1.5 / 0.9 = 1.67 for the mean, 3 / 0.9 = 3.33 for each seed).
  - none: Burst minus Torch 2pct-none, seeds 0-3 (Torch 13.07 / 8.22 / 9.68 / 9.28, mean 10.06; 8-seed SD 1.78,
    so 1.14): |mean diff| <= 1.9 and every |diff| <= 3.8.
  - win-ctx: Burst minus Torch pt999 2pct-win-ctx (Batch 1 T2; its job files exist, not submitted on 2026-10-16),
    seeds 0-3, once T2 is in: |mean diff| <= max(1.5, 1.07 x SD) and every |diff| <= max(3, 2.13 x SD), SD = T2's
    8-seed area SD.
  - read from `results/35[0-3]-pt2b-lr1e-3-{none,win-ctx}-s*/check.json` (`info.area`, `info.lm0`).
- secondary: W b33_area |mean diff| <= 5 (replicate SD ~3.8); Burst mean W - none within Torch's 37.9 +- 12 (none's
  seed SD 8.4 makes only the effect testable, not none's level); offline bpb |mean diff| <= 0.006, descriptive (n = 4
  cannot resolve 0.004); final `cka_ops_helix3` |mean diff| <= 0.05 against ptgw main-W once it exists (ptrangeg's
  older code does not log it).
- Outcome. P0 + P1a + P1b-none pass: Burst cells may be pooled with Torch only as within-site paired differences (arm
  minus its own-site reference, site as a block); absolute levels are reported per site; the arms of one comparison
  never run on two sites. At 2% this stays provisional until P1b-win-ctx is in, and is withdrawn if it fails. P0 pass,
  P1 fail: Burst only for self-contained comparisons (pt2main is designed that way), and the site offset is reported.
  P0 fail: no Burst pretraining result is used until the cause is found.
- Rescaling: `p0_check.json` `projected_run_h` (median seconds per step between evals x 18310) gives the measured
  A100-hours per 53M 3e8-token run; the status `elapsed_min` counts only the last attempt (`run_task.sbatch` rewrites
  `start` on a retry), so after a retry it is low: use `projected_run_h`, or sacct's Elapsed summed over attempts.
  Every Burst 53M estimate scales by measured / 0.75 h; the 190M planning rate (2.0 A100-h per 3e8 tokens) by the same
  factor until a 190M run is timed; later `# time:` limits are 1.5-2x the measured time. If the median GPU utilisation in `gpu.csv` is below 50%, 53M
  Burst tasks run 2 jobs per GPU.

## Predictions (written before results)
- P0 passes; |step-0 lm - Torch| < 1e-4 (same init and first batch; only the GPU kernels differ).
- W area within +-1.5 of 88.86 on average; none 35-65 per seed; W - none positive in 4/4 (Torch: 29-49).
- 2% (P1b, added before results): Burst pt2b lr1e-3 none within +-1.9 of Torch's 10.06 on average (seeds 0-3).
- A100 40GB: 0.6-1.0 h per run (A100-80GB ran 53M digits in 0.64 h); GPU utilisation median 60-95%; peak memory
  20-28 GB (no LM-loss chunking at bs 32; the audit measured ~45 GB of logits work at bs 64).

## Verified on Torch before pushing (2026-10-16)
- `bash -n` on every new script: OK. Task headers parsed with the tick's own `hdr()` and `tl_hours()` (taken from
  `burst/tick.sh`): every `after:` exists; reserves 2 h per GPU task.
- `pretrain_data.py`: default path. Argument handling: one new flag `--revision` (default None); the "already built"
  check also compares the revision (`m.get`, so old metas behave as before); the download only runs for missing files,
  as before. Bytes: old (64abc19) vs new code, same args, 1e6 + 1e5 tokens: identical, and equal to the prefix of
  Torch's train.bin; boundary cases (train ends exactly at a document end; val = one token; val spanning a document):
  old == new. `--revision 87f0914...` on an empty dir downloaded both default parquet files with huggingface_hub (sha256
  = the LFS etags; metadata commit 87f09149) and gave the same bins. **Full rebuild** of `pretrain_data` with the new
  code from Torch's parquet files: train 4de2ed19..., val d6f744a9... = Torch (`done 402117333`, as the Torch log;
  18.9 min on 2 CPUs; RSS ~1.1 GB, flat). The 3B rebuild was not run to the end on 2 CPUs (stopped at 78M tokens;
  the code path is the same, and the 4e8 data is the 3B prefix).
- `pin_run.sh` on a temporary clone: help runs, PINNED line written (sha256 a3bce3ab63c1); branch name, unknown
  commit, missing script, `../` path, unset OUT -> exit 3; an edited or deleted file in the pinned tree -> exit 3;
  restored -> runs; script exit codes pass through (argparse error -> 2). 8 concurrent first calls on an empty
  RUNS_DIR: all succeed, one `pinned_64abc19`, no temp dirs, `diff -r` = `git archive 64abc19 tools`, no .pyc.
- Pinned trainer + xsite args on Torch's data and tokenizer: `--dry_inputs` lines identical to the Torch pt999
  7pct-none seed-0 log (9 lines) and, for W, to the ptrangeg seed-0 log (6 shared lines).
- `331-env311.sh` with a scratch LLMADD_ROOT and an empty fake HOME, PATH=/usr/bin:/bin: fresh build 2 min 10 s, 60
  packages = open_r1, torch 2.10.0+cu128 same wheel tag as open_r1, arch list includes sm_80; re-run 7 s (checks only);
  numpy changed to 2.0.2 -> exit 3 and the completion marker removed; nothing written to HOME (`uv python install
  --no-bin`; an earlier test without it had put a python3.11 link in ~/.local/bin, removed).
- `332-data.sh` end to end except the tokenisation itself (data dirs = links to Torch's bins and parquet files with a
  meta.json matching 332's request, so the builder answers "already built"; env311 from the 331 test; tokenizer
  downloaded from HF at f872bfe9): 7 tokenizer files OK, token probe identical to Torch, 2 + 5 parquet files and 4 bins
  OK, CPU `--dry_inputs` of the pinned trainer = the Torch log digest; rc 0. With a wrong 3B val.bin: `val.bin: FAILED`,
  the dir renamed `pretrain_data_3B.bad`, rc 3.
- `hash_ckpt.py`: Torch fp32 42B -> 4133fa61ec69ece6 (84 s), 210B -> 08e4c3a401069fd0 (109 s), both = the design's
  prototype; toy checkpoint: fp32 file and its fetch_olmo-style bf16 conversion (different sharding) give the same
  digest, one changed weight changes it, `--expect` 0 / 3. `333-hash-olmo.sh` on Torch's models: rc 0; with 210B
  missing: reported, rc 3.
- `330-probe.sh` on Torch with `sbatch` stubbed: completes, every failing command noted, rc 0.
- xsite task plumbing on Torch CPU (copies of 340 and 341; only the training call gets a tiny model and 12 steps
  appended, offline_lm 64 windows at bs 4; env311 from the 331 test; Torch data and tokenizer): data sha OK, dry-inputs
  digest = Torch's, PINNED = a3bce3ab63c1, training, offline_lm (|orig_minus_logged| 5e-8 / 9e-9), P0 written and exit 4.
  As intended, PASS: code_sha, none never on; FAIL by construction: last step (11 of 12), step-0 lm / val_loss (other
  model), full val (64 windows), W events ([[1, 1], [5, 0]] = the window rule at 12 steps), geo-while-on (no eval
  inside the 12-step window). A second attempt of each (a retry: same `$OUT`): run guard "already complete", offline_lm
  and P0 again, PINNED.txt appended (6 lines). A first attempt was OOM-killed in offline_lm at bs 32 on the 24 GB CPU
  host (fp32 logits of 32 x 512 x 100278); on the A100 40 GB this is ~17 GB with nothing else resident.

## Adversarial review (fresh subagent, 2026-10-16) and fixes
| # | finding | real? | done |
|---|---|---|---|
| 1 | blocking: 330 listed every user of a course account (`sacctmgr show assoc account=...`) into `probe.txt`, which is pushed to the public repo | yes (on Torch the same query returns 16 rows of other users for one account) | 330 keeps only the account rows and this user's (`awk` on the User field); `sshare -u $USER`, `sacct -u $USER` for the job list; the per-account hours count other users' jobs without naming them; a last `sed` redacts any other `User=` / `UserName=` value |
| 2 | important: P1 at 7% can barely fail (W at its ceiling; Torch W - none 29-49) and does not test 2% | yes | P1 split into P1a (7%, as before, "W - none > 0" demoted to a sanity check) and P1b (2%, required) on the pt2b lr 1e-3 cells, which run Torch pt999 2pct-none / 2pct-win-ctx flags (checked with both argparses); tolerances above. The 34x runs are unchanged |
| 3 | important: `burst/tick.sh` finalises a job as RUNNING when `squeue` fails once (sacct state not in the retry list) | yes, by reading (lines 43-62) | not this piece's file: for the tick owner (skip the task this tick when sacct reports a non-terminal state) |
| 4 | important: the plan's B0 gate (no GPU task before 330-333 COMPLETED) is not encoded | partly | the 34x tasks now wait for 330 too; 330 says per GPU account accepted / refused (3b) and always exits 0. Not encoded: 333 (LoRA checkpoints only, gating pretraining on it is not useful), the pt2main tasks (other piece), and "accepted but pending on a limit" (no read-only test catches it): a human check after the first GPU submission (open items) |
| 5 | minor: `pin_run.sh` checked the tree only after the rename, so a damaged `pinned_64abc19` failed every task for good | yes | the tree is checked in the temporary dir before the rename; a tree that fails the check is moved aside and extracted again (3 attempts) |
| 6 | minor: `# time: 02:00:00` is 2.7x the plan; `elapsed_min` counts only the last attempt | yes | time kept (the tick does not retry a TIMEOUT; cost = reservation only); rescaling text now uses `projected_run_h` / sacct |
| 7 | minor: the tick's per-task `git archive HEAD` is 202 MB / 6,410 files (54 s); 12 GPU submissions outlast the tick's `timeout 600` | yes (auditor measured) | not this piece's file: for the tick owner (`git archive HEAD burst tasks tools`) |
| 8 | minor: a kill between the last eval write and the model save leaves a complete run JSON and no model | yes (`pretrain_range.py` deletes the checkpoint, then saves the model) | not fixed: a window of about a second; P0 fails loudly (offline_lm missing, exit 4), the area numbers stay valid |
| 9 | minor: 333 did not check the other ft36b checkpoints | yes | 333 requires 9B 21B 84B 630B 3896B complete (exit 3 otherwise) |
| 10a | 331's `rm -rf` could delete env311 under running tasks if a pin change re-ran | yes (only via a new env task) | a completed env with other pins is never deleted (exit 3) |
| 10b | the tasks pass the 7-character pin `64abc19` | no change | with `^{commit}` git disambiguates among commits only, and an ambiguous prefix exits 3 (fails safe); the PINNED line records the full hash. The shared interface and the pt2main tasks use the same short form |
| 10c | 332 downloads parquet 000 / 001 twice (4.3 GB) | yes | hard links from the verified 4e8 files (a failed link falls back to the download) |

Verification of the fixes (Torch CPU, 2026-10-16):
- 330 on Torch with a fake config (two Torch accounts as GPU_ACCOUNTS; `sbatch` stubbed to refuse one): rc 0; the
  19 other users of the three accounts appear 0 times in `probe.txt` and the task log (`grep -w` per NetID); account rows
  and this user's rows kept; 3b reads "REFUSED" / "accepted" as stubbed. The redaction `sed` on a synthetic line keeps
  this user and empty values and replaces other names.
- `pin_run.sh` (scratch clone, `test_pin_repair.sh`): fresh call rc 0; an edited file, a deleted file, a missing
  COMMIT and an empty tree are each moved aside and re-extracted (rc 0, tree = `git archive 64abc19 tools`); a `tar` that
  silently extracts nothing -> rc 3 and nothing installed; 8 concurrent calls on a damaged tree -> all rc 0, one tree
  (= archive) + the moved-aside copy; 8 concurrent first calls -> all rc 0, one tree, no temp dirs, no .pyc; branch
  name, unknown commit, missing script, `../`, unset OUT -> rc 3; argparse error of the script -> rc 2.
- 333 on a fake models dir (42B / 210B = Torch's dirs, the 5 others = `.complete` + links to Torch's shards): rc 0, both
  digests OK, 5 x ok; with one 84B shard removed and the 630B marker removed: rc 3, both BAD, the rest ok.
- 331: a completed env with other pins -> rc 3, the env dir untouched, no uv dir created; the same-pins re-run ->
  "checking the versions only", 60 packages equal, rc 0, nothing in HOME. (The env copy used by the earlier plumbing
  tests still had the deliberate numpy 2.0.2 of the tamper test; it was rebuilt by 331 itself, and the 332 re-run below
  uses the restored env.)
- 332: the link step in isolation (hard links, same inode; a second run is a no-op; missing sources -> no error, the
  download path); `test_332.sh` end to end again on the restored env: run A rc 0 (7 tokenizer files, probe identical,
  2 + 5 parquet files and 4 bins OK, pinned CPU `--dry_inputs` = the Torch digest); run B (wrong 3B val.bin) rc 3, dir
  renamed `.bad`. (A first re-run failed at the dry-input step, `No module named 'numpy'`: 331's rebuild of that same
  env copy was running at that moment; the test was repeated after it.)
- 34x regenerated from the generator with `# after: 330-probe 331-env311 332-data`; `diff` against the previous files
  = that header line only. Headers parsed with the tick's `hdr()` / `tl_hours()`: every `after:` exists, 16 h reserved
  while the 8 GPU tasks are in flight.
- Mid-run resume, not tested before (`test_xsite3.sh`, regenerated 340 + new `pin_run.sh` + restored env, tiny CPU
  model): attempt 1 with `--ckpt_every 4 --stop_after 6` (an outside stop after the step-3 checkpoint) -> no model,
  offline_lm "skip", P0 exit 4; attempt 2 in the same OUT -> "resumed from ckpt... at step 4 of 12", model saved, and
  the run JSON's curve equals the uninterrupted run of the earlier test (same tiny arguments) in every field except
  wall-clock `sec`; args differ only in paths and `ckpt_every`. PINNED.txt 6 lines (3 calls per attempt).

## Audit status (experiment-audit, minimal mode + known answers)
i double path: data = rebuild bytes vs Torch sha; inputs = `--dry_inputs` vs Torch logs; offline_lm double path in
every task. ii asserts: sha / digest / PINNED checks exit 3 before training, P0 exit 4. iii known answers: Torch data
rebuilt byte-identically, Torch checkpoint digests reproduced, toy fp32 / bf16 digests, tokenizer probe. iv: not
applicable (no results yet). v: fresh adversarial subagent review done (2026-10-16; table above: 1 blocking, 3
important, 6 minor; fixed or assigned). vi: pitfalls 1 (same flags, same dry inputs), 7 (n per eval set printed by
`--dry_inputs`), 8 (default path of pretrain_data.py byte-identical; escaping checked by running every script) covered.
vii: `--dry_inputs` prints decoded training rows with the prior sites in ⟦ ⟧ and decoded eval rows in every task log
and `dry_inputs.txt`.

## Pitfalls and open items
- Code snapshot: a task runs the HEAD of the moment the tick submits it. The trainer and offline_lm are pinned
  (64abc19); the task scripts, `pin_run.sh` and the new `pretrain_data.py` come from the submit-time snapshot and are
  guarded by hashes (data) and the PINNED check (trainer).
- A retry re-runs the whole task script: `--dry_inputs` again, training resumes from the last checkpoint (<= 2000
  steps lost), offline_lm again (cheap). The tick retries CANCELLED / NODE_FAIL / PREEMPTED / BOOT_FAIL / LOST up to 3
  times, not TIMEOUT.
- `pretrain_data.py` trusts files already under `raw/`; 332 hashes them. meta.json holds the site path: never hash it.
- The P1 computation (a `tools/xsite_summary.py`) is not part of this piece. It needs P1a from
  `results/34[0-7]-*/p0_check.json` (`info.area`, `info.b33_area`, per seed) and P1b from
  `results/35[0-3]-pt2b-lr1e-3-{none,win-ctx}-s*/check.json` (`info.area`), against the Torch run JSONs of pt999
  2pct-none and, once it exists, 2pct-win-ctx (Batch 1 T2).
- 330 answers: whether `ds_ga_1006_001-2026fa` is valid on the GPU partitions and whether CPU time is billed (section
  3b; section 4 also says whether `sacct -A`, which the tick's budget guard uses, counts other users' jobs).
- B0 gate, human part: after the first GPU submission, read `heartbeat.json` `in_flight`. A job pending with an
  `Assoc*` / `QOS*` limit reason on ds_ga_1006 will not start: push config.env without 1006 in GPU_ACCOUNTS, then `scancel` those jobs on
  Burst by hand (not through cancel.list, which makes the cancel final): the tick then retries them on the next account.
  The pt2main tasks (350+) do not wait for 330 (their generator's choice).
- For the tick owner (`burst/tick.sh`, not this piece): (a) a failed `squeue` + sacct RUNNING / PENDING / REQUEUED /
  COMPLETING finalises a live job; skip the task this tick instead; if both fail, LOST makes the tick resubmit a job that
  may still run (same out/); (b) the per-task snapshot `git archive HEAD` (202 MB) makes 12 submissions outlast
  `timeout 600`; `git archive HEAD burst tasks tools` avoids it.
- Reaper: CPU tasks are short or busy (332 tokenises with all cores); a reaped task is retried and 332 is idempotent.

## 2026-10-16 amendment: the P1 read-out is code (before any xsite53 / pt2b result)
Written and tested before any result of 330-361 existed (no `results/` or `status/` entry for them, origin at 7672427).
- `tools/xsite_summary.py` applies the pass rule above exactly and prints PASS / FAIL / INCOMPLETE per rule, the outcome
  paragraph of this file, and exits 0 (P0, P1a, P1b-none and P1b-win-ctx pass), 1 (any rule fails) or 2 (incomplete /
  withheld, none fails). Area is `tools/readout_rules.py` (the one definition set of the Batch 1 read-outs, shared with
  `pt2main_summary.py` and `ptscale_summary.py`; = `pt999_summary.py`'s formula), and the P1b runs are chosen with its
  Burst re-run rule (an EXTRA re-run of a pt2b lr 1e-3 task counts as in pt2main).
- Choices fixed here (the rule text left them open): P0.2 also recomputes the `--dry_inputs` digest from the returned
  `dry_inputs.txt` (bytes, as `grep | head -n 9 | sha256sum`; = the Torch log digests) and compares the run args with the
  Torch cell's (only data / tokenizer / out / tag may differ; keys new in the pinned code must hold their inert
  defaults); the task's own `p0_check.json` must agree with the recomputation (a disagreement is a P0 failure). The Torch
  reference areas must equal the table above (exit 3 otherwise). P1b: a pt2b run whose `check.json` failed makes the
  rule WITHHELD (= incomplete until an EXTRA re-run), a step-0 lm more than 1e-3 off Torch's fails it; the T2 SD is the
  8-seed SD with ddof 1 (as the 1.78 of 2pct-none, recomputed: 1.785). "W - none > 0 in 4/4" is printed as SANITY, not
  part of the verdict. Hours: per run `projected_run_h` (recomputed and compared with `p0_check.json`), the last eval's
  `sec`, the status `elapsed_min`; factor = median projected_run_h / 0.75 h; each current limit (xsite53, pt2b,
  pt2long, pt2s190) is printed as a multiple of its rescaled time (pt2long = 4 x 53M, 190M = 2.0 h x factor); median GPU
  utilisation < 50% prints the 2-jobs-per-GPU rule.
- Re-runs (after the read-out's audit, same day, still before any result): a TIMEOUT is not retried by the tick, so its
  results come back with an unfinished run JSON. The xsite53 run of (arm, seed) is the first FINISHED run in task order
  among `results/<n>-xsite53-<arm>-s<seed>` (the 34x task, then any re-run under a new number); an unfinished or missing
  run keeps that seed INCOMPLETE (not a P0.2 failure) until a re-run finishes. **Choice for the user to confirm:** a
  re-run never replaces a finished run, so a finished run that fails P0 stays a P0 failure ("stop and debug", above);
  later finished runs are listed and ignored.
- Reference validity (same audit): P1b compares against Torch pt999 2pct-none and T2 2pct-win-ctx, so both must pass
  `readout_rules.check_2pct` (runs/ptlr/AUDIT.md 'Controls': code_sha 79e9b410dcc3, 18310 steps / 75 evals, the same
  seed's 2pct-none args except target / schedule, step-0 lm vs the 2% values within 1e-3, T2 prior_on_frac 6408 / 18310,
  2pct-none never on); an invalid reference makes the rule WITHHELD.
- Tests on fake results (scratch `readout/fake_xsite.py`: Burst runs = copies of the Torch reference runs with the pinned
  code's args, curves shifted by designed area deltas, `sec` x 0.8, the seed's Torch log as `dry_inputs.txt`; fake T2 =
  2pct-gate-ctx's 8 runs with window args), designed -> printed: pass without T2 -> P1a +0.35 (max 1.00), P1b-none
  +0.38, P1b-win-ctx INCOMPLETE, exit 2, factor 1.02; with T2 -> all PASS, tolerances 3.12 / 6.22, exit 0, util 45% ->
  2 jobs per GPU; W deltas +1.6..+1.8 -> P1a FAIL (mean +1.62), exit 1; one seed -3.2 -> P1a FAIL (max 3.20); step-0 lm
  +2e-3 -> P0.3 FAIL, outcome "P0 FAILS"; wrong W prior_events with a passing p0_check -> P0.4 and p0_check FAIL; one
  changed row in dry_inputs.txt -> P0.2 FAIL; P1b none deltas +1.8..+2.5 -> FAIL (mean +2.05); a failed pt2b check ->
  P1b-none WITHHELD, exit 2; 6 of 8 tasks (one TIMEOUT) -> INCOMPLETE, exit 2; the args of a real pinned-code run
  (ptgw lr5e-4-W s1, lr set to 1e-3) pass P0.2 against ptrangeg's older args, with its lr 5e-4 left in -> FAIL {'lr'}.
  After the audit (the fake T2 now carries the window's prior_on_frac; the earlier one had the gate's 0.68-1.0 and still
  passed, which is how the missing check was found): T2 seed 5 with the gate's prior_on_frac -> P1b-win-ctx WITHHELD,
  exit 2; a TIMEOUT that returns an unfinished run JSON -> INCOMPLETE (was P0.2 FAIL), exit 2; the same plus a finished
  re-run 348 -> all PASS, "superseded" note, exit 0; a P0-failing 343 plus a clean re-run 349 -> P0.3 FAIL, 349
  "ignored", exit 1; 332 FAILED rc 3 -> P0.1 FAIL; one offline_lm |orig_minus_logged| 0.01 -> P0.5 FAIL; Burst pt2b
  none s2 step-0 lm +2e-3 -> P1b-none FAIL (pairing broken); Burst win-ctx deltas +3.5..+4.5 -> P1b-win-ctx FAIL (mean
  +3.95 > 3.12), outcome "the 2% pooling is withdrawn", exit 1.
