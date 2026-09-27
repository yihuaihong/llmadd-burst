# llmadd-burst

Runs LLM_addition extension experiments on NYU HPC Burst, driven entirely through git. Only our own code
lives here; the collaborator's repository is not copied in.

## How it works

A sentinel job on Burst (`llmadd_sentinel`: 1 CPU, no GPU, renews itself every 48 h) syncs this repo every
minute and runs `burst/tick.sh`:

- every `tasks/NNN-name.sh` without a `status/NNN-name.json` whose `after:` tasks completed is submitted
  once, from a frozen snapshot of the repo, to the CPU or GPU partition named in its header;
- when the job ends, `status/<task>.json` and the small files it wrote to `$OUT` (json/csv/png/md/txt,
  < 5 MB each) are pushed to `results/<task>/`, with the last 200 log lines;
- `heartbeat.json` is pushed at least every 10 minutes.

Task header (comment lines at the top of the script):

```
# kind: cpu | gpu        (gpu = 1 x A100 40GB on GPU_PARTITION)
# time: HH:MM:SS         (Slurm limit)
# cpus: N
# gpus: 2                (optional; 2 x A100 on GPU2_PARTITION)
# after: 001-env 002-x   (space separated; waits until those completed)
# after_ended: 006-x     (waits until those ended, in any state)
```

Tasks see `$OUT` (write results here), `$PY` (the env's python), `$MODELS`, `$HF_HOME`. A task runs once;
to re-run, add a new task file. `burst/config.env` holds ENABLED, partitions, the account, the GPU budget.

## One-time setup on Burst (the only manual step)

```bash
test -f ~/.ssh/id_ed25519_llmadd || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519_llmadd -C burst-llmadd; cat ~/.ssh/id_ed25519_llmadd.pub
# the public key is added to this repo as a deploy key WITH write access, then:
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_llmadd -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  git clone -q git@github.com:yihuaihong/llmadd-burst.git /scratch/$USER/llmadd && bash /scratch/$USER/llmadd/burst/boot.sh
```

Cancel one task from anywhere: add its name to `burst/cancel.list` and push.

Stop everything: `scancel -n llmadd_sentinel` (running tasks: `scancel -n llmadd_<task>`), or set
`ENABLED=0` in `burst/config.env` to stop new submissions.
