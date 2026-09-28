#!/bin/bash
# Runs ONE task inside its frozen code snapshot ($ROOT/llmadd_runs/<task>/code). Submitted by tick.sh,
# which passes TASK and LLMADD_ROOT. Writes start/end/rc next to the snapshot; tick.sh picks them up.
set -u
ROOT=${LLMADD_ROOT:-/scratch/$USER}
RUN=$ROOT/llmadd_runs/$TASK
cd "$RUN/code" || exit 1
export OUT=$RUN/out
export ENV_DIR=$ROOT/llmadd_env
export PY=$ENV_DIR/bin/python
export MODELS=$ROOT/models
export HF_HOME=$ROOT/hf_cache
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
rm -f "$RUN/rc"
date +%s > "$RUN/start"
echo "## task $TASK  job ${SLURM_JOB_ID:-none}  host $(hostname)  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
# keep the pipeline ticking while this task runs (lock-protected; a no-op if the sentinel holds the lock)
( while true; do bash "$ROOT/llmadd/burst/tick_locked.sh" >/dev/null 2>&1; sleep 90; done ) &
TICKER=$!
bash "tasks/$TASK.sh"
rc=$?
kill "$TICKER" 2>/dev/null
date +%s > "$RUN/end"
echo "$rc" > "$RUN/rc"
echo "## rc=$rc  $(date -u +%FT%TZ)"
# final tick: publish this task's results and submit whatever it unblocked (no sentinel needed)
TICK_WAIT=300 bash "$ROOT/llmadd/burst/tick_locked.sh" >/dev/null 2>&1
exit "$rc"
