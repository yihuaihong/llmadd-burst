#!/bin/bash
# Sync the repo and run one tick under an exclusive lock. Called every minute by the sentinel AND in the
# background of every running task job, so the pipeline keeps moving while any task runs even if the
# sentinel chain is broken (Burst reclaims the sentinel's CPU node every ~39 min).
ROOT=${LLMADD_ROOT:-/scratch/$USER}
REPO=$ROOT/llmadd
STATE=$ROOT/llmadd_state
mkdir -p "$STATE"
exec 9>"$STATE/tick.lock"
# TICK_WAIT=<seconds>: wait for the lock (the final tick of a finishing task must not be skipped)
if [ -n "${TICK_WAIT:-}" ]; then flock -w "$TICK_WAIT" 9 || exit 0; else flock -n 9 || exit 0; fi
cd "$REPO" || exit 0
git fetch -q origin 2>/dev/null && git reset -q --hard origin/main 2>/dev/null
# SENTINEL=off in burst/config.env: a running sentinel chain retires itself (root cancels idle jobs after
# ~38 min anyway, and a 1-CPU sentinel holds a whole exclusive n2c48m24 node of the course allotment).
# Tasks keep the pipeline moving: each ticks while it runs and once more when it ends.
SENTINEL=on; . burst/config.env
if [ "$SENTINEL" = off ] && [ "${SLURM_JOB_NAME:-}" = llmadd_sentinel ]; then
  echo "[$(date -u +%FT%TZ)] SENTINEL=off: retiring the sentinel chain" >> logs/tick.log
  scancel -n llmadd_sentinel
  exit 0
fi
# run a copy: the tick itself pulls, and must not rewrite the file bash is reading
cp burst/tick.sh "$STATE/tick_run.sh" && bash "$STATE/tick_run.sh"
