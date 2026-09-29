#!/bin/bash
# Sync the repo and run one tick under an exclusive lock. Called every minute by the sentinel AND in the
# background of every running task job, so the pipeline keeps moving while any task runs even if the
# sentinel chain is broken (Burst reclaims the sentinel's CPU node every ~39 min).
ROOT=${LLMADD_ROOT:-/scratch/$USER}
REPO=$ROOT/llmadd
STATE=$ROOT/llmadd_state
mkdir -p "$STATE"
# Lock = a directory on the shared scratch (mkdir is atomic across nodes). flock on this NFS scratch left a
# stale lock on 2026-09-29 (a node went away while holding it) and every later tick exited silently for hours.
# A lock older than 15 min is broken; a tick itself is capped at 10 min, so a live holder is never older.
# TICK_WAIT=<seconds>: wait for the lock (the final tick of a finishing task must not be skipped)
LOCK="$STATE/tick.lockdir"
deadline=$(( $(date +%s) + ${TICK_WAIT:-0} ))
until mkdir "$LOCK" 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || date +%s) ))
  if [ "$age" -gt 900 ]; then
    echo "[$(date -u +%FT%TZ)] breaking stale tick lock ($age s old: $(cat "$LOCK/owner" 2>/dev/null))" >> "$REPO/logs/tick.log"
    rm -rf "$LOCK"; continue
  fi
  [ "$(date +%s)" -lt "$deadline" ] || exit 0
  sleep 5
done
echo "$(hostname -s) $$ ${SLURM_JOB_ID:-manual} $(date -u +%FT%TZ)" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT
cd "$REPO" || exit 0
timeout 120 git fetch -q origin 2>/dev/null && git reset -q --hard origin/main 2>/dev/null
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
cp burst/tick.sh "$STATE/tick_run.sh" && timeout 600 bash "$STATE/tick_run.sh"
