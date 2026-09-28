#!/bin/bash
#SBATCH --job-name=llmadd_sentinel
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#
# The Burst sentinel: 1 CPU, no GPU. Every minute it syncs the repo to origin/main and runs one tick
# (burst/tick.sh): finish tasks that ended, submit tasks that are ready, push status/results/heartbeat.
# It queues its own successor (--dependency=afterany) so the chain survives the 48 h wall-time limit.
# Submitted once by burst/boot.sh. Stop for good: scancel -n llmadd_sentinel
set -u
# Under Slurm $0 is a spooled copy of this file, so every path is absolute.
ROOT=${LLMADD_ROOT:-/scratch/$USER}
REPO=$ROOT/llmadd
STATE=$ROOT/llmadd_state
cd "$REPO" || { echo "repo not found at $REPO"; exit 1; }
mkdir -p logs "$STATE"
set -a; . burst/config.env; set +a
. "$REPO/burst/lib.sh"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
ME=${SLURM_JOB_ID:-manual}

# Record how the previous link of the chain ended, so a dead chain can be diagnosed from git.
prev=$(cat "$STATE/sentinel_job" 2>/dev/null || true)
if [ -n "$prev" ] && [ "$prev" != "$ME" ]; then
  sacct -n -X -j "$prev" -o JobID,State,ExitCode,Elapsed,NodeList > "$STATE/prev_sentinel.txt" 2>&1 || true
fi
echo "$ME" > "$STATE/sentinel_job"
date +%s > "$STATE/sentinel_start"

# Exactly one pending successor. A requeued run of this job finds it already queued and skips.
if squeue --me -h -n llmadd_sentinel -t PENDING 2>/dev/null | grep -q .; then
  log "successor already pending"
else
  succ=$(sbatch_first "$CPU_PARTITION" --parsable --account="$ACCOUNT" --cpus-per-task=1 --time=2-00:00:00 \
         --requeue --job-name=llmadd_sentinel --output="$REPO/logs/sentinel_%j.out" \
         --dependency=afterany:"$ME" "$REPO/burst/sentinel.sbatch" 2>&1)
  log "successor: $succ"; echo "$succ" > "$STATE/sentinel_successor"
fi
log "sentinel $ME on $(hostname) started"

while true; do
  bash "$REPO/burst/tick_locked.sh"
  sleep "${TICK_SECONDS:-60}"
done
