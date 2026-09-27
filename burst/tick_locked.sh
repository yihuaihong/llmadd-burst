#!/bin/bash
# Sync the repo and run one tick under an exclusive lock. Called every minute by the sentinel AND in the
# background of every running task job, so the pipeline keeps moving while any task runs even if the
# sentinel chain is broken (Burst reclaims the sentinel's CPU node every ~39 min).
ROOT=${LLMADD_ROOT:-/scratch/$USER}
REPO=$ROOT/llmadd
STATE=$ROOT/llmadd_state
mkdir -p "$STATE"
exec 9>"$STATE/tick.lock"
flock -n 9 || exit 0
cd "$REPO" || exit 0
git fetch -q origin 2>/dev/null && git reset -q --hard origin/main 2>/dev/null
# run a copy: the tick itself pulls, and must not rewrite the file bash is reading
cp burst/tick.sh "$STATE/tick_run.sh" && bash "$STATE/tick_run.sh"
