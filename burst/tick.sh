#!/bin/bash
# ONE tick of the sentinel (the caller has just synced the repo to origin/main):
#   1. tasks whose Slurm job ended  -> status/<task>.json + results/<task>/ (small files only)
#   2. tasks that are ready         -> sbatch on the CPU or GPU partition named in their header
#   3. anything changed, or 10 min  -> commit + push status/, results/, heartbeat.json
# Local bookkeeping lives outside the repo in $STATE: <task>.job (in flight), <task>.fin (ended),
# <task>.pushed (its results reached GitHub). A task never runs twice; to re-run, add a new task file.
set -u
ROOT=${LLMADD_ROOT:-/scratch/$USER}
REPO=$ROOT/llmadd
STATE=$ROOT/llmadd_state
RUNS=$ROOT/llmadd_runs
cd "$REPO" || exit 0
mkdir -p "$STATE" "$RUNS" logs status results
ENABLED=0; MAX_GPU_JOBS=1; GPU_HOURS_BUDGET=0
set -a; . burst/config.env; set +a
. "$REPO/burst/lib.sh"
log() { echo "[$(date -u +%FT%TZ)] $*" >> logs/tick.log; }
hdr() { sed -n "s/^# $2: *//p" "$1" | head -1; }
in_queue() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
changed=0

# 0. cancellations requested through git: one task name per line in burst/cancel.list.
# The job is scancelled once; the task then ends as CANCELLED like any other (re-run = new task file).
if [ -f burst/cancel.list ]; then
  while read -r t _; do
    [ -n "$t" ] && [ "${t#\#}" = "$t" ] || continue
    if [ -e "$STATE/$t.job" ] && [ ! -e "$STATE/$t.cancel_sent" ]; then
      read -r jid _ < "$STATE/$t.job"
      scancel "$jid" 2>/dev/null && log "cancelled $t ($jid) on request"
      touch "$STATE/$t.cancel_sent"; changed=1
    fi
  done < burst/cancel.list
fi

# 1. ended jobs
for jf in "$STATE"/*.job; do
  [ -e "$jf" ] || continue
  t=$(basename "$jf" .job)
  read -r jid kind < "$jf"
  if [ -f "$RUNS/$t/rc" ]; then
    rc=$(cat "$RUNS/$t/rc"); [ "$rc" = 0 ] && st=COMPLETED || st=FAILED
  elif in_queue "$jid"; then
    continue
  else
    rc=""; st=$(sacct -n -X -j "$jid" -o State 2>/dev/null | head -1 | awk '{print $1}'); st=${st:-LOST}
  fi
  printf '%s %s\n%s %s\n' "$jid" "$kind" "$st" "${rc:-none}" > "$STATE/$t.fin"
  rm -f "$jf"
  log "$t ($jid) ended: $st rc=${rc:-none}"
done

# write status/results for every ended task whose results have not reached GitHub yet
to_push=()
for ff in "$STATE"/*.fin; do
  [ -e "$ff" ] || continue
  t=$(basename "$ff" .fin)
  [ -e "$STATE/$t.pushed" ] && continue
  { read -r jid kind; read -r st rc; } < "$ff"
  rm -rf "results/$t"; mkdir -p "results/$t"
  if [ -d "$RUNS/$t/out" ]; then
    (cd "$RUNS/$t/out" && find . -type f -size -5M \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' \
        -o -name '*.png' -o -name '*.md' -o -name '*.txt' \) -print0 | xargs -0 -r cp --parents -t "$REPO/results/$t/")
  fi
  tail -n 200 "$RUNS/$t/slurm.out" > "results/$t/log_tail.txt" 2>/dev/null || true
  s=$(cat "$RUNS/$t/start" 2>/dev/null || echo 0); e=$(cat "$RUNS/$t/end" 2>/dev/null || date +%s)
  [ "$s" = 0 ] && mins=0 || mins=$(( (e - s) / 60 ))
  printf '{"task": "%s", "job": "%s", "kind": "%s", "state": "%s", "rc": "%s", "elapsed_min": %s, "ended": "%s"}\n' \
    "$t" "$jid" "$kind" "$st" "$rc" "$mins" "$(date -u -d "@$e" +%FT%TZ)" > "status/$t.json"
  to_push+=("$t")
done

# GPU hours already spent by finished GPU tasks (the budget guard reads git, so it survives scratch loss)
gpu_min=$(cat status/*.json 2>/dev/null | awk -F'"elapsed_min": ' '/"kind": "gpu"/ {split($2,a,","); s+=a[1]} END {print s+0}')
gpu_h=$(( gpu_min / 60 ))

# 2. submit ready tasks
if [ "${ENABLED:-0}" = 1 ]; then
  ngpu=$(cat "$STATE"/*.job 2>/dev/null | grep -c ' gpu$' || true)
  for s in $(ls tasks/*.sh 2>/dev/null | sort); do
    t=$(basename "$s" .sh)
    if [ -e "status/$t.json" ] || [ -e "$STATE/$t.job" ] || [ -e "$STATE/$t.fin" ]; then continue; fi
    if [ -e "$STATE/$t.err" ] && [ $(( $(date +%s) - $(stat -c %Y "$STATE/$t.err") )) -lt 600 ]; then continue; fi
    ready=1
    for a in $(hdr "$s" after); do
      grep -q '"state": "COMPLETED"' "status/$a.json" 2>/dev/null || ready=0
    done
    for a in $(hdr "$s" after_ended); do   # ended in any state (e.g. wait for a failed twin to clear)
      [ -e "status/$a.json" ] || ready=0
    done
    [ "$ready" = 1 ] || continue
    kind=$(hdr "$s" kind); kind=${kind:-cpu}
    tl=$(hdr "$s" time); tl=${tl:-04:00:00}
    cpus=$(hdr "$s" cpus); cpus=${cpus:-4}
    if [ "$kind" = gpu ]; then
      [ "$ngpu" -lt "$MAX_GPU_JOBS" ] || continue
      [ "$gpu_h" -lt "$GPU_HOURS_BUDGET" ] || { log "GPU budget reached (${gpu_h}h); $t waits"; continue; }
      ng=$(hdr "$s" gpus); ng=${ng:-1}
      if [ "$ng" -gt 1 ]; then part=${GPU2_PARTITION:-c24m170-a100-2}; else part=$GPU_PARTITION; fi
      extra=(--gres=gpu:$ng)
    else
      part=$CPU_PARTITION; extra=()
    fi
    rm -rf "${RUNS:?}/$t"; mkdir -p "$RUNS/$t/code" "$RUNS/$t/out"
    git archive HEAD | tar -x -C "$RUNS/$t/code"
    out=$(sbatch_first "$part" --parsable --account="$ACCOUNT" ${extra[@]+"${extra[@]}"} --cpus-per-task="$cpus" \
            --time="$tl" --requeue --job-name="llmadd_$t" --output="$RUNS/$t/slurm.out" \
            --export=ALL,TASK="$t",LLMADD_ROOT="$ROOT" "$RUNS/$t/code/burst/run_task.sbatch" 2>&1)
    if [[ "$out" =~ ^[0-9]+ ]]; then
      echo "${out%%;*} $kind" > "$STATE/$t.job"; rm -f "$STATE/$t.err"
      log "submitted $t as ${out%%;*} ($kind on $part, $tl)"; changed=1
      [ "$kind" = gpu ] && ngpu=$((ngpu + 1))
    else
      echo "$out" > "$STATE/$t.err"; log "sbatch failed for $t: $out"; changed=1
    fi
  done
fi

# 3. heartbeat + push
last=$(stat -c %Y "$STATE/hb_pushed" 2>/dev/null || echo 0)
if [ "${#to_push[@]}" -gt 0 ] || [ "$changed" = 1 ] || [ $(( $(date +%s) - last )) -ge 600 ]; then
  inflight=""
  for jf in "$STATE"/*.job; do
    [ -e "$jf" ] || continue
    read -r jid kind < "$jf"
    qs=$(squeue -h -j "$jid" -o "%T %r" 2>/dev/null | head -1 | tr -d '"\\')
    inflight="$inflight\"$(basename "$jf" .job)\": \"$jid $kind ${qs:-?}\", "
  done
  errs=""
  for ef in "$STATE"/*.err; do
    [ -e "$ef" ] || continue
    errs="$errs\"$(basename "$ef" .err)\": \"$(tail -1 "$ef" | tr -d '"\\' | cut -c1-200)\", "
  done
  prev=$( [ -f "$STATE/prev_sentinel.txt" ] && tr -s ' \n' ' ' < "$STATE/prev_sentinel.txt" | tr -d '"\\' | cut -c1-200)
  printf '{"time": "%s", "host": "%s", "sentinel_job": "%s", "sentinel_started": "%s", "enabled": "%s", "gpu_hours_used": %s, "in_flight": {%s}, "submit_errors": {%s}, "prev_sentinel": "%s"}\n' \
    "$(date -u +%FT%TZ)" "$(hostname -s)" "$(cat "$STATE/sentinel_job" 2>/dev/null)" \
    "$(date -u -d "@$(cat "$STATE/sentinel_start" 2>/dev/null || echo 0)" +%FT%TZ)" "$ENABLED" "$gpu_h" \
    "${inflight%, }" "${errs%, }" "$prev" > heartbeat.json
  git add -A status results heartbeat.json
  if git commit -q -m "sentinel: $(date -u +%FT%TZ) ${to_push[*]:-heartbeat}" 2>/dev/null; then
    if git pull -q --rebase origin main 2>>logs/tick.log && git push -q origin HEAD:main 2>>logs/tick.log; then
      touch "$STATE/hb_pushed"
      for t in ${to_push[@]+"${to_push[@]}"}; do touch "$STATE/$t.pushed"; done
      log "pushed ${to_push[*]:-heartbeat}"
    else
      log "push failed; will retry next tick"
      git rebase --abort 2>/dev/null; git reset -q --hard origin/main
    fi
  fi
fi
exit 0
