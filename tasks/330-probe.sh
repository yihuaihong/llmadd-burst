# kind: cpu
# time: 00:20:00
# cpus: 1
# Account / partition / limit probe before the 2026-10-16 pretraining batch (Batch 1 B0; intent spec
# report/audits/burst_infra_AUDIT.md). Read-only: Slurm associations, QOS, fairshare and GPU-hour limits (with usage) of
# every account in GPU_ACCOUNTS and ACCOUNT; 'sbatch --test-only' (validates, submits nothing) for each account x
# partition; the GPU- and CPU-hours each account has used since 2026-08-01; node sizes of the three partitions; python,
# glibc and CUDA-driver facts. Everything goes to $OUT/probe.txt; a failing command is noted and the probe goes on.
# probe.txt is pushed to the PUBLIC repo: only the accounts' own rows and this user's rows are queried or kept (a course
# account lists every student's NetID), and any other User= / UserName= value left is redacted before the file is
# written out. Section 3b says which GPU account the tick will use first. Always exits 0 (the xsite53 tasks wait for it).
set -u
. burst/config.env
declare -A GPU_OK
ACCTS=$( { for ab in $GPU_ACCOUNTS; do echo "${ab%%:*}"; done; echo "$ACCOUNT"; } | awk '!seen[$0]++')
run() { echo "\$ $*"; timeout 60 "$@" 2>&1; local rc=$?; [ "$rc" = 0 ] || echo "(rc=$rc)"; echo; LAST_RC=$rc; }
own() { awk -F'|' -v u="$USER" 'NF < 2 || $2 == "" || $2 == u'; }   # '|' rows with field 2 (User) empty or this user
{
  echo "## probe $(date -u +%FT%TZ) on $(hostname -s)  accounts: $(echo $ACCTS)"
  echo "## GPU_ACCOUNTS=\"$GPU_ACCOUNTS\" ACCOUNT=$ACCOUNT CPU=$CPU_PARTITION GPU=$GPU_PARTITION GPU2=$GPU2_PARTITION"; echo
  echo "## 1. associations, QOS, limits"
  run sacctmgr -nP show assoc user="$USER" format=Cluster,Account,User,Partition,QOS,DefaultQOS,GrpTRESMins,GrpTRESRunMins,GrpTRES,MaxTRES,MaxTRESMins,MaxWall,MaxJobs,MaxSubmit,Fairshare
  qos=$(sacctmgr -nP show assoc user="$USER" format=QOS 2>/dev/null | tr ',' '\n' | sort -u | paste -sd, -)
  [ -n "$qos" ] && run sacctmgr -nP show qos "$qos" format=Name,Priority,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU,GrpTRES,GrpTRESMins,MaxTRESMins,Flags
  for a in $ACCTS; do
    run scontrol show assoc_mgr users="$USER" accounts="$a" flags=assoc
    echo "\$ sacctmgr -nP show assoc account=$a format=Account,User,Partition,QOS,GrpTRESMins,GrpTRES,MaxTRESMins (account rows and this user's only)"
    timeout 60 sacctmgr -nP show assoc account="$a" format=Account,User,Partition,QOS,GrpTRESMins,GrpTRES,MaxTRESMins 2>&1 | own; echo
  done
  echo "## 2. fairshare"
  echo "\$ sshare -P -l -U -u $USER -A $(echo $ACCTS | tr ' ' ,) (account rows and this user's only)"
  timeout 60 sshare -P -l -U -u "$USER" -A "$(echo $ACCTS | tr ' ' ,)" 2>&1 | own; echo
  echo "## 3. sbatch --test-only (nothing is submitted)"
  for a in $ACCTS; do
    for spec in "$CPU_PARTITION:" "$GPU_PARTITION:gpu:1" "$GPU2_PARTITION:gpu:2"; do
      p=${spec%%:*}; g=${spec#*:}
      run sbatch --test-only --account="$a" --partition="$p" ${g:+--gres=$g} --cpus-per-task=1 --time=00:10:00 --wrap true
      [ "$p:$g" = "$GPU_PARTITION:gpu:1" ] && GPU_OK[$a]=$LAST_RC
    done
  done
  echo "## 3b. GPU accounts on $GPU_PARTITION (1 GPU) in GPU_ACCOUNTS order = the order the tick tries them"
  for ab in $GPU_ACCOUNTS; do
    a=${ab%%:*}; [ "${GPU_OK[$a]:-1}" = 0 ] && echo "  $a: accepted (sbatch --test-only rc 0)" || echo "  $a: REFUSED (rc ${GPU_OK[$a]:-?}, see 3.)"
  done
  echo "  An accepted account can still hold jobs pending on an association / QOS limit: after the first GPU submission,"
  echo "  heartbeat.json in_flight shows each job's state and reason (e.g. PENDING AssocGrpGRESMinutes)."; echo
  echo "## 4. hours used since 2026-08-01 per account: every job 'sacct -A' returns, as the tick's GPU budget guard counts"
  echo "##    them (queued/running jobs at their time limit); jobs of other users are counted, never named"
  for a in $ACCTS; do
    timeout 60 sacct -n -X -P -A "$a" -S 2026-08-01 -E now -o State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES,NCPUS,User 2>/dev/null |
      awk -F'|' -v a="$a" -v u="$USER" '{ tres = ($4 != "" ? $4 : $5); g = 0
                 if (match(tres, /gres\/gpu=[0-9]+/)) g = substr(tres, RSTART + 9, RLENGTH - 9) + 0
                 live = ($1 ~ /^(PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING)/)
                 t = live ? $3 * 60 : $2
                 n++; if (g > 0) { gs += t * g; ng++ } else { cs += t * $6; nc++ }
                 if ($7 != u) { no++; if (g > 0) gso += t * g; if (!($7 in seen)) { seen[$7]; nu++ } } }
           END { printf "%s: %d jobs; GPU jobs %d, %.1f GPU-h; CPU-only jobs %d, %.1f CPU-h (allocated CPUs x time); of these, %d jobs (%.1f GPU-h) of %d other users\n",
                        a, n, ng, gs / 3600, nc, cs / 3600, no, gso / 3600, nu }'
    run sacct -X -u "$USER" -A "$a" -S 2026-10-01 -E now -o JobID,JobName%28,Partition%16,State,Elapsed,Timelimit,AllocTRES%60
  done
  echo "## 5. partitions and nodes"
  for p in "$CPU_PARTITION" "$GPU_PARTITION" "$GPU2_PARTITION"; do
    run scontrol show partition "$p"
    run sinfo -p "$p" -h -o "%P cpus=%c mem_MB=%m gres=%G nodes=%D state=%t timelimit=%l"
  done
  echo "## 6. this node, python, CUDA driver"
  run nproc
  run free -g
  run ldd --version
  run python3 --version
  ls -d /scratch/"$USER"/llmadd_env* 2>&1
  [ -x "/scratch/$USER/llmadd_env311/bin/python" ] && run "/scratch/$USER/llmadd_env311/bin/python" --version
  run nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  ls /usr/lib64/libcuda.so* /usr/lib/x86_64-linux-gnu/libcuda.so* 2>&1 | head -3
  echo "last driver seen by a GPU task (results/*/log_tail.txt):"
  grep -h -A1 "name, memory.total \[MiB\], driver_version" results/*/log_tail.txt 2>/dev/null | grep -v "^name\|^--" | sort | uniq -c | tail -3
  command -v module >/dev/null 2>&1 && { module -t avail python cuda 2>&1 | head -20; }
  echo "## 7. scratch"
  run df -h "/scratch/$USER"
} > "$OUT/probe.raw" 2>&1
# safety net for the public repo: any User= / UserName= value other than this user's -> <other-user>
sed -E -e "s/(User(Name)?=)$USER([^A-Za-z0-9_.-]|\$)/\1@@SELF@@\3/g" -e 's/(User(Name)?=)[A-Za-z0-9_.-]+/\1<other-user>/g' \
  -e "s/@@SELF@@/$USER/g" "$OUT/probe.raw" > "$OUT/probe.txt"
rm -f "$OUT/probe.raw"   # .raw is not a pushed suffix; removed anyway
cat "$OUT/probe.txt"
exit 0
