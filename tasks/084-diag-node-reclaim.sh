# kind: cpu
# time: 00:15:00
# cpus: 1
# Why do n2c48m24 jobs end CANCELLED after ~38.7 min (sentinels 2335, 2348) or before they start (2358)?
# Who cancelled them (sacct shows "CANCELLED by <uid>"), what Slurm/GCP said about the node, and which
# suspend / idle / preempt settings the cluster runs with. Read-only.
set -u
{
echo "## now $(date -u +%FT%TZ) on $(hostname) job ${SLURM_JOB_ID:-?} partition ${SLURM_JOB_PARTITION:-?}"
echo "## my llmadd jobs, last 3 days (full state, reason, times, node)"
sacct -X -S now-3days -u "$USER" --name=llmadd_sentinel,llmadd_080-dl-early2,llmadd_080b-dl-early2 \
  -o JobID%10,JobName%24,Partition%16,State%40,Reason%30,Submit,Start,End,Elapsed,NodeList%16 2>&1 | head -60
echo; echo "## all my jobs, last 2 days, any name"
sacct -X -S now-2days -u "$USER" -o JobID%10,JobName%28,Partition%16,State%40,Elapsed,NodeList%16 2>&1 | tail -40
echo; echo "## events for the reclaimed jobs (all steps, incl. requeues)"
for j in 2335 2348 2358; do sacct -j $j -D -o JobID%14,State%40,Reason%30,Start,End,Elapsed,NodeList%14 2>&1; done
echo; echo "## node reasons (down/drained)"
sinfo -R -o "%30E %20H %N" 2>&1 | head -30
echo; echo "## the nodes the sentinel used"
for n in b-10-1 b-10-2 b-19-21; do scontrol show node $n 2>&1 | grep -E "NodeName|State|Reason|BootTime|LastBusyTime|Partitions|Comment" ; done
echo; echo "## partition n2c48m24"
scontrol show partition n2c48m24 2>&1
echo; echo "## cluster power-saving / preemption config"
scontrol show config 2>&1 | grep -i -E "suspend|resume|idle|preempt|reboot|kill|timeout|batchstart|prolog|epilog|job_submit|cloud" | head -40
} | tee "$OUT/diag.txt"
