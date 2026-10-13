#!/bin/bash
# Report (and exit on) the first Torch job group that has fully left the queue since this waiter started.
# Groups = job-name prefixes after "llmadd_"; pass them as arguments, e.g.
#   bash tools/wait_groups.sh gate-add ge- pt-win pt-helix_acc pt-helix_cka pt7- ptdiag s1-curve reg- toy- dm-mixed
groups=("$@")
while true; do
  q=$(squeue -u "$USER" -h -o "%j" 2>/dev/null) || { sleep 120; continue; }
  for g in "${groups[@]}"; do
    if ! echo "$q" | grep -q "^llmadd_$g"; then
      echo "group finished (no job left in the queue): $g"
      sacct -u "$USER" -S now-2days -X -n --format=JobName%40,State,Elapsed | grep "llmadd_$g" | sort | uniq | tail -24
      exit 0
    fi
  done
  sleep 600
done
