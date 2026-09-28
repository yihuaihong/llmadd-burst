# Sourced by boot.sh, sentinel.sbatch and tick.sh.
# Burst's submit filter validates the account against ONE partition: a Slurm partition list such as
# "n2c48m24,interactive" is refused with "please validate account ... or partition 'n2c48m24,interactive'"
# (2026-09-28, which killed the sentinel restart and task 080b). sbatch_first tries the partitions of a
# comma-separated list one at a time, in order, and prints the first accepted submission's output
# (exit 0), or the last refusal (exit 1).
#     sbatch_first "<p1,p2,...>" <other sbatch args...>
sbatch_first() {
  local parts=$1 out="" p
  shift
  local -a list
  IFS=',' read -ra list <<< "$parts"
  for p in "${list[@]}"; do
    [ -n "$p" ] || continue
    if out=$(sbatch --partition="$p" "$@" 2>&1) && [[ "$out" =~ ^[0-9]+ ]]; then
      echo "$out"; return 0
    fi
  done
  echo "$out"; return 1
}
