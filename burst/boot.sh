#!/bin/bash
# One-time bootstrap on Burst, run from any Burst terminal after the repo is cloned to /scratch/$USER/llmadd:
#     bash /scratch/$USER/llmadd/burst/boot.sh
# It checks that this checkout can push (results come back through git), then submits the sentinel:
# a 1-CPU, no-GPU job that renews itself every 48 h. After that nobody needs a Burst terminal.
# Stop everything for good: scancel -n llmadd_sentinel
set -u
ROOT=${LLMADD_ROOT:-/scratch/$USER}
REPO=$ROOT/llmadd
cd "$REPO" || { echo "## clone the repo to $REPO first"; exit 1; }
KEY=${LLMADD_KEY:-$HOME/.ssh/id_ed25519_llmadd}
git config core.sshCommand "ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git config user.name "sentinel[burst]"
git config user.email "sentinel@burst.local"
git fetch -q origin && git reset -q --hard origin/main
mkdir -p logs "$ROOT/llmadd_state" "$ROOT/llmadd_runs"
set -a; . burst/config.env; set +a
echo "## push access:"
if out=$(git push --dry-run origin HEAD:main 2>&1); then echo "   ok"; else
  echo "   FAILED: $(echo "$out" | tail -1)"; echo "## the key needs write access to the repo; tell Claude."; exit 1; fi
echo "## scratch: $(df -h "$ROOT" | tail -1)"
if squeue --me -h -n llmadd_sentinel 2>/dev/null | grep -q .; then
  if [ "${RESTART:-0}" = 1 ]; then scancel -n llmadd_sentinel && echo "## cancelled the old sentinel chain (RESTART=1)"; sleep 3
  else echo "## a sentinel is already queued/running (RESTART=1 bash burst/boot.sh replaces it):"; squeue --me -n llmadd_sentinel; exit 0; fi
fi
jid=$(sbatch --parsable --account="$ACCOUNT" --partition="$CPU_PARTITION" --cpus-per-task=1 --time=2-00:00:00 \
      --requeue --job-name=llmadd_sentinel --output="$REPO/logs/sentinel_%j.out" "$REPO/burst/sentinel.sbatch" 2>&1)
if [[ "$jid" =~ ^[0-9]+ ]]; then
  echo "## sentinel submitted as job ${jid%%;*}. Everything else goes through git; you can close this terminal."
else
  echo "## sbatch failed: $jid"; exit 1
fi
