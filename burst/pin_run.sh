#!/bin/bash
# Run a tools/ script from a PINNED commit instead of the task's own code snapshot. A task runs the repo HEAD of the
# moment the tick SUBMITS it, so a task submitted after a later push would silently run newer code (2026-10-16, Burst
# pretraining batch: the trainer is pinned so every cell of a comparison runs the same bytes):
#     bash burst/pin_run.sh <commit> <script under tools/> [args...]
# - tools/ at <commit> is extracted once (git archive from the clone at $LLMADD_ROOT/llmadd) into
#   $RUNS_DIR/pinned_<commit> (default RUNS_DIR = $LLMADD_ROOT/llmadd_runs): built in a temporary dir, checked there and
#   renamed into place, so concurrent tasks never see half a tree; a task that loses the race uses the winner's tree;
# - every call checks that the tracked files of the pinned tree are the git blobs of <commit>; a tree that is not (a
#   scratch purge, an edit) is moved aside to pinned_<commit>.bad.<time>.<pid> and extracted again (exit 3 only if
#   that fails 3 times);
# - one line per call is appended to $OUT/PINNED.txt: "<full commit> <sha256 of the script> tools/<script> <UTC time>";
# - the script runs with the env311 python ($LLMADD_ROOT/llmadd_env311) from <pinned>/tools, so args must use absolute
#   paths. No .pyc files are written into the pinned tree. The exit code is the script's (3 = the pin failed).
set -u
ROOT=${LLMADD_ROOT:-/scratch/$USER}
REPO=$ROOT/llmadd
RUNS_DIR=${RUNS_DIR:-$ROOT/llmadd_runs}
PY311=$ROOT/llmadd_env311/bin/python
fail() { echo "pin_run: $*" >&2; exit 3; }
[ $# -ge 2 ] || fail "usage: bash burst/pin_run.sh <commit> <script under tools/> [args...]"
commit=$1; script=$2; shift 2
[[ "$commit" =~ ^[0-9a-f]{7,40}$ ]] || fail "'$commit' is not a commit hash (a branch name would move)"
[[ "$script" =~ ^[A-Za-z0-9_.-]+\.py$ ]] || fail "'$script' is not a script file name under tools/"
[ -x "$PY311" ] || fail "no env311 python at $PY311 (task 331-env311)"
[ -n "${OUT:-}" ] || fail "OUT is not set (run_task.sbatch sets it)"
full=$(git -C "$REPO" rev-parse -q --verify "$commit^{commit}") || fail "commit $commit is not in $REPO"
blob=$(git -C "$REPO" rev-parse -q --verify "$full:tools/$script") || fail "tools/$script does not exist at $commit"
PIN=$RUNS_DIR/pinned_$commit
# tree_ok <dir>: <dir>/COMMIT is the full hash and <dir>/tools holds every tracked file of the commit under tools/ with its
# git blob (a missing, partial or edited tree is not ok; untracked extra files are ignored)
want=$(git -C "$REPO" ls-tree -r "$full" tools | awk '{print $3, $4}')
paths=$(git -C "$REPO" ls-tree -r --name-only "$full" tools)
[ -n "$want" ] || fail "no tracked files under tools/ at $full"
tree_ok() {
  [ "$(cat "$1/COMMIT" 2>/dev/null)" = "$full" ] &&
    [ "$want" = "$(cd "$1" && git hash-object --stdin-paths <<< "$paths" 2>/dev/null | paste -d' ' - <(echo "$paths"))" ]
}
for attempt in 1 2 3; do   # 3 attempts: two tasks repairing the same tree at once can take one rename from each other
  tree_ok "$PIN" && break
  [ "$attempt" = 1 ] || { sleep $((5 + RANDOM % 10)); tree_ok "$PIN" && break; }
  mkdir -p "$RUNS_DIR"
  if [ -e "$PIN" ]; then
    mv -T "$PIN" "$PIN.bad.$(date -u +%Y%m%dT%H%M%SZ).$$" 2>/dev/null &&
      echo "pin_run: $PIN was not the git tree of $full: moved aside, extracting again" >&2
  fi
  tmp=$(mktemp -d "$RUNS_DIR/.pinned_$commit.XXXXXX") || fail "cannot create a temporary dir in $RUNS_DIR"
  if chmod 755 "$tmp" && git -C "$REPO" archive "$full" tools | tar -x -C "$tmp" && echo "$full" > "$tmp/COMMIT" && tree_ok "$tmp"; then
    mv -T "$tmp" "$PIN" 2>/dev/null || rm -rf "$tmp"   # rename fails if another task created $PIN first: keep its tree
  else
    rm -rf "$tmp"; fail "git archive of $full did not give its tools/ tree"
  fi
done
tree_ok "$PIN" || fail "$PIN/tools differs from the git tree of $full"
[ "$(git hash-object "$PIN/tools/$script")" = "$blob" ] || fail "$PIN/tools/$script is not the blob $blob"
sha=$(sha256sum "$PIN/tools/$script" | cut -d' ' -f1)
mkdir -p "$OUT"
echo "$full $sha tools/$script $(date -u +%FT%TZ)" >> "$OUT/PINNED.txt"
echo "pin_run: tools/$script at $full (sha256 ${sha:0:12}) with $("$PY311" --version 2>&1)"
cd "$PIN/tools" || fail "cannot cd to $PIN/tools"
export PYTHONDONTWRITEBYTECODE=1
exec "$PY311" "$script" "$@"
