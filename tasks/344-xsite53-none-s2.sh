# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 330-probe 331-env311 332-data
# xsite53 (Batch 1 B1: does Burst A100 40GB reproduce Torch L40S? intent spec and pass rule
# report/audits/burst_infra_AUDIT.md): 53M d384x6, 7% arithmetic (4 lines), lr 1e-3, lambda 5, 3e8 tokens = 18310 steps of
# 32 x 512 on pretrain_data, seed 2, arm none.
# Torch reference runs/pt999 7pct-none (same flags); only --data / --tokenizer / --out / --tag differ. The tag is unique so a summary never merges sites.
# Trainer pinned at 64abc19 (tools/pretrain_range.py sha256 a3bce3ab63c1...) through burst/pin_run.sh; env llmadd_env311.
# 1. inputs = Torch's: data sha256; the --dry_inputs rows equal the Torch log's (exit 3 otherwise, before any training)
# 2. training with --save_model --ckpt_every 2000 (a retry keeps $OUT and resumes from the checkpoint)
# 3. offline_lm.py on the saved model (full val.bin)   4. P0 checks vs Torch -> $OUT/p0_check.{json,txt} (exit 4 on a fail)
# nvidia-smi utilisation / memory every 60 s -> $OUT/gpu.csv (planning: ~0.75 A100-h; limit = 2x a pessimistic 1 h).
set -euo pipefail
SEED=2; TAG=x53-7pct-none; ARM="--target none"
ROOT=${LLMADD_ROOT:-/scratch/$USER}; DATA=$ROOT/llmadd_data/pretrain_data; TOK=$ROOT/llmadd_data/olmo2_tok; PIN=64abc19
[ -s "$OUT/gpu.csv" ] || echo "timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > "$OUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader -l 60 >> "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{ date -u +%FT%TZ; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "cpus $(nproc)"; } >> "$OUT/node.txt" 2>&1 || true

echo "## 1. inputs $(date -u +%FT%TZ)"
(cd "$DATA" && sha256sum -c -) <<'SHA' || exit 3
4de2ed19093c37b302b6a1ced56136e6a8b805e6d6b48057a77316811704ff11  train.bin
d6f744a99b988ed6928fb588356cc8fe3d61bcf348b8d06fc0eece25c624a4c8  val.bin
SHA
ARGS="--data $DATA --tokenizer $TOK --tokenization whole --out $OUT/$TAG $ARM --arith_lines 4 --seed $SEED --tag _$TAG --save_model --ckpt_every 2000"
bash burst/pin_run.sh $PIN pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || { cat "$OUT/dry_inputs.txt"; exit 3; }
cat "$OUT/dry_inputs.txt"
# sha256 of the same lines in the Torch log llmadd_pt999-7pct-none-s2_*.out (a seed's rows do not depend on the arm)
got=$(set +o pipefail; grep -E "^(line:|held_b|ood_b|\{'held_b11'|model:|train row)" "$OUT/dry_inputs.txt" | head -n 9 | sha256sum | cut -d' ' -f1)
[ "$got" = 52ab489ca66f606a304ec196fce9239a946bb9cd3af9a8b67f0085696ae6579c ] || { echo "## dry_inputs differ from Torch (sha256 $got)"; exit 3; }
grep -q " a3bce3ab63c1[0-9a-f]* tools/pretrain_range.py " "$OUT/PINNED.txt" || { echo "## the pinned trainer is not a3bce3ab63c1"; exit 3; }

echo "## 2. training $(date -u +%FT%TZ)"
bash burst/pin_run.sh $PIN pretrain_range.py $ARGS
echo "## 3. offline LM $(date -u +%FT%TZ)"
bash burst/pin_run.sh $PIN offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"
echo "## 4. P0 checks vs Torch runs/pt999/out/7pct-none (79e9b410dcc3) $(date -u +%FT%TZ)"
"$ROOT/llmadd_env311/bin/python" - "$OUT/$TAG/run_whole_none_seed${SEED}_$TAG.json" "$OUT/offline_lm.json" "$OUT/gpu.csv" \
  "$OUT/PINNED.txt" none $SEED 11.579556465148926 11.56964123249054 52.29 <<'PY' | tee "$OUT/p0_check.txt"
import csv, json, statistics, sys
from pathlib import Path
run, offl, gpu, pinned, arm = sys.argv[1:6]
seed = int(sys.argv[6]); ref_lm0, ref_val0, ref_area = map(float, sys.argv[7:10])
B = ("b11", "b12", "b13", "b22", "b23", "b33")
r = json.loads(Path(run).read_text()); c = r["curve"]
mean = statistics.fmean
area = lambda b=None: 100 * mean([e[f"held_{b}"] if b else mean([e[f"held_{k}"] for k in B]) for e in c])
checks = []
def chk(name, ok, value):
    checks.append({"check": name, "ok": bool(ok), "value": value})
chk("code_sha a3bce3ab63c1", r.get("code_sha") == "a3bce3ab63c1", r.get("code_sha"))
chk("finished: last eval at step 18309 of 18310", c[-1]["step"] == r["steps"] - 1 == 18309, [c[-1]["step"], r["steps"]])
chk("step-0 lm within 1e-3 of Torch", abs(c[0]["lm"] - ref_lm0) <= 1e-3, c[0]["lm"] - ref_lm0)
chk("step-0 val_loss within 2e-3 of Torch", abs(c[0]["val_loss"] - ref_val0) <= 2e-3, c[0]["val_loss"] - ref_val0)
if arm == "W":
    chk("prior_events == [[916, 1], [7324, 0]]", r.get("prior_events") == [[916, 1], [7324, 0]], r.get("prior_events"))
    on = [e for e in c if e["prior_on"]]
    chk("geo > 0 at every eval with the prior on", bool(on) and all((e["geo"] or 0) > 0 for e in on), len(on))
else:
    chk("prior never on", c[-1]["prior_on_frac"] == 0 and r.get("prior_events") == [], c[-1]["prior_on_frac"])
o = json.loads(Path(offl).read_text()).get(str(Path(run).resolve())) if Path(offl).exists() else None
chk("offline_lm on the full val, |orig_minus_logged| < 0.002",
    o is not None and o["max_windows"] == 0 and abs(o["orig_minus_logged"]) < 0.002, o and o["orig_minus_logged"])
rate = [(b["sec"] - a["sec"]) / (b["step"] - a["step"]) for a, b in zip(c, c[1:]) if b["sec"] > a["sec"] and b["step"] > a["step"]]
util, mem = [], []
for row in csv.reader(open(gpu)) if Path(gpu).exists() else []:
    try:
        util.append(int(row[2].split()[0])); mem.append(int(row[4].split()[0]))
    except (IndexError, ValueError):
        pass
lines = Path(pinned).read_text().splitlines() if Path(pinned).exists() else []
info = {"seed": seed, "arm": arm, "tag": r["args"]["tag"], "area": round(area(), 2), "b33_area": round(area("b33"), 2),
        "torch_area": ref_area, "area_minus_torch": round(area() - ref_area, 2),
        "final_val_bpb": c[-1]["val_bits_per_byte"], "bpb_full": o and o["bpb_full"], "cka_ops_helix3": c[-1].get("cka_ops_helix3"),
        "prior_used_frac": c[-1].get("prior_used_frac"), "sec_per_step_median": statistics.median(rate) if rate else None,
        "projected_run_h": statistics.median(rate) * r["steps"] / 3600 if rate else None,
        "gpu_util_median": statistics.median(util) if util else None, "gpu_mem_max_MiB": max(mem) if mem else None,
        "gpu_samples": len(util), "pinned_calls": len(lines), "evals": len(c)}
ok = all(k["ok"] for k in checks)
Path(run).parent.parent.joinpath("p0_check.json").write_text(json.dumps({"p0_pass": ok, "checks": checks, "info": info}, indent=1))
for k in checks:
    print(("PASS " if k["ok"] else "FAIL ") + k["check"] + f"  ({k['value']})")
print(json.dumps(info))
print("P0", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
PY
