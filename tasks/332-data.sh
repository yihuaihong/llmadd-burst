# kind: cpu
# time: 03:00:00
# cpus: 16
# after: 331-env311
# Burst copies of the Torch pretraining inputs (2026-10-16, Batch 1 B0; intent spec report/audits/burst_infra_AUDIT.md),
# in /scratch/$USER/llmadd_data:
#   olmo2_tok         tokenizer files of allenai/OLMo-2-1124-7B at stage1-step2000-tokens9B (commit f872bfe9, = the Torch
#                     --tokenizer dir); same sha256 per file and the same ids as Torch on a fixed probe set
#   pretrain_data     4e8 + 2e6 tokens, the Torch build's arguments, fineweb-edu at the commit the Torch builds used
#   pretrain_data_3B  3e9 + 2e6 tokens from parquet files 000-004 (Torch: runs/data3B/job_data3B.sbatch)
# Expected sha256 = the Torch files (computed on Torch 2026-10-16). A mismatch renames that dir to <dir>.bad and exits 3,
# so no GPU task ('# after: 332-data') starts on wrong inputs. Last, the pinned trainer (64abc19) prints its --dry_inputs
# on CPU: the decoded rows must equal the Torch pt999 7pct-none seed-0 log (exit 3 otherwise). Idempotent: a finished
# build is not redone (tools/pretrain_data.py skips it), but every file is hashed again.
set -euo pipefail
ROOT=${LLMADD_ROOT:-/scratch/$USER}
D=$ROOT/llmadd_data; TOK=$D/olmo2_tok
PY311=$ROOT/llmadd_env311/bin/python
REV=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9   # fineweb-edu commit of both Torch builds (raw/.cache/huggingface/download)
export RAYON_NUM_THREADS=$(nproc) TOKENIZERS_PARALLELISM=true PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$D"
bad() { echo "## $1: MISMATCH with Torch -> renamed to $1.bad"; rm -rf "$1.bad"; mv -T "$1" "$1.bad"; exit 3; }
check() { (cd "$1" && sha256sum -c -) || bad "$1"; }   # check <dir>, "sha256  relative path" lines on stdin

echo "## 1. tokenizer $(date -u +%FT%TZ)"
"$PY311" - "$TOK" <<'PY'
import sys
from huggingface_hub import hf_hub_download
for f in ("config.json", "generation_config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.json"):
    hf_hub_download("allenai/OLMo-2-1124-7B", f, revision="f872bfe9299c775c947b59d93391c822531cf198", local_dir=sys.argv[1])
PY
check "$TOK" <<'SHA'
9db31043be9d1ddf55865444aca90e6b807ad5a4d555700bc38f0fda0220a2c9  config.json
73e7f06e4ea172c6edeb1a44cab4a603e76962a85ea7ef3245769ba46ca23a30  generation_config.json
b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502  merges.txt
3c6bf7c09d5473c303cee8575a22bb51e5153c17d177a721b43cd4785c6d09ae  special_tokens_map.json
73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca  tokenizer.json
91c69c665697785ace4ec7dbd159e1839bc9bb5033ab05a56bb0547521dc9ab0  tokenizer_config.json
9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195  vocab.json
SHA
# same ids as the Torch tokenizer (transformers 4.57.6 / tokenizers 0.22.2 on Torch): the probe strings one by one, and a
# digest over 0..9999, " 0".." 999", the four line formats on an operand grid and the batched call pretrain_data.py makes
"$PY311" - "$TOK" <<'PY' | tee "$OUT/tokenizer_probe.txt" || bad "$TOK"
import hashlib, json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
PROBES = ["Q: 7 + 538 = 545\n", "7 plus 538 equals 545\n", "The sum of 999 and 1 is 1000\n",
          "Compute 12 + 34. Result: 46\n", "1234 56789 0 00 007 3.14 1,000 -5 2024-10-16",
          "  leading spaces\tand tabs\n\nnew lines\r\n", "Ünïcödé café naïve — “quotes” 数字与结构 😀",
          "<|endoftext|>B1 VO2 T64 #5 $3.50 x2", "Visit https://example.org/path?x=1&y=2 for details (2024)."]
FORMATS = ("Q: {a} + {b} = ", "{a} plus {b} equals ", "The sum of {a} and {b} is ", "Compute {a} + {b}. Result: ")
EXPECT = {"vocab": 100278, "eos": 100257, "bos": 100257, "pad": 100277, "probes": [
    [48, 25, 220, 22, 489, 220, 22600, 284, 220, 20749, 198], [22, 5636, 220, 22600, 17239, 220, 20749, 198],
    [791, 2694, 315, 220, 5500, 323, 220, 16, 374, 220, 1041, 15, 198], [47354, 220, 717, 489, 220, 1958, 13, 5832, 25, 220, 2790, 198],
    [4513, 19, 220, 19282, 4578, 220, 15, 220, 410, 220, 11194, 220, 18, 13, 975, 220, 16, 11, 931, 482, 20, 220, 2366, 19, 12, 605, 12, 845],
    [220, 6522, 12908, 53577, 23204, 271, 943, 5238, 319],
    [53591, 77, 38672, 66, 3029, 67, 978, 53050, 95980, 588, 2001, 1054, 54382, 863, 48785, 19113, 58318, 37985, 78935, 91416],
    [100257, 33, 16, 44752, 17, 350, 1227, 674, 20, 400, 18, 13, 1135, 865, 17],
    [27312, 3788, 1129, 8858, 2726, 52076, 30, 87, 28, 16, 5, 88, 28, 17, 369, 3649, 320, 2366, 19, 570]],
    "ext_sha256": "e278563efa495c882fbefb40d508c2e21483d5b413f340313eac474d3a4314f5"}
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
ext = [enc(str(v)) for v in range(10000)] + [enc(" " + str(v)) for v in range(1000)]
grid = (0, 5, 9, 10, 42, 99, 100, 537, 999, 1000, 4321)
ext += [enc(f.format(a=a, b=b) + f"{a + b}\n") for f in FORMATS for a in grid for b in grid]
ext += tok(PROBES, add_special_tokens=False)["input_ids"]
got = {"vocab": len(tok), "eos": tok.eos_token_id, "bos": tok.bos_token_id, "pad": tok.pad_token_id,
       "probes": [enc(s) for s in PROBES], "ext_sha256": hashlib.sha256(json.dumps(ext).encode()).hexdigest()}
diff = [k for k in EXPECT if got[k] != EXPECT[k]]
print("tokenizer probe:", "identical to Torch" if not diff else f"DIFFERENT in {diff}: {json.dumps(got)}")
sys.exit(3 if diff else 0)
PY

build() {   # build <dir> <train tokens> <files>; then the parquet files and the bins against Torch (stdin)
  echo "## 2. $1 $(date -u +%FT%TZ)"
  "$PY311" tools/pretrain_data.py --tokenizer "$TOK" --out "$1" --tokens "$2" --files "$3" --revision "$REV" < /dev/null | tail -n 3
  check "$1"
}
build "$D/pretrain_data" 400000000 sample/10BT/000_00000.parquet,sample/10BT/001_00000.parquet <<'SHA'
b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871  raw/sample/10BT/000_00000.parquet
3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693  raw/sample/10BT/001_00000.parquet
4de2ed19093c37b302b6a1ced56136e6a8b805e6d6b48057a77316811704ff11  train.bin
d6f744a99b988ed6928fb588356cc8fe3d61bcf348b8d06fc0eece25c624a4c8  val.bin
SHA
# parquet files 000 and 001 were just verified in pretrain_data: hard links instead of a second 4.3 GB download (if the
# link fails, pretrain_data.py downloads them); the 3B check below hashes them again
mkdir -p "$D/pretrain_data_3B/raw/sample/10BT"
for f in 000_00000 001_00000; do
  [ -e "$D/pretrain_data_3B/raw/sample/10BT/$f.parquet" ] ||
    ln "$D/pretrain_data/raw/sample/10BT/$f.parquet" "$D/pretrain_data_3B/raw/sample/10BT/" 2>/dev/null || true
done
build "$D/pretrain_data_3B" 3000000000 sample/10BT/000_00000.parquet,sample/10BT/001_00000.parquet,sample/10BT/002_00000.parquet,sample/10BT/003_00000.parquet,sample/10BT/004_00000.parquet <<'SHA'
b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871  raw/sample/10BT/000_00000.parquet
3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693  raw/sample/10BT/001_00000.parquet
547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7  raw/sample/10BT/002_00000.parquet
22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb  raw/sample/10BT/003_00000.parquet
33557ddd87a07a4ae6fcaf7a4789c7b484e5cc0c273ca12a65b74200e6d8748b  raw/sample/10BT/004_00000.parquet
5e456d775670562dbdd96680eed93d36de2ef0d299bff32a5cc975943828d485  train.bin
4b7c27cfd8613cac30ecec1f634cce023bb895611af0b14ec4c4cb02866ca609  val.bin
SHA
cp "$D/pretrain_data/meta.json" "$OUT/meta_pretrain_data.json"; cp "$D/pretrain_data_3B/meta.json" "$OUT/meta_pretrain_data_3B.json"

echo "## 3. pinned trainer, --dry_inputs on CPU vs the Torch log $(date -u +%FT%TZ)"
# 64abc19 = the pin of the 2026-10-16 batch; the selected lines of the Torch log llmadd_pt999-7pct-none-s0_19840089.out
# (79e9b410dcc3 snapshot) have this sha256, and so does a3bce3ab63c1 on Torch's data (checked on Torch 2026-10-16)
bash burst/pin_run.sh 64abc19 pretrain_range.py --data "$D/pretrain_data" --tokenizer "$TOK" --tokenization whole \
  --out "$OUT/smoke" --target none --arith_lines 4 --seed 0 --tag _smoke --dry_inputs > "$OUT/dry_inputs_smoke.txt" 2>&1 \
  || { cat "$OUT/dry_inputs_smoke.txt"; exit 3; }
got=$(set +o pipefail; grep -E "^(line:|held_b|ood_b|\{'held_b11'|model:|train row)" "$OUT/dry_inputs_smoke.txt" | head -n 9 | sha256sum | cut -d' ' -f1)
cat "$OUT/dry_inputs_smoke.txt"
[ "$got" = af96f9c50f36e2f18dfae7c55121a57180afb1fd34076fa3c0b3c2775738e1dc ] \
  || { echo "## dry_inputs differ from Torch (sha256 $got)"; exit 3; }
echo "## dry_inputs identical to Torch pt999 7pct-none seed 0; data verified $(date -u +%FT%TZ)" | tee "$OUT/data_ok.txt"
du -sh "$D"/* | tee -a "$OUT/data_ok.txt"; df -h "$ROOT" | tail -1 | tee -a "$OUT/data_ok.txt"
