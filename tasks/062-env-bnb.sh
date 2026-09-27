# kind: cpu
# time: 01:00:00
# cpus: 4
# after: 001-env
# bitsandbytes for the 8-bit AdamW used by FSDP full FT on 2 x A100 40GB.
set -eu
"$PY" -m pip install -q bitsandbytes
"$PY" -c "import bitsandbytes as b; print('bitsandbytes', b.__version__)" | tee "$OUT/bnb.txt"
