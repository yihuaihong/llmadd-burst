# kind: cpu
# time: 02:00:00
# cpus: 4
# after: 000-hello
# Python env for all later tasks, at /scratch/$USER/llmadd_env. transformers is pinned to 4.57.x
# (the version whose hidden_states / hook behaviour we checked on Torch).
set -eu
[ -x "$PY" ] || python3 -m venv "$ENV_DIR"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch
"$PY" -m pip install -q "transformers>=4.57,<4.58" peft accelerate safetensors huggingface_hub \
    numpy scipy scikit-learn matplotlib tqdm
"$PY" - > "$OUT/env.txt" <<'PY'
import sys, torch, transformers, peft, sklearn, numpy
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "peft", peft.__version__)
print("numpy", numpy.__version__, "sklearn", sklearn.__version__)
PY
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
cat "$OUT/env.txt"
