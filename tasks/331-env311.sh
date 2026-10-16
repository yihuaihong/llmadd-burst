# kind: cpu
# time: 01:30:00
# cpus: 8
# Python env for the from-scratch pretraining tasks (2026-10-16, Batch 1 B0; intent spec report/audits/burst_infra_AUDIT.md):
# /scratch/$USER/llmadd_env311 = CPython 3.11.13 + the versions of Torch's open_r1 env (peft / geonamescache: Torch's
# llmadd_torch/pylib) for every package the tools import and all their dependencies, from PyPI. open_r1's torch 2.10.0
# is the PyPI wheel, built with CUDA 12.8 for sm_70..sm_120 (Burst A100: sm_80, driver 610.43.02). CPython comes from
# uv (Burst's python3 is 3.9); uv, its cache and the interpreter live under /scratch/$USER/llmadd_uv, nothing in $HOME.
# The old py3.9 env (llmadd_env, $PY) is not touched. Idempotent: an env completed with the same pins is only checked; one
# completed with other pins is never deleted (exit 3).
# Exit 3 = a version differs from Torch's.
set -euo pipefail
ROOT=${LLMADD_ROOT:-/scratch/$USER}
E=$ROOT/llmadd_env311; U=$ROOT/llmadd_uv
export UV_CACHE_DIR=$U/cache UV_PYTHON_INSTALL_DIR=$U/python UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed PIP_CACHE_DIR=$U/pip_cache
# pip freeze of /scratch/yh6210/miniconda3/envs/open_r1 (+ llmadd_torch/pylib) on Torch, 2026-10-16, restricted to the
# dependency closure of torch transformers tokenizers numpy scipy pyarrow peft huggingface_hub hf-xet safetensors
# accelerate geonamescache scikit-learn matplotlib tqdm
cat > "$OUT/requirements311.txt" <<'REQ'
accelerate==1.13.0
certifi==2026.1.4
charset-normalizer==3.4.4
contourpy==1.3.3
cuda-bindings==12.9.4
cuda-pathfinder==1.5.3
cycler==0.12.1
filelock==3.20.3
fonttools==4.61.1
fsspec==2026.2.0
geonamescache==3.0.2
hf-xet==1.4.3
huggingface-hub==0.36.2
idna==3.11
jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.4.9
markupsafe==3.0.2
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
numpy==2.0.1
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
packaging==25.0
peft==0.17.1
pillow==12.0.0
psutil==7.0.0
pyarrow==23.0.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pyyaml==6.0.2
regex==2026.1.15
requests==2.32.5
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.17.0
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.2
torch==2.10.0
tqdm==4.67.3
transformers==4.57.6
triton==3.6.0
typing-extensions==4.15.0
urllib3==2.5.0
REQ
if [ -f "$E/.complete" ] && cmp -s "$E/.complete" "$OUT/requirements311.txt"; then
  echo "## $E was completed with these pins: checking the versions only"
elif [ -f "$E/.complete" ]; then   # running tasks may use it: a task never deletes a completed env
  echo "## $E was completed with OTHER pins and may be in use: not rebuilt (new pins = a new env dir, or remove it by hand"
  echo "## when no task runs)"; diff "$E/.complete" "$OUT/requirements311.txt" || true; exit 3
else
  rm -rf "$E"   # no env or an unfinished build: nothing uses it yet (the tasks wait for 331-env311 COMPLETED)
  if [ ! -x "$U/boot/bin/uv" ]; then
    rm -rf "$U/boot"; python3 -m venv "$U/boot"
    "$U/boot/bin/pip" install -q --upgrade pip && "$U/boot/bin/pip" install -q uv==0.12.5
  fi
  "$U/boot/bin/uv" python install --no-bin 3.11.13   # --no-bin: no python3.11 link in ~/.local/bin
  "$U/boot/bin/uv" venv -q --python 3.11.13 "$E"
  "$U/boot/bin/uv" pip install -q --python "$E/bin/python" --index-url https://pypi.org/simple -r "$OUT/requirements311.txt"
fi
if ! "$E/bin/python" - "$OUT/requirements311.txt" <<'PY' | tee "$OUT/env311.txt"; then
import sys
from importlib import metadata

bad = []
pins = [l.strip().split("==") for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
for name, want in pins:
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = None
    if have != want:
        bad.append(f"{name}: {have} (Torch {want})")
if sys.version.split()[0] != "3.11.13":
    bad.append(f"python {sys.version.split()[0]} (Torch 3.11.13)")
import numpy, pyarrow.parquet, scipy, tokenizers, torch, transformers, peft, geonamescache   # noqa: F401 (imports work)
from transformers import Olmo2Config, Olmo2ForCausalLM   # the from-scratch model class (pretrain_helix.build_model)
m = Olmo2ForCausalLM(Olmo2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2))
loss = m(input_ids=torch.arange(16)[None], labels=torch.arange(16)[None]).loss
arch = torch._C._cuda_getArchFlags()
if torch.version.cuda != "12.8" or "sm_80" not in arch.split():
    bad.append(f"torch CUDA build {torch.version.cuda} arch {arch} (Torch 12.8, needs sm_80)")
wheel = metadata.distribution("torch").read_text("WHEEL") or ""
print("python", sys.version.split()[0], sys.executable)
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "arch", arch,
      "| wheel", " ".join(l for l in wheel.splitlines() if l.startswith(("Tag", "Build"))))
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__, "numpy", numpy.__version__,
      "scipy", scipy.__version__, "pyarrow", pyarrow.__version__, "peft", peft.__version__,
      "geonamescache", metadata.version("geonamescache"), "huggingface_hub", metadata.version("huggingface-hub"))
print(f"tiny Olmo2 forward on CPU: loss {float(loss):.3f}")
print(f"{len(pins)} pinned packages: " + ("all equal to Torch's open_r1" if not bad else "MISMATCH " + "; ".join(bad)))
sys.exit(3 if bad else 0)
PY
  rm -f "$E/.complete"; exit 3   # a re-run rebuilds
fi
cp "$OUT/requirements311.txt" "$E/.complete"
"$U/boot/bin/uv" pip freeze -q --python "$E/bin/python" > "$OUT/pip_freeze311.txt"
"$U/boot/bin/uv" cache clean -q   # ~7 GB of wheels; a rebuild downloads them again (~2 min)
du -sh "$E" "$U" 2>/dev/null | tee -a "$OUT/env311.txt"
