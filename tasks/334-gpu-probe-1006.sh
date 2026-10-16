#!/bin/bash
# kind: gpu
# time: 00:15:00
# cpus: 4
# account: ds_ga_1006_001-2026fa
# after: 331-env311
# Does the second course account actually START GPU jobs here? (2026-10-16 re-audit: 330 only validates with
# sbatch --test-only; a job that is accepted but pends on an association / QOS limit would stall the pipeline, because
# only running tasks tick.) Pinned to ds_ga_1006_001-2026fa by the '# account:' header; tick.sh lists 1011 first, so
# the batch never waits on this. Status COMPLETED = the account runs GPU jobs; a long PENDING in heartbeat.json
# in_flight (with its reason) = it does not, and 1006 is taken out of GPU_ACCOUNTS.
set -u
{
  echo "## gpu probe on account ds_ga_1006_001-2026fa  job ${SLURM_JOB_ID:-none}  $(hostname -s)  $(date -u +%FT%TZ)"
  echo "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-?} partition=${SLURM_JOB_PARTITION:-?}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  "$LLMADD_ROOT/llmadd_env311/bin/python" - <<'PY'
import time, torch
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(50):
    y = x @ x
torch.cuda.synchronize()
print(f"torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name()} "
      f"bf16 matmul {50 * 2 * 4096**3 / (time.time() - t) / 1e12:.0f} TFLOP/s")
PY
} 2>&1 | tee "$OUT/gpu_probe.txt"
test "${SLURM_JOB_ACCOUNT:-}" = ds_ga_1006_001-2026fa
