"""GPU smoke test on Burst, plus a first look at the position-0 question.

1. Load an OLMo-2 revision in bf16 on the GPU and time it.
2. First-token accuracy on 200 two-digit additions in two prompt formats.
3. Residual-stream norms, per layer, for a number presented alone ("23", position 0) versus the same
   number as operand A inside "Output ONLY a number.23+45=". The collaborator's helix targets come from
   the former and are imposed on the latter. Captured with hooks on the decoder layers, so every layer
   (31 included) is the pre-final-norm residual.

    python tools/gpu_smoke.py --model /scratch/$USER/models/OLMo-2-1124-7B/main --out $OUT
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def capture(model, input_ids, attention_mask):
    """Return {layer: [B, T, H] float32} residual outputs of every decoder layer."""
    layers = model.model.layers
    store = {}
    hooks = [
        layer.register_forward_hook(
            lambda _m, _i, out, idx=idx: store.__setitem__(idx, (out[0] if isinstance(out, tuple) else out).float())
        )
        for idx, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        for h in hooks:
            h.remove()
    return logits, store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    report = {"model": args.model}

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    report["load_seconds"] = round(time.time() - t0, 1)
    report["gpu"] = torch.cuda.get_device_name(0)
    report["tokenization"] = {s: tok.convert_ids_to_tokens(tok(s)["input_ids"]) for s in ["23", "Output ONLY a number.23+45=", "Q: 23 + 45 = "]}

    rng = random.Random(0)
    pairs = [(a, b) for a in range(10, 100) for b in range(10, 100) if a + b <= 99]
    rng.shuffle(pairs)
    pairs = pairs[:200]
    formats = {"train_format": "Output ONLY a number.{a}+{b}=", "eval_format": "Q: {a} + {b} = "}
    acc = {}
    for name, fmt in formats.items():
        enc = tok([fmt.format(a=a, b=b) for a, b in pairs], return_tensors="pt", padding=True).to("cuda")
        assert enc["attention_mask"].all(), "prompts differ in length; this smoke test assumes equal lengths"
        with torch.no_grad():
            pred = model(**enc).logits[:, -1].argmax(-1).tolist()
        gold = [tok(str(a + b), add_special_tokens=False)["input_ids"][0] for a, b in pairs]
        acc[name] = sum(p == g for p, g in zip(pred, gold)) / len(pairs)
    report["first_token_accuracy_200_two_digit_additions"] = acc

    # position-0 number vs in-context operand A, same numbers
    nums = list(range(10, 100, 3))
    alone = tok([str(n) for n in nums], return_tensors="pt").to("cuda")
    ctx = tok([f"Output ONLY a number.{n}+45=" for n in nums], return_tensors="pt").to("cuda")
    a_pos = ctx["input_ids"].shape[1] - 4  # tokens: ... '.', A, '+', '45', '='
    assert tok.convert_ids_to_tokens(int(ctx["input_ids"][0, a_pos])) == str(nums[0])
    _, h_alone = capture(model, alone["input_ids"], alone["attention_mask"])
    _, h_ctx = capture(model, ctx["input_ids"], ctx["attention_mask"])
    rows = []
    for layer in sorted(h_alone):
        x = h_alone[layer][:, 0]  # [N, H]
        y = h_ctx[layer][:, a_pos]
        diff = (x - y).pow(2)
        top = torch.topk(diff.mean(0), 5)
        rows.append({
            "layer": layer,
            "norm_alone_pos0": round(x.norm(dim=-1).mean().item(), 2),
            "norm_in_context_A": round(y.norm(dim=-1).mean().item(), 2),
            "mse_alone_vs_context": round(diff.mean().item(), 4),
            "share_of_mse_in_top5_dims": round((top.values.sum() / diff.mean(0).sum()).item(), 3),
            "top5_dims": top.indices.tolist(),
            # variance across numbers: how much of the state actually depends on the number
            "number_variance_alone": round(x.var(0).sum().item(), 2),
            "number_variance_in_context": round(y.var(0).sum().item(), 2),
        })
    report["position0_vs_context"] = rows
    report["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    (out / "gpu_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "position0_vs_context"}, indent=2))
    for r in rows[:: max(1, len(rows) // 8)]:
        print(r)


if __name__ == "__main__":
    main()
