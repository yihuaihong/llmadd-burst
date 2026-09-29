"""Controls for the point-wise "pull towards main" intervention (Tab 1, 2026-09-23 in the shared notes).

Setting (re-implemented here, not their code): LoRA on two-term addition AND subtraction with operands and
results in 10..99, plus an MSE term that pulls the states at ONE layer towards a target model's states:
    L = CE + lam * [ relMSE(h_A, T(a)) + relMSE(h_B, T(b or a op b)) ]
with T(x) the target model's state of x at the same layer, read at x in the prefix "Q: x" (the operand slot),
and relMSE(h, t) = mean((h - t)^2) / mean(t^2). Evaluated on unseen two-term problems and on three-term
addition / subtraction (A+B+C, A-B-C, results in 10..99), which were never trained.

Arms (--arms):
    none          CE only
    opid_main     A -> main(a),        B -> main(b)          ("operand identity")
    run_main      A -> main(a),        B -> main(a op b)     ("running result at B")
    run_shuf      as run_main, but main's states of a fixed permutation of the numbers (same targets, wrong numbers)
    run_self      as run_main, but the checkpoint's OWN initial states (no information from main)
    run_random    A -> main(a),        B -> main(r), r a fixed random number per problem (running-result format, wrong value)

    python tools/mse_control.py --model <ckpt> --main_model <main> --out <dir> --layer 13 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import numlib as nl
from manifold_train import decoder_layers, number_states

ARMS = ("none", "opid_main", "run_main", "run_shuf", "run_self", "run_random")


def prompt(a, b, op): return f"Q: {a} {op} {b} = "
def prompt3(a, b, c, op): return f"Q: {a} {op} {b} {op} {c} = "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--main_model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--eval_bs", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(","); assert all(a in ARMS for a in arms), arms
    seeds = [int(s) for s in args.seeds.split(",")]
    L = args.layer
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda ps: nl.encode(tok, ps, dev)
    for op in "+-":
        toks = tok.convert_ids_to_tokens(tok(prompt(23, 45, op))["input_ids"])
        assert toks[-6:] == ["23", "Ġ" + op, "Ġ", "45", "Ġ=", "Ġ"], toks
    A, B = len(toks) - 6, len(toks) - 3
    prefix_all = torch.zeros(1000, A + 1, dtype=torch.long, device=dev)
    prefix_all[:] = enc([f"Q: {x}" for x in range(1000)])      # the operand-slot state of x depends only on this prefix

    # ---------------------------------------------------------------- data (operands and results in 10..99)
    rng = np.random.default_rng(0)
    add = [(a, b, "+") for a in range(10, 100) for b in range(10, 100) if a + b <= 99]
    sub = [(a, b, "-") for a in range(10, 100) for b in range(10, 100) if a - b >= 10]
    add = [add[i] for i in rng.permutation(len(add))]; sub = [sub[i] for i in rng.permutation(len(sub))]
    h = args.n_train // 2
    train = add[:h] + sub[:h]
    test = {"test_add": add[h:h + 500], "test_sub": sub[h:h + 500]}
    t3a = [(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 99]
    t3s = [(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a - b - c >= 10]
    t3a = [t3a[i] for i in rng.permutation(len(t3a))][:400]; t3s = [t3s[i] for i in rng.permutation(len(t3s))][:400]
    val = lambda a, b, op: a + b if op == "+" else a - b
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    evalsets = {k: (enc([prompt(a, b, op) for a, b, op in v]), torch.tensor([val(a, b, op) for a, b, op in v], device=dev)) for k, v in test.items()}
    evalsets["three_add"] = (enc([prompt3(a, b, c, "+") for a, b, c in t3a]), torch.tensor([a + b + c for a, b, c in t3a], device=dev))
    evalsets["three_sub"] = (enc([prompt3(a, b, c, "-") for a, b, c in t3s]), torch.tensor([a - b - c for a, b, c in t3s], device=dev))
    train_ids = enc([prompt(a, b, op) for a, b, op in train])
    train_ans = torch.tensor([val(a, b, op) for a, b, op in train], device=dev)
    tr_a = torch.tensor([a for a, _, _ in train], device=dev)
    tr_b = torch.tensor([b for _, b, _ in train], device=dev)
    tr_r = torch.tensor([val(a, b, op) for a, b, op in train], device=dev)
    tr_rand = torch.tensor(np.random.default_rng(7).integers(10, 100, len(train)), device=dev)
    perm = torch.tensor(np.concatenate([np.arange(10), 10 + np.random.default_rng(123).permutation(990)]), device=dev)

    # ---------------------------------------------------------------- target states
    tm = AutoModelForCausalLM.from_pretrained(args.main_model, torch_dtype=torch.bfloat16).to(dev)
    T_main = _states(tm, [L], prefix_all, A, args.eval_bs)[L]
    del tm
    torch.cuda.empty_cache() if dev == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    T_self = _states(model, [L], prefix_all, A, args.eval_bs)[L]
    print(f"targets ready ({time.time() - t0:.0f}s); |main| {T_main[10:100].norm(dim=1).mean():.1f} |self| {T_self[10:100].norm(dim=1).mean():.1f}")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    store = {}
    decoder_layers(model)[L].register_forward_hook(lambda _m, _i, o: store.__setitem__("h", nl._hidden(o)))

    def targets(arm, idx):
        a, b, r, rnd = tr_a[idx], tr_b[idx], tr_r[idx], tr_rand[idx]
        if arm == "opid_main":
            return T_main[a], T_main[b]
        if arm == "run_main":
            return T_main[a], T_main[r]
        if arm == "run_shuf":
            return T_main[perm[a]], T_main[perm[r]]
        if arm == "run_self":
            return T_self[a], T_self[r]
        if arm == "run_random":
            return T_main[a], T_main[rnd]
        raise ValueError(arm)

    rel = lambda h, t: ((h.float() - t) ** 2).mean() / (t ** 2).mean()

    def forward(ids):
        return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for name, (ids, ans) in evalsets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs]).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[name] = float((am == num_ids[ans]).float().mean())
        return res

    base = evaluate()
    (out / "base.json").write_text(json.dumps({"eval": base, "layer": L, "n_train": len(train)}, indent=1))
    print(f"base {base}")
    epochs = max(1, math.ceil(args.steps * args.bs / len(train)))
    for seed in seeds:
        for arm in arms:
            path = out / f"run_{arm}_seed{seed}.json"
            if path.exists():
                continue
            torch.manual_seed(seed)
            from peft.tuners.lora import LoraLayer
            for m in model.modules():
                if isinstance(m, LoraLayer):
                    m.reset_lora_parameters("default", True)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
            order_rng = np.random.default_rng(1000 + seed)
            log = []
            model.train()
            for ep in range(epochs):
                order = order_rng.permutation(len(train))
                ce_s = m_s = 0.0; nb = 0
                for i in range(0, len(order), args.bs):
                    idx = torch.tensor(order[i:i + args.bs], device=dev)
                    opt.zero_grad(set_to_none=True)
                    ce = Fn.cross_entropy(forward(train_ids[idx]), num_ids[train_ans[idx]])
                    if arm == "none":
                        loss, mse = ce, torch.zeros(())
                    else:
                        tA, tB = targets(arm, idx)
                        mse = rel(store["h"][:, A], tA) + rel(store["h"][:, B], tB)
                        loss = ce + args.lam * mse
                    loss.backward()
                    opt.step()
                    ce_s += float(ce.detach()); m_s += float(mse.detach()); nb += 1
                log.append({"epoch": ep, "ce": ce_s / nb, "mse": m_s / nb})
            del opt
            ev = evaluate()
            path.write_text(json.dumps({"arm": arm, "seed": seed, "layer": L, "lam": args.lam, "epochs": epochs,
                                        "train_log": log, "eval": ev}, indent=1))
            print(f"{arm} seed{seed}: {ev} ({time.time() - t0:.0f}s)")
    summarize(out)


def _states(model, layers, prefix_all, pos, bs):
    """number_states() for rows 0..999 (it skips 0..9)."""
    st = number_states(model, layers, prefix_all, pos, bs)
    dls = decoder_layers(model); got = {}
    hooks = [dls[L].register_forward_hook(lambda _m, _i, o, L=L: got.__setitem__(L, nl._hidden(o)[:, pos, :].float())) for L in layers]
    with torch.no_grad():
        model(input_ids=prefix_all[:10], logits_to_keep=1)
    for hk in hooks:
        hk.remove()
    for L in layers:
        st[L][:10] = got[L]
    return st


def summarize(out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    base = json.loads((out / "base.json").read_text())
    keys = list(base["eval"])
    lines = [f"# pull-towards-main controls (layer {base['layer']}, lam {runs[0]['lam'] if runs else '-'})\n\n",
             "| arm | seeds | " + " | ".join(keys) + " |\n", "|---|---|" + "---|" * len(keys) + "\n",
             "| base | - | " + " | ".join(f"{base['eval'][k]:.3f}" for k in keys) + " |\n"]
    for arm in ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        cells = []
        for k in keys:
            v = np.array([r["eval"][k] for r in rs])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        lines.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + " |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
