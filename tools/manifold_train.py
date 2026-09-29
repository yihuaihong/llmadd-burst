"""Manifold-regularised fine-tuning of an OLMo-2 checkpoint on two-digit addition.

Question: if fine-tuning also pulls the operand representations onto an ideal number manifold, does the
model generalise better (held-out pairs, three-term, three-digit) than with the task loss alone?

Loss:  CE(answer token) + lam * L_geo
L_geo: the GEOMETRY of the operand representation, not a readout of it. Every step the 90 prefixes "Q: x"
       (x = 10..99) are run through the model; at each layer in `--layers` the states of x form H [90, d],
       and L_geo = mean_L (1 - CKA(H, F_G)), with F_G [90, k] the ideal coordinates of 10..99 and linear
       CKA on centred Gram matrices. CKA is invariant only to rotations and isotropic scaling, so it is
       satisfied only if the representation itself has the shape of the ideal manifold.
       (v1 used ||(h - mu) R - F||^2 with a readout R fitted in 4096-d; 90 points can be mapped linearly
       onto ANY target there - shuffled included - so v1 constrained nothing. Found on s1-10k, 2026-09-27.)
Geometries G: none (task only) | helix | helix_shuf | digit | digit_shuf. "_shuf" uses the same basis on a
       fixed permutation of 10..99: same dimensionality and strength, wrong numbers.
       Learned targets (TEACHER): main | main_shuf | self - the target Gram is a MODEL's own states of the same
       numbers at the same layer (or its embedding rows with --geo_site emb): --main_model's (a later, more
       mature checkpoint) or this checkpoint's own before training ("self": an anchor that only resists drift).
       "3" variants (helix3, digit3, main3, ...) run over 10..999, sampled per step.
Tasks (--task add|sub) and splits (--split random | holdout_operand: 15 operand values never seen in training |
       carry: train without a units carry/borrow, test only with one); --save_preds keeps every eval answer.
Methods (--mode): lora (r=8, all linear layers) | reft (low-rank edit of the operand states at the
       manifold layers, base frozen) | full (all weights, fp32 master + AdamW; needs ~120 GB, e.g. one H200).

Every (geometry, seed) run starts from the same base weights; for a given seed all geometries share the
data order and the initialisation of the trainable parameters, so comparisons are paired. One JSON per
run under --out; existing runs are skipped (restartable).

    python tools/manifold_train.py --model <dir> --out <dir> --mode lora --geoms none,helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import numlib as nl

GEOMS = ("none", "helix", "helix_shuf", "digit", "digit_shuf")
# three-digit geometries over 10..999 (sampled per step, see --geo_sample)
GEOMS3 = ("helix3", "helix3_shuf", "digit3", "digit3_shuf")
# learned targets: name -> (source model, shuffled, 2 = 10..99 / 3 = 10..999)
TEACHER = {"main": ("main", False, 2), "main_shuf": ("main", True, 2), "self": ("self", False, 2),
           "main3": ("main", False, 3), "main3_shuf": ("main", True, 3), "self3": ("self", False, 3)}
ALL_GEOMS = GEOMS + GEOMS3 + tuple(TEACHER)


def geom_range(G: str) -> np.ndarray:
    return np.arange(10, 1000) if G.startswith(("helix3", "digit3")) else np.arange(10, 100)


def is_three(G: str) -> bool:
    return G in GEOMS3 or (G in TEACHER and TEACHER[G][2] == 3)


def two_prompt(a, b, op="+"): return f"Q: {a} {op} {b} = "
def three_prompt(a, b, c, op="+"): return f"Q: {a} {op} {b} {op} {c} = "
def terse_prompt(a, b, op="+"): return f"{nl.PREFIX}{a}{op}{b}="


def geom_features(G: str, x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if G.endswith("_shuf"):
        x = perm[x]; G = G[: -len("_shuf")]
    if G == "helix":
        return nl.features(x)[0]
    if G == "digit":
        return np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1)
    if G == "helix3":
        return np.concatenate([nl.features(x, periods=(2, 5, 10, 100, 1000), linear=False)[0], (x / 1000.0)[:, None]], 1)
    if G == "digit3":
        return np.concatenate([np.eye(10)[x // 100], np.eye(10)[(x // 10) % 10], np.eye(10)[x % 10]], 1)
    raise ValueError(G)


def cka(H: torch.Tensor, K_F: torch.Tensor) -> torch.Tensor:
    """Linear CKA between states H [n, d] and a centred target Gram K_F [n, n] (differentiable in H)."""
    H = H.float() - H.float().mean(0, keepdim=True)
    K_H = H @ H.T
    return (K_H * K_F).sum() / (K_H.norm() * K_F.norm() + 1e-12)


def target_grams(perm: np.ndarray, dev) -> dict:
    """Centred Gram matrices of the ideal coordinates of 10..99 for every two-digit geometry."""
    out = {}
    for G in ("helix", "helix_shuf", "digit", "digit_shuf"):
        F = geom_features(G, np.arange(10, 100), perm)
        F = F[:, F.std(0) > 1e-9]
        F = (F - F.mean(0)) / F.std(0)
        Ft = torch.tensor(F, dtype=torch.float32, device=dev)
        out[G] = Ft @ Ft.T
    return out


def target_features(G: str, perm: np.ndarray, dev) -> torch.Tensor:
    """Standardised ideal coordinates, row x for x in 0..999 (rows outside geom_range(G) are unused)."""
    xs = geom_range(G)
    F = geom_features(G, xs, perm)
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    out = torch.zeros(1000, F.shape[1], device=dev)
    out[torch.tensor(xs, device=dev)] = torch.tensor(F, dtype=torch.float32, device=dev)
    return out


def gram_of(F_rows: torch.Tensor) -> torch.Tensor:
    Fc = F_rows - F_rows.mean(0, keepdim=True)
    return Fc @ Fc.T


@torch.no_grad()
def transplant_number_embeddings(emb_mod, num_all: torch.Tensor, main_path: str, how: str, dev) -> dict:
    """Overwrite the input-embedding rows of the number tokens 0..999 with main's rows mapped into this
    checkpoint's embedding space by scaled orthogonal Procrustes fitted on all NON-number tokens.
    how = main | main_shuf (rows permuted within 10..99 and within 100..999). Returns fit diagnostics.
    how = main_shape | main_shape_shuf: fit the Procrustes map on the NUMBER rows themselves, i.e. keep this
    checkpoint's location/scale of the number cloud and adopt main's relative geometry (the shuffled variant
    permutes main's rows before the fit: same procedure, wrong correspondence)."""
    from safetensors import safe_open
    index = Path(main_path) / "model.safetensors.index.json"
    shard = (json.loads(index.read_text())["weight_map"]["model.embed_tokens.weight"] if index.exists() else "model.safetensors")
    with safe_open(str(Path(main_path) / shard), framework="pt") as f:
        Em = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    Ec = emb_mod.weight.detach().float()
    rng = np.random.default_rng(0)
    if how.startswith("main_shape"):
        Mn = Em[num_all]
        if how == "main_shape_shuf":
            p = np.arange(1000); p[10:100] = 10 + rng.permutation(90); p[100:] = 100 + rng.permutation(900)
            Mn = Mn[torch.tensor(p, device=dev)]
        Cn = Ec[num_all]
        A, B = Mn - Mn.mean(0), Cn - Cn.mean(0)
        U, S, Vt = torch.linalg.svd(A.T @ B, full_matrices=False)
        R = U @ Vt
        scale = S.sum() / (A ** 2).sum()
        rows = A @ R * scale + Cn.mean(0)
        r2 = 1 - ((rows - Cn) ** 2).sum() / (B ** 2).sum()
        emb_mod.weight[num_all] = rows.to(emb_mod.weight.dtype)
        return {"shape_fit_r2_number_rows": float(r2), "scale": float(scale),
                "mean_row_change_norm": float((rows - Cn).norm(dim=1).mean()), "mean_row_norm": float(Cn.norm(dim=1).mean())}
    V = Ec.shape[0]
    keep = torch.ones(V, dtype=torch.bool, device=dev); keep[num_all] = False
    perm_ids = torch.randperm(int(keep.sum()), generator=torch.Generator().manual_seed(0)).to(dev)
    others = torch.nonzero(keep).squeeze(1)[perm_ids]
    fit, held = others[: len(others) * 9 // 10], others[len(others) * 9 // 10:]
    A, B = Em[fit] - Em[fit].mean(0), Ec[fit] - Ec[fit].mean(0)
    U, S, Vt = torch.linalg.svd(A.T @ B)
    R = U @ Vt
    scale = S.sum() / (A ** 2).sum()
    mapped = lambda X: (X - Em[fit].mean(0)) @ R * scale + Ec[fit].mean(0)
    r2 = 1 - ((mapped(Em[held]) - Ec[held]) ** 2).sum() / ((Ec[held] - Ec[held].mean(0)) ** 2).sum()
    rows = mapped(Em[num_all])
    if how == "main_shuf":
        p = np.arange(1000); p[10:100] = 10 + rng.permutation(90); p[100:] = 100 + rng.permutation(900)
        rows = rows[torch.tensor(p, device=dev)]
    before = Ec[num_all].clone()
    emb_mod.weight[num_all] = rows.to(emb_mod.weight.dtype)
    return {"procrustes_heldout_r2_non_number_tokens": float(r2), "scale": float(scale),
            "mean_row_change_norm": float((rows - before).norm(dim=1).mean()), "mean_row_norm": float(before.norm(dim=1).mean())}


def decoder_layers(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "Olmo2DecoderLayer"]


@torch.no_grad()
def number_states(model, layers: list, prefix_all: torch.Tensor, pos: int, bs: int) -> dict:
    """{L: [1000, d] float32}: the state at `pos` after decoder layer L for the prefixes "Q: x" (rows 10..999)."""
    dls = decoder_layers(model)
    got: dict = {}
    hooks = [dls[L].register_forward_hook(lambda _m, _i, o, L=L: got.__setitem__(L, nl._hidden(o)[:, pos, :].float()))
             for L in layers]
    out = {L: torch.zeros(1000, model.config.hidden_size, device=prefix_all.device) for L in layers}
    try:
        for i in range(10, 1000, bs):
            model(input_ids=prefix_all[i:i + bs], logits_to_keep=1)
            for L in layers:
                out[L][i:i + bs] = got[L]
    finally:
        for h in hooks:
            h.remove()
    return out


class Reft(torch.nn.Module):
    """Low-rank additive edit of selected token states at one layer: h + (h Wd^T + b) Wu^T (Wu starts at 0)."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.down = torch.nn.Linear(d, r)
        self.up = torch.nn.Linear(r, d, bias=False)

    def reset(self) -> None:
        torch.nn.init.normal_(self.down.weight, std=self.down.in_features ** -0.5)
        torch.nn.init.zeros_(self.down.bias)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.down(h.float())).to(h.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("lora", "reft", "full"), required=True)
    ap.add_argument("--geoms", default=",".join(GEOMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--train_pairs", type=int, default=1500)
    ap.add_argument("--test_pairs", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--geo_site", choices=("layers", "emb"), default="layers",
                    help="layers: CKA on the A-slot states at --layers; emb: CKA on the input-embedding rows of 10..99 (needs --emb_delta unless mode=full)")
    ap.add_argument("--emb_delta", action="store_true", help="make the input-embedding rows of the number tokens 0..999 trainable (additive delta)")
    ap.add_argument("--emb_init", choices=("orig", "main", "main_shuf", "main_shape", "main_shape_shuf"), default="orig",
                    help="replace the number-token rows 0..999 of the input embedding before training: main = Procrustes-aligned rows of --main_model, main_shuf = same rows shuffled among the numbers")
    ap.add_argument("--main_model", default=None)
    ap.add_argument("--geo_sample", type=int, default=180, help="numbers per step for the 10..999 geometries (helix3/digit3)")
    ap.add_argument("--task", choices=("add", "sub"), default="add", help="sub: a - b with a >= b (answers stay single tokens)")
    ap.add_argument("--split", choices=("random", "holdout_operand", "carry"), default="random")
    ap.add_argument("--geo_numbers", choices=("all", "seen"), default="all",
                    help="two-digit geometries over all of 10..99, or only over the operand values seen in training")
    ap.add_argument("--save_preds", action="store_true", help="store every predicted answer of the eval sets (not train)")
    ap.add_argument("--device_map", default=None, help="'auto': split the model over all visible GPUs (e.g. 32B on 2 x 40 GB); inputs and losses live on the first GPU")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lr = args.lr or {"lora": 1e-4, "reft": 1e-3, "full": 1e-5}[args.mode]
    assert not (args.geo_site == "emb" and args.mode != "full" and not args.emb_delta), "--geo_site emb needs trainable embeddings (--emb_delta)"
    assert args.emb_init == "orig" or args.main_model, "--emb_init main* needs --main_model"
    geoms = args.geoms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    assert all(G in ALL_GEOMS for G in geoms), geoms
    assert not any(TEACHER.get(G, ("",))[0] == "main" for G in geoms) or args.main_model, "main* targets need --main_model"
    assert args.geo_numbers == "all" or not any(is_three(G) for G in geoms), "--geo_numbers seen is for the two-digit geometries"
    man_layers = [int(v) for v in args.layers.split(",")]
    op = {"add": "+", "sub": "-"}[args.task]
    t0 = time.time()

    # ---------------------------------------------------------------- tokenizer, prefixes, teacher states
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = ("cuda:0" if args.device_map else "cuda") if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    toks = tok.convert_ids_to_tokens(tok(two_prompt(23, 45, op))["input_ids"])
    assert toks[-6:] == ["23", "Ġ" + op, "Ġ", "45", "Ġ=", "Ġ"], toks
    slot_pos = {"A": len(toks) - 6, "B": len(toks) - 3}
    enc = lambda ps: nl.encode(tok, ps, dev)
    prefix_all = torch.zeros(1000, slot_pos["A"] + 1, dtype=torch.long, device=dev)
    prefix_all[10:] = enc([f"Q: {x}" for x in range(10, 1000)])      # A-slot state of x depends only on this prefix
    num_prefix = prefix_all[10:100]
    teach: dict = {}    # {"main" | "self": {L | "emb": [1000, d] float32}}, rows 0..9 unused
    if args.main_model:
        tm = AutoModelForCausalLM.from_pretrained(args.main_model, torch_dtype=torch.bfloat16).to(dev)
        tm.config.use_cache = False
        teach["main"] = number_states(tm, man_layers, prefix_all, slot_pos["A"], args.eval_bs)
        tn = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
        teach["main"]["emb"] = tm.get_input_embeddings().weight[tn].detach().float()
        del tm
        torch.cuda.empty_cache() if dev == "cuda" else None
        print(f"teacher states from {args.main_model} ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- model
    base_dtype = torch.float32 if (args.mode == "full" or dev == "cpu") else torch.bfloat16
    if args.device_map:
        # GPU 0 also holds the inputs, the embedding and the geometry prompts' activations: give it less of the model
        mem = ({i: ("30GiB" if i == 0 else "38GiB") for i in range(torch.cuda.device_count())}
               if torch.cuda.is_available() else None)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype, device_map=args.device_map, max_memory=mem)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype).to(dev)
    model.config.use_cache = False
    layers = decoder_layers(model)
    d = model.config.hidden_size
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (args.mode == "full" and dev == "cuda") else torch.autocast("cpu", enabled=False)
    if any(TEACHER.get(G, ("",))[0] == "self" for G in geoms):
        teach["self"] = number_states(model, man_layers, prefix_all, slot_pos["A"], args.eval_bs)
        teach["self"]["emb"] = model.get_input_embeddings().weight[num_ids[:1000]].detach().float()

    reft_mods = None
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        if args.device_map:   # 32B on 2 x 40 GB: recompute activations in backward (same gradients, less memory)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layers = decoder_layers(model)
    elif args.mode == "reft":
        for p in model.parameters():
            p.requires_grad_(False)
        reft_mods = torch.nn.ModuleDict({str(L): Reft(d, args.rank) for L in man_layers}).to(dev)
    else:
        init_state = None   # taken after the optional embedding transplant (below)

    def trainable():
        ps = list(reft_mods.parameters()) if args.mode == "reft" else [p for p in model.parameters() if p.requires_grad]
        return ps + ([emb_delta] if emb_delta is not None else [])

    def reset(seed: int) -> None:
        torch.manual_seed(seed)
        if emb_delta is not None:
            with torch.no_grad():
                emb_delta.zero_()
        if args.mode == "lora":
            from peft.tuners.lora import LoraLayer
            for m in model.modules():
                if isinstance(m, LoraLayer):
                    m.reset_lora_parameters("default", True)
        elif args.mode == "reft":
            for m in reft_mods.values():
                m.reset()
        else:
            model.load_state_dict({k: v.to(torch.float32) for k, v in init_state.items()})

    # ---------------------------------------------------------------- hooks: capture (+ ReFT edit) at manifold layers
    store: dict = {}
    reft_on = {"v": False}

    def make_hook(L: int):
        def hook(_m, _i, o):
            h = nl._hidden(o)
            if reft_mods is not None and reft_on["v"]:
                h0, h = h, h.clone()          # read from h0, write into the copy (autograd-safe)
                for p in slot_pos.values():
                    if p < h.shape[1]:
                        h[:, p, :] = reft_mods[str(L)](h0[:, p, :])
                o = nl._with_hidden(o, h)
            store[L] = h
            return o
        return hook

    for L in man_layers:
        layers[L].register_forward_hook(make_hook(L))

    # ---------------------------------------------------------------- number-token embeddings
    emb_mod = model.get_input_embeddings()
    num_all = num_ids[:1000]
    emb_info = {}
    if args.emb_init != "orig":
        emb_info = transplant_number_embeddings(emb_mod, num_all, args.main_model, args.emb_init, dev)
        print(f"embedding init {args.emb_init}: {emb_info}")
    emb_delta = None
    if args.emb_delta:
        emb_delta = torch.nn.Parameter(torch.zeros(1000, d, device=dev))
        row_of = torch.full((emb_mod.weight.shape[0],), -1, dtype=torch.long, device=dev)
        row_of[num_all] = torch.arange(1000, device=dev)

        def add_delta(_m, inp, out):
            r = row_of[inp[0]]
            m = r >= 0
            if not m.any():
                return out
            out = out.clone()
            out[m] = out[m] + emb_delta[r[m]].to(out.dtype)
            return out
        emb_mod.register_forward_hook(add_delta)

    if args.mode == "full":
        init_state = {k: v.detach().to("cpu", torch.bfloat16).clone() for k, v in model.state_dict().items()}

    def number_embedding_rows(xs=None) -> torch.Tensor:
        xs = torch.arange(10, 100, device=dev) if xs is None else xs
        E = emb_mod.weight[num_all[xs]].float()
        return E + emb_delta[xs] if emb_delta is not None else E

    def forward(ids: torch.Tensor, reft: bool) -> torch.Tensor:
        reft_on["v"] = reft
        with autocast:
            return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float().to(dev)

    # ---------------------------------------------------------------- data
    ans_of = (lambda P: P.sum(1)) if args.task == "add" else (lambda P: P[:, 0] - P[:, 1:].sum(1))
    rng = np.random.default_rng(0)
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100) if args.task == "add" or a >= b])
    pairs = pairs[rng.permutation(len(pairs))]
    heldout = np.array([], dtype=int)
    if args.split == "random":
        train, test = pairs[: args.train_pairs], pairs[args.train_pairs: args.train_pairs + args.test_pairs]
    else:
        if args.split == "holdout_operand":
            heldout = np.sort(np.random.default_rng(7).choice(np.arange(10, 100), 15, replace=False))
            in_train = ~np.isin(pairs, heldout).any(1)
        else:   # units carry (add) / borrow (sub) only in the test set
            u = pairs % 10
            in_train = (u[:, 0] + u[:, 1] < 10) if args.task == "add" else (u[:, 0] >= u[:, 1])
        train, test = pairs[in_train][: args.train_pairs], pairs[~in_train][: args.test_pairs]
    assert len(train) == args.train_pairs and len(test) == args.test_pairs, (len(train), len(test))
    # per-number means are taken per slot, so every value must occur in every slot of the probe prompts
    probe_is_train = args.task == "add" and args.split == "random"
    probe = train if probe_is_train else pairs
    for j in range(2):
        assert set(np.unique(probe[:, j])) >= set(range(10, 100)), f"slot {j}: some operand value never occurs in the probe prompts"
    seen = np.unique(train)
    if args.task == "add":
        trip = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 199])
        big = np.array([(a, b) for a in range(100, 500) for b in range(100, 500)])
    else:
        trip = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a - b - c >= 0])
        big = np.array([(a, b) for a in range(100, 500) for b in range(100, 500) if a >= b])
    three = trip[rng.permutation(len(trip))][:400]
    big = big[rng.permutation(len(big))][:400]
    perm = np.arange(100); perm[10:] = 10 + np.random.default_rng(123).permutation(90)
    evalsets = {
        "test": (enc([two_prompt(*x, op) for x in test]), ans_of(test)),
        "train": (enc([two_prompt(*x, op) for x in train]), ans_of(train)),
        "three_term": (enc([three_prompt(*x, op) for x in three]), ans_of(three)),
        "three_digit": (enc([two_prompt(*x, op) for x in big]), ans_of(big)),
        "test_terse": (enc([terse_prompt(*x, op) for x in test]), ans_of(test)),
    }
    items = {"test": test, "three_term": three, "three_digit": big, "test_terse": test}
    train_ids = evalsets["train"][0]
    probe_ids = train_ids if probe_is_train else enc([two_prompt(*x, op) for x in probe])
    num_of = torch.full((model.config.vocab_size,), -1, dtype=torch.long, device=dev)
    num_of[num_ids[:1000]] = torch.arange(1000, device=dev)

    @torch.no_grad()
    def evaluate(reft: bool, preds: dict | None = None) -> dict:
        """Accuracy per eval set; with `preds` also fills {set: predicted number or -1} (not for train)."""
        model.eval()
        res = {}
        for name, (ids, ans) in evalsets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs], reft).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[name] = float((am == num_ids[torch.tensor(ans, device=dev)]).float().mean())
            if preds is not None and name in items:
                preds[name] = num_of[am].tolist()
        return res

    @torch.no_grad()
    def number_means(reft: bool) -> dict:
        """{(L, slot): [90, d]} per-number mean state over the probe prompts (the training prompts by default)."""
        model.eval()
        sums = {(L, s): torch.zeros(90, d, device=dev) for L in man_layers for s in slot_pos}
        cnt = {s: torch.zeros(90, device=dev) for s in slot_pos}
        for i in range(0, len(probe_ids), args.eval_bs):
            forward(probe_ids[i:i + args.eval_bs], reft)
            chunk = probe[i:i + args.eval_bs]
            for j, s in enumerate(slot_pos):
                idx = torch.tensor(chunk[:, j] - 10, device=dev)
                for L in man_layers:
                    sums[(L, s)].index_add_(0, idx, store[L][:, slot_pos[s], :].float().to(dev))
                cnt[s].index_add_(0, idx, torch.ones(len(idx), device=dev))
        return {k: (v / cnt[k[1]][:, None]).cpu().numpy() for k, v in sums.items()}

    grams = target_grams(perm, dev)
    feats2 = {G: target_features(G, perm, dev) for G in grams}
    perm3 = np.arange(1000); perm3[10:] = 10 + np.random.default_rng(321).permutation(990)
    feats3 = {G: target_features(G, perm3, dev) for G in GEOMS3}
    perm_t, perm3_t = torch.tensor(perm, device=dev), torch.tensor(perm3, device=dev)
    geo_xs2 = None if args.geo_numbers == "all" else torch.tensor(seen, device=dev)   # None = all of 10..99
    geo_rng = np.random.default_rng(99)
    report_xs3 = torch.tensor(np.sort(np.random.default_rng(5).choice(np.arange(10, 1000), 300, replace=False)), device=dev)

    def geo_states(reft: bool, xs=None) -> dict:
        """{L: H [n, d]} A-slot states of the numbers xs (default 10..99), with grad; with --geo_site emb the
        single entry "emb" holds the input-embedding rows instead (no forward needed)."""
        if args.geo_site == "emb":
            return {"emb": number_embedding_rows(xs)}
        forward(num_prefix if xs is None else prefix_all[xs], reft)
        return {L: store[L][:, slot_pos["A"], :].to(dev) for L in man_layers}

    def target_gram(G: str, key, xs) -> torch.Tensor:
        """Centred target Gram of geometry G for the numbers xs (None = 10..99) at site `key` (a layer or "emb")."""
        if G in TEACHER:
            src, shuf, n = TEACHER[G]
            rows = torch.arange(10, 100, device=dev) if xs is None else xs
            if shuf:
                rows = (perm3_t if n == 3 else perm_t)[rows]
            return gram_of(teach[src][key][rows])
        if G in GEOMS3:
            return gram_of(feats3[G][xs])
        return grams[G] if xs is None else gram_of(feats2[G][xs])

    def geo_loss(G: str, reft: bool) -> torch.Tensor:
        if is_three(G):
            xs = torch.tensor(np.sort(geo_rng.choice(np.arange(10, 1000), args.geo_sample, replace=False)), device=dev)
        else:
            xs = geo_xs2
        Hs = geo_states(reft, xs)
        return sum(1 - cka(Hs[L], target_gram(G, L, xs)) for L in Hs) / len(Hs)

    @torch.no_grad()
    def geometry_report(reft: bool) -> dict:
        model.eval()
        key = lambda L: f"L{L}" if L != "emb" else "emb"
        forward(num_prefix, reft)
        Hs = {L: store[L][:, slot_pos["A"], :].to(dev) for L in man_layers}
        Hs["emb"] = number_embedding_rows()
        rep = {key(L): {"cka_" + G: float(cka(H, grams[G])) for G in grams} for L, H in Hs.items()}
        for src in teach:
            for L, H in Hs.items():
                rep[key(L)]["cka_" + src] = float(cka(H, gram_of(teach[src][L][10:100])))
        forward(prefix_all[report_xs3], reft)
        H3 = {L: store[L][:, slot_pos["A"], :].to(dev) for L in man_layers}
        H3["emb"] = number_embedding_rows(report_xs3)
        for L, H in H3.items():
            rep[key(L)].update({"cka_" + G: float(cka(H, gram_of(feats3[G][report_xs3]))) for G in GEOMS3})
            for src in teach:
                rep[key(L)]["cka_" + src + "3"] = float(cka(H, gram_of(teach[src][L][report_xs3])))
        return rep

    base_preds = {} if args.save_preds else None
    base_eval = evaluate(reft=False, preds=base_preds)
    base_means = number_means(reft=False)
    (out / "base.json").write_text(json.dumps({"eval": base_eval, "manifold": diagnostics(base_means, None, slot_pos),
                                               "geometry": geometry_report(reft=False), "task": args.task, "split": args.split,
                                               "heldout": heldout.tolist(), "items": {k: v.tolist() for k, v in items.items()},
                                               "preds": base_preds}, indent=1))
    print(f"base: {base_eval} ({time.time() - t0:.0f}s)")

    for seed in seeds:
        for G in geoms:
            path = out / f"run_{G}_seed{seed}.json"
            if path.exists():
                continue
            reset(seed)
            params = trainable()
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, **({"fused": True} if (args.mode == "full" and dev == "cuda") else {}))
            order_rng = np.random.default_rng(1000 + seed)
            log = []
            model.train()
            for ep in range(args.epochs):
                order = order_rng.permutation(len(train))
                ce_s = man_s = 0.0; nb = 0
                for i in range(0, len(order), args.bs):
                    bi = order[i:i + args.bs]
                    chunk = train[bi]
                    # free the previous gradients before the forwards, and run both forwards inside ONE autocast
                    # region so they share a single bf16 copy of the weights (full FT: two copies + stale grads
                    # overflowed 140 GB)
                    opt.zero_grad(set_to_none=True)
                    with autocast:
                        logits = forward(train_ids[torch.tensor(bi, device=dev)], reft=True)
                        ce = Fn.cross_entropy(logits, num_ids[torch.tensor(ans_of(chunk), device=dev)])
                        if G == "none":
                            with torch.no_grad():
                                ml = geo_loss("helix", reft=True)
                            loss = ce
                        else:
                            ml = geo_loss(G, reft=True)
                            loss = ce + args.lam * ml
                    loss.backward()
                    opt.step()
                    ce_s += float(ce.detach()); man_s += float(ml.detach()); nb += 1
                    del logits, loss
                log.append({"epoch": ep, "ce": ce_s / nb, "geo_loss": man_s / nb})
            del opt
            preds = {} if args.save_preds else None
            ev = evaluate(reft=True, preds=preds)
            diag = diagnostics(number_means(reft=True), None, slot_pos)
            path.write_text(json.dumps({"mode": args.mode, "loss": "v2_cka", "geo_site": args.geo_site, "emb_delta": args.emb_delta,
                                        "task": args.task, "split": args.split, "geo_numbers": args.geo_numbers, "preds": preds,
                                        "emb_init": args.emb_init, "emb_init_info": emb_info, "geom": G, "seed": seed, "lam": args.lam, "lr": lr,
                                        "geo_sample": args.geo_sample if is_three(G) else None,
                                        "layers": man_layers, "train_log": log, "eval": ev, "manifold": diag,
                                        "geometry": geometry_report(reft=True)}, indent=1))
            print(f"{args.mode} {G} seed{seed}: {ev} ({time.time() - t0:.0f}s)")
    summarize(out)


def diagnostics(means: dict, targets: dict, slot_pos: dict) -> dict:
    """Held-out-number R^2 of the helix and digit geometries on the per-number mean states (slot A)."""
    out = {}
    x = np.arange(10, 100)
    for (L, s), X in means.items():
        if s != "A":
            continue
        out[f"L{L}"] = {"r2_helix": nl.cv_r2(nl.features(x)[0], X),
                        "r2_digit": nl.cv_r2(np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1), X)}
    return out


def summarize(out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    if not runs:
        return
    base = json.loads((out / "base.json").read_text())
    keys = list(runs[0]["eval"])
    ts = f", task {runs[0].get('task', 'add')}, split {runs[0].get('split', 'random')}" + (f", geo_numbers {runs[0]['geo_numbers']}" if runs[0].get("geo_numbers", "all") != "all" else "")
    lines = [f"# manifold fine-tuning ({runs[0]['mode']}, lam {runs[0]['lam']}, lr {runs[0]['lr']}, layers {runs[0]['layers']}{ts})\n\n",
             "| geom | n seeds | " + " | ".join(keys) + " | r2 helix / digit (first layer) |\n", "|---|---|" + "---|" * (len(keys) + 1) + "\n",
             "| base | - | " + " | ".join(f"{base['eval'][k]:.3f}" for k in keys) + " | - |\n"]
    for G in ALL_GEOMS:
        rs = [r for r in runs if r["geom"] == G]
        if not rs:
            continue
        cells = []
        for k in keys:
            v = np.array([r["eval"][k] for r in rs])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        first = next(iter(rs[0]["manifold"]))
        r2 = np.mean([[r["manifold"][first]["r2_helix"], r["manifold"][first]["r2_digit"]] for r in rs], 0)
        ck = ""
        if "geometry" in rs[0]:
            gl = [k for k in rs[0]["geometry"] if k.startswith("L")][-1]
            c = np.mean([[r["geometry"][gl]["cka_helix"], r["geometry"][gl]["cka_digit"]] for r in rs], 0)
            ck = f" CKA@{gl} helix {c[0]:.2f} digit {c[1]:.2f}"
            if "cka_main" in rs[0]["geometry"][gl]:
                ck += f" main {np.mean([r['geometry'][gl]['cka_main'] for r in rs]):.2f} main3 {np.mean([r['geometry'][gl]['cka_main3'] for r in rs]):.2f}"
            if "emb" in rs[0]["geometry"]:
                e = np.mean([[r["geometry"]["emb"]["cka_helix"], r["geometry"]["emb"]["cka_digit"]] for r in rs], 0)
                ck += f"; emb helix {e[0]:.2f} digit {e[1]:.2f}"
        lines.append(f"| {G} | {len(rs)} | " + " | ".join(cells) + f" | {r2[0]:.2f} / {r2[1]:.2f}{ck} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
