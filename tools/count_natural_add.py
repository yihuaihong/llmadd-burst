"""Natural additions in a tokenised corpus (flat uint32 OLMo-2 tokens, as written by pretrain_data.py).

Counts occurrences of "a + b = c" (spaces optional; OLMo-2 tokens: '7', ' +', ' ', '538', ' =', ' ', '545') whose operands
are standalone numbers 0..999: not part of a longer number, a decimal, a negative number or a larger expression (the
token before a must not be a digit, '.', ',', '-', '+', '*', 'x', '/', '^' ...; the token after b is '='). Reports:
  - matches with an answer c, and how many are correct (c == a + b), per operand-length bucket (1+1 ... 3+3)
  - distinct unordered pairs seen with a correct answer, and their share of all pairs of the bucket
  - for the held-out split of tools/pretrain_range.py (held_out(default_rng(seed)), seeds --seeds): how many held-out
    pairs occur in either order (any answer), and their token spans -> <out>_heldspans_seed<s>.npy, the list a training
    run on this corpus must exclude (reject text chunks that overlap a span)
  - the word form "a plus b equals / is c" separately

    python tools/count_natural_add.py --data /scratch/.../pretrain_finemath --tokenizer <olmo2 dir> --seeds 0-7 --out <prefix>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pretrain_range import BUCKETS, held_out

LEFT_BAD = tuple(".,-+*x×/÷^−_=")   # a token ending in one of these makes a part of something larger (except '=')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--out", required=True, help="prefix for <out>.json and <out>_heldspans_seed<s>.npy")
    ap.add_argument("--chunk", type=int, default=50_000_000)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    dec = [tok.decode([i]) for i in range(len(tok))]
    numv = np.array([int(t) if t.isascii() and t.isdigit() and len(t) <= 3 and (t == "0" or t[0] != "0") else -1 for t in dec])
    alldig = np.array([len(t) > 0 and t.isascii() and t.isdigit() for t in dec])
    plus = np.array([t.strip() == "+" for t in dec])
    plusw = np.array([t.strip().lower() == "plus" for t in dec])
    eq = np.array([t.strip() == "=" for t in dec])
    eqw = np.array([t.strip().lower() in ("equals", "is", "=") for t in dec])
    space = np.array([len(t) > 0 and t.strip() == "" and "\n" not in t for t in dec])
    dstart = np.array([len(t) > 0 and t[0].isdigit() for t in dec])
    dend = np.array([len(t) > 0 and t[-1].isdigit() for t in dec])
    lbad = np.array([len(t.strip()) > 0 and (t.strip()[-1] in LEFT_BAD and t.strip() != "=") for t in dec])
    T = np.memmap(Path(args.data) / "train.bin", dtype=np.uint32, mode="r")
    a0, a1 = (int(v) for v in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
    held = {s: {(min(a, b), max(a, b)) for v in held_out(np.random.default_rng(s)).values() for a, b in v} for s in range(a0, a1 + 1)}

    def blen(a, b):
        la, lb = sorted((len(str(a)), len(str(b))))
        return f"b{la}{lb}"

    stats = {"symbolic": Counter(), "word": Counter()}
    pairs_ok = {"symbolic": Counter(), "word": Counter()}
    spans = defaultdict(list)   # seed -> [(start, end)] of held-out pair occurrences (either form, any answer)
    n = len(T)
    for start in range(0, n, args.chunk):
        lo = max(0, start - 8)
        seg = np.asarray(T[lo: min(n, start + args.chunk + 24)]).astype(np.int64)
        for form, opmask, eqmask in (("symbolic", plus, eq), ("word", plusw, eqw)):
            for p in np.flatnonzero(opmask[seg]):
                if not (start <= lo + p < start + args.chunk):
                    continue
                i = p - 1
                if i >= 0 and space[seg[i]]:
                    i -= 1
                if i < 0 or numv[seg[i]] < 0:
                    continue
                if i >= 1 and (dend[seg[i - 1]] or lbad[seg[i - 1]]):
                    continue
                if i >= 2 and space[seg[i - 1]] and (dend[seg[i - 2]] or lbad[seg[i - 2]]):
                    continue
                j = p + 1
                if j < len(seg) and space[seg[j]]:
                    j += 1
                if j + 1 >= len(seg) or numv[seg[j]] < 0 or dstart[seg[j + 1]]:
                    continue
                k = j + 1
                if k < len(seg) and space[seg[k]]:
                    k += 1
                if k >= len(seg) or not eqmask[seg[k]]:
                    continue
                a, b = int(numv[seg[i]]), int(numv[seg[j]])
                key = (min(a, b), max(a, b))
                for s, h in held.items():
                    if key in h:
                        spans[s].append((lo + i, lo + k + 4))
                k += 1
                if k < len(seg) and space[seg[k]]:
                    k += 1
                cs = ""
                while k < len(seg) and alldig[seg[k]] and len(cs) < 7:
                    cs += dec[seg[k]]
                    k += 1
                bk = blen(a, b)
                stats[form][(bk, "no_answer" if not cs else ("correct" if int(cs) == a + b else "wrong"))] += 1
                if cs and int(cs) == a + b:
                    pairs_ok[form][key] += 1
        print(f"{min(n, start + args.chunk) / 1e6:.0f}M / {n / 1e6:.0f}M tokens; symbolic correct so far "
              f"{sum(v for (b, c), v in stats['symbolic'].items() if c == 'correct')}", flush=True)

    def n_pairs(bk):
        la, lb = int(bk[1]), int(bk[2])
        sizes = {1: 10, 2: 90, 3: 900}
        return sizes[la] * (sizes[la] + 1) // 2 if la == lb else sizes[la] * sizes[lb]

    report = {"data": args.data, "tokens": int(n)}
    for form in ("symbolic", "word"):
        rows = {}
        for la, lb in BUCKETS:
            bk = f"b{la}{lb}"
            ok = [k for k in pairs_ok[form] if blen(*k) == bk]
            rows[bk] = {c: stats[form][(bk, c)] for c in ("correct", "wrong", "no_answer")}
            rows[bk]["distinct_pairs_correct"] = len(ok)
            rows[bk]["pair_coverage"] = len(ok) / n_pairs(bk)
        report[form] = rows
        report[form + "_top_pairs"] = [[list(k), v] for k, v in pairs_ok[form].most_common(20)]
    report["held_out"] = {}
    for s, h in held.items():
        sp = np.array(sorted(set(spans[s])), dtype=np.int64).reshape(-1, 2)
        np.save(f"{args.out}_heldspans_seed{s}.npy", sp)
        report["held_out"][str(s)] = {"held_pairs": len(h), "occurrences": int(len(sp))}
    # which held-out pairs occur (per seed)
    for s, h in held.items():
        occurring = {k for k in list(pairs_ok["symbolic"]) + list(pairs_ok["word"]) if k in h}
        report["held_out"][str(s)]["pairs_with_correct_occurrence"] = len(occurring)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    print(json.dumps({f: report[f] for f in ("symbolic", "word")}, indent=1))
    print(json.dumps(report["held_out"], indent=1))


if __name__ == "__main__":
    main()
