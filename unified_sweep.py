#!/usr/bin/env python3
"""Unified gated-power-blend (GPB) experiment.

Question: can one parameterized family beat (or at least match) the best of
the individual blend methods, with parameters chosen by grid search?

Family (per document):

    score = na^a * t^b * gate
    na   = minmax(annif_score)
    t    = lam*minmax(noul_score) + (1 - lam)*noul_score
    gate = 1                    if tau is None
         = 1{noul >= tau}       if s == 0   (hard)
         = sigmoid((noul-tau)/s) if s > 0   (soft)

Limit cases (verified at runtime against the existing strategies):
    b=0, tau=None            -> pure Annif order
    s=0, b=0                 -> strat_gate(tau)
    lam=1, b=1-a, tau=None   -> strat_geom(a)
    a=1, lam=0, tau=None     -> strat_prod(beta=b)

Protocol (to avoid pure overfitting on 50-63 doc datasets):
  1. recompute per-dataset best of the existing methods;
  2. per-dataset best GPB (optimistic upper bound);
  3. SELECT GPB params on en-sharp per model;
  4. evaluate the SAME params on en-state and both fi datasets (transfer);
  5. paired bootstrap of GPB(en-sel) vs best existing on fi-sharp (is it noise?).
"""
import random
from pathlib import Path

import numpy as np

from blend import load_docs, evaluate

# Rerank output CSVs live in results/ (see run_model.sh).
RESULTS = "results"
from metrics import (
    ndcg,
    minmax,
    rank_by,
    strat_gate,
    strat_geom,
    strat_linear,
    strat_odds,
    strat_prod,
    strat_rrf,
)


def strat_gpb(df, a: float, b: float, lam: float, tau, s) -> list[int]:
    na = np.asarray(minmax(df["annif_score"].tolist()), dtype=float)
    nn = np.asarray(minmax(df["noul_score"].tolist()), dtype=float)
    nr = df["noul_score"].to_numpy(dtype=float)
    t = lam * nn + (1.0 - lam) * nr
    if b == 0:
        blend = na**a
    else:
        blend = na**a * np.where(t > 0, t**b, 0.0)
    if tau is None:
        gate = 1.0
    elif s == 0:
        gate = (nr >= tau).astype(float)
    else:
        gate = 1.0 / (1.0 + np.exp(-(nr - tau) / s))
    scores = blend * gate
    idx = list(df.index)
    order = np.argsort(-scores, kind="stable")
    return [idx[i] for i in order]


GATES = [
    (None, None),
    (0.3, 0.0),
    (0.4, 0.0),
    (0.5, 0.0),
    (0.3, 0.15),
    (0.4, 0.15),
    (0.5, 0.15),
]
GRID = [
    (a, b, lam, tau, s)
    for a in (0.85, 1.0)
    for b in (0.0, 0.3, 0.5, 0.8)
    for lam in (0.0, 0.5, 1.0)
    for (tau, s) in GATES
]

EXISTING = [
    ("gate", "gate", [0.3, 0.35, 0.4, 0.45, 0.5], strat_gate),
    ("geom", "a", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], strat_geom),
    ("linear", "alpha", [0.8, 0.85, 0.87, 0.9, 0.92, 0.95], strat_linear),
    ("prod", "beta", [0.1, 0.2, 0.3, 0.4, 0.5], strat_prod),
    ("rrf", "k", [1, 3, 5, 10, 20, 30], strat_rrf),
    # the dm_rerank backend's odds-ratio blend (gate off, tau=-1.0): unlike
    # the other methods its noul factor is not bounded by 1, so b > 1 can
    # boost candidates above their source score
    (
        "odds",
        "b",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.45, 1.5, 1.7, 2.0],
        lambda df, b: strat_odds(df, b),
    ),
]


def fmt_p(p):
    a, b, lam, tau, s = p
    if tau is None:
        g = "off"
    elif s == 0:
        g = f"hard{tau:g}"
    else:
        g = f"soft tau={tau:g} s={s:g}"
    return f"a={a:g} b={b:g} lam={lam:g} gate={g}"


def best_existing(docs):
    """Return (ndcg, f1, method, param_value) of the best existing method."""
    best = (0.0, 0.0, "Annif", "-")
    ndc, f1 = evaluate(docs, lambda g: rank_by(g, "annif_score"))
    if ndc > best[0]:
        best = (ndc, f1, "Annif", "-")
    for name, pname, values, fn in EXISTING:
        for v in values:
            ndc, f1 = evaluate(docs, lambda g, fn=fn, v=v: fn(g, v))
            if ndc > best[0]:
                best = (ndc, f1, name, v)
    return best


def best_gpb(docs):
    best = (0.0, 0.0, None)
    for p in GRID:
        ndc, f1 = evaluate(docs, lambda g, p=p: strat_gpb(g, *p))
        if ndc > best[0]:
            best = (ndc, f1, p)
    return best


def eval_gpb(docs, p):
    return evaluate(docs, lambda g, p=p: strat_gpb(g, *p))


def bootstrap_diff(docs, fn_a, fn_b, n_boot=2000, seed=0):
    """Mean NDCG difference fn_a - fn_b over bootstrap resamples of docs."""
    rng = random.Random(seed)
    n = len(docs)
    diffs = []
    for _ in range(n_boot):
        da = db = 0.0
        for i in (rng.randrange(n) for _ in range(n)):
            df = docs[i]
            da += ndcg([bool(df.at[j, "in_gold"]) for j in fn_a(df)])
            db += ndcg([bool(df.at[j, "in_gold"]) for j in fn_b(df)])
        diffs.append((da - db) / n)
    diffs.sort()
    return (sum(diffs) / n_boot, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)])


MODELS = {
    "lev": {
        "en-sharp": "rerank-en-lev-sharp.csv",
        "en-state": "rerank-en-lev-sharp-state.csv",
        "fi-sharp": "rerank-fi-lev-sharp.csv",
        "fi-state": "rerank-fi-lev-sharp-state.csv",
    },
    "kev": {
        "en-sharp": "rerank-en-kev-sharp.csv",
        "en-state": "rerank-en-kev-sharp-state.csv",
        "fi-sharp": "rerank-fi-kev-sharp.csv",
        "fi-state": "rerank-fi-kev-sharp-state.csv",
    },
    "clefflash": {
        "en-sharp": "rerank-en-clefflash-sharp.csv",
        "en-state": "rerank-en-clefflash-sharp-state.csv",
        "fi-sharp": "rerank-fi-clefflash-sharp.csv",
        "fi-state": "rerank-fi-clefflash-sharp-state.csv",
    },
    "openjev": {
        "en-sharp": "rerank-en-openjev-sharp.csv",
        "en-state": "rerank-en-openjev-sharp-state.csv",
        "fi-sharp": "rerank-fi-openjev-sharp.csv",
        "fi-state": "rerank-fi-openjev-sharp-state.csv",
    },
    "bespoke-nimble": {
        "en-sharp": "rerank-en-bespoke-nimble-sharp.csv",
        "en-state": "rerank-en-bespoke-nimble-sharp-state.csv",
        "fi-sharp": "rerank-fi-bespoke-nimble-sharp.csv",
        "fi-state": "rerank-fi-bespoke-nimble-sharp-state.csv",
    },
    "h2o-lightning": {
        "en-sharp": "rerank-en-h2o-lightning-sharp.csv",
        "en-state": "rerank-en-h2o-lightning-sharp-state.csv",
        "fi-sharp": "rerank-fi-h2o-lightning-sharp.csv",
        "fi-state": "rerank-fi-h2o-lightning-sharp-state.csv",
    },
    "vllm-d3-mini": {
        "en-sharp": "rerank-en-vllm-d3-mini-sharp.csv",
        "en-state": "rerank-en-vllm-d3-mini-sharp-state.csv",
        "fi-sharp": "rerank-fi-vllm-d3-mini-sharp.csv",
        "fi-state": "rerank-fi-vllm-d3-mini-sharp-state.csv",
    },
}


def main():
    print(f"GPB grid: {len(GRID)} combinations\n")

    # --- verify limit cases against the existing strategies ---
    probe = load_docs(Path(RESULTS) / MODELS["openjev"]["en-sharp"])[0]
    checks = [
        (
            "b=0, gate off  == Annif",
            strat_gpb(probe, 0.85, 0.0, 0.5, None, None)
            == rank_by(probe, "annif_score"),
        ),
        (
            "hard, b=0      == gate(0.4)",
            strat_gpb(probe, 0.85, 0.0, 0.5, 0.4, 0.0) == strat_gate(probe, 0.4),
        ),
        (
            "lam=1, b=1-a   == geom(0.65)",
            strat_gpb(probe, 0.65, 0.35, 1.0, None, None) == strat_geom(probe, 0.65),
        ),
        (
            "a=1, lam=0     == prod(0.4)",
            strat_gpb(probe, 1.0, 0.4, 0.0, None, None) == strat_prod(probe, 0.4),
        ),
    ]
    for label, ok in checks:
        print(f"limit check {label}: {'OK' if ok else 'MISMATCH'}")
    print()

    for model, files in MODELS.items():
        docs = {k: load_docs(Path(RESULTS) / v) for k, v in files.items()}
        print(f"=== {model} ===")

        # per-dataset best of existing methods and best GPB (optimistic bound)
        best5 = {k: best_existing(d) for k, d in docs.items()}
        bg = {k: best_gpb(d) for k, d in docs.items()}

        for k in ("en-sharp", "en-state", "fi-sharp", "fi-state"):
            nd5, f5, m5, p5 = best5[k]
            ng, fg, pg = bg[k]
            print(
                f"  {k:9s}  best-existing: {m5:6s} {p5!s:>6}  {nd5:.3f}/{f5:.3f}"
                f"   |  GPB-best: {ng:.3f}/{fg:.3f}   {fmt_p(pg)}"
            )

        # select on en-sharp, transfer to the other three
        sel = bg["en-sharp"][2]
        print(f"  selected on en-sharp: {fmt_p(sel)}")
        for k in ("en-state", "fi-sharp", "fi-state"):
            nd5, f5, m5, _ = best5[k]
            nsel, fsel = eval_gpb(docs[k], sel)
            print(
                f"  transfer {k:9s}  GPB(en-sel) {nsel:.3f}/{fsel:.3f}"
                f"   vs best-existing ({m5}) {nd5:.3f}/{f5:.3f}"
                f"   dNDCG {nsel - nd5:+.3f}"
            )

        # significance on fi-sharp: GPB(en-sel) vs its best existing method
        nd5, f5, m5, p5 = best5["fi-sharp"]
        if m5 == "Annif":
            fn_b = lambda g: rank_by(g, "annif_score")
        else:
            fn_b = lambda g: {n: f for n, _, _, f in EXISTING}[m5](g, p5)
        mean, lo, hi = bootstrap_diff(
            docs["fi-sharp"], lambda g: strat_gpb(g, *sel), fn_b
        )
        print(
            f"  bootstrap fi-sharp (GPB-en-sel - best-existing): "
            f"{mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
        )
        print()


if __name__ == "__main__":
    main()
