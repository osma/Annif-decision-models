#!/usr/bin/env python3
"""Verify the numbers for the README's 'Recommended production blending'
and 'Optuna search space' sections under the odds formula.

1. Fixed-constant averages over all 32 model x lang x config datasets:
   old default (prod beta=0.5) vs. candidate new defaults (odds b=0.2..0.5),
   plus the per-dataset best (oracle) for the gap statement.
2. Search space: on the 20 model x dataset runs used by the old
   verification (5 models, en/fi x sharp/state), does a coarse grid of the
   odds formula (b in [0, 2], tau in a few values) match the per-dataset
   oracle (best of all six blend.py methods)?
"""

from __future__ import annotations

from pathlib import Path

from blend import evaluate, load_docs
from metrics import rank_by, strat_odds, strat_prod
from unified_sweep import EXISTING

RESULTS = Path("results")
MODELS = ["laya", "d1-omni-600M", "d1-3B", "lev", "kev", "clefflash", "bespoke-nimble", "openjev"]
CONFIGS = ["sharp", "sharp-state"]


def per_dataset_best(docs):
    ndc, f1 = evaluate(docs, lambda g: rank_by(g, "annif_score"))
    best = (ndc, f1, "Annif", None)
    for name, _pn, values, fn in EXISTING:
        for v in values:
            r = evaluate(docs, lambda g, fn=fn, v=v: fn(g, v))
            if r[0] > best[0]:
                best = (r[0], r[1], name, v)
    return best


def main() -> None:
    # Part 1: fixed constants over all 32 datasets
    print("Part 1: fixed constant, mean NDCG over 32 datasets")
    for label, fn in (
        ("prod beta=0.5 (old default)", lambda g: strat_prod(g, 0.5)),
        *[(f"odds b={b:g}", (lambda b: lambda g: strat_odds(g, b))(b)) for b in (0.2, 0.3, 0.4, 0.5)],
    ):
        acc = []
        for slug in MODELS:
            for lang in ("en", "fi"):
                for cfg in CONFIGS:
                    docs = load_docs(RESULTS / f"rerank-{lang}-{slug}-{cfg}.csv")
                    acc.append(evaluate(docs, fn)[0])
        print(f"  {label:28s} {sum(acc) / len(acc):.4f}")
    # oracle (per-dataset best of all methods)
    acc = []
    for slug in MODELS:
        for lang in ("en", "fi"):
            for cfg in CONFIGS:
                docs = load_docs(RESULTS / f"rerank-{lang}-{slug}-{cfg}.csv")
                acc.append(per_dataset_best(docs)[0])
    print(f"  {'per-dataset best (oracle)':28s} {sum(acc) / len(acc):.4f}")

    # Part 2: odds grid vs oracle on the 20 runs
    print("\nPart 2: odds (b, tau) grid vs per-dataset oracle, 20 runs")
    grid_models = ["lev", "kev", "clefflash", "openjev", "bespoke-nimble"]
    gaps = []
    for slug in grid_models:
        for lang in ("en", "fi"):
            for cfg in CONFIGS:
                docs = load_docs(RESULTS / f"rerank-{lang}-{slug}-{cfg}.csv")
                oracle = per_dataset_best(docs)[0]
                best = -1.0
                for b in [round(x * 0.1, 1) for x in range(21)]:
                    for tau in (-1.0, -0.25, 0.2, 0.3, 0.4, 0.5):
                        r = evaluate(docs, lambda g, b=b, t=tau: strat_odds(g, b, t))[0]
                        if r > best:
                            best = r
                gaps.append(oracle - best)
                print(f"  {slug:16s} {lang} {cfg:12s} oracle {oracle:.4f}  grid {best:.4f}  gap {oracle - best:+.4f}")
    gaps = sorted(gaps)
    print(f"  mean gap {sum(gaps) / len(gaps):+.4f}  worst {gaps[0]:+.4f}  n={len(gaps)}")


if __name__ == "__main__":
    main()
