#!/usr/bin/env python3
"""Regenerate the README's Findings tables (raw noul scores + best blend per
model, now including the odds strategy) from the committed result CSVs.

Usage:
    python regen_tables.py

Prints markdown tables ready to paste into README.md.
"""

from __future__ import annotations

from pathlib import Path

from blend import evaluate, load_docs
from metrics import rank_by, strat_odds
from unified_sweep import EXISTING

import numpy as np

RESULTS = Path("results")

MODELS = [
    ("Laya (421M BF16)", "laya"),
    ("d1-omni-600M (600M BF16)", "d1-omni-600M"),
    ("d1-3B (3B Q8_0)", "d1-3B"),
    ("lev (4B Q8_0)", "lev"),
    ("Kev-4B (4B Q8_0)", "kev"),
    ("Clef-Flash (9B Q4_K_M)", "clefflash"),
    ("Bespoke-Nimble (9B Q4_K_M)", "bespoke-nimble"),
    ("OpenJev (27B Q4_K_M)", "openjev"),
    ("H2O-Lightning (4B Q8_0)", "h2o-lightning"),
    ("vllm-d3-mini (4B Q8_0)", "vllm-d3-mini"),
]

CONFIGS = ["sharp", "sharp-state"]

_docs_cache: dict[Path, list] = {}


def load(path: str) -> list:
    p = RESULTS / path
    if p not in _docs_cache:
        _docs_cache[p] = load_docs(p)
    return _docs_cache[p]


def best_blend(docs):
    """Return (ndcg, f1, method, param) of the best method (incl. odds),
    or the Annif baseline when the model adds nothing."""
    ndc, f1 = evaluate(docs, lambda g: rank_by(g, "annif_score"))
    best = (ndc, f1, "Annif", "—")
    for name, _pname, values, fn in EXISTING:
        for v in values:
            r = evaluate(docs, lambda g, fn=fn, v=v: fn(g, v))
            if r[0] > best[0]:
                best = (r[0], r[1], name, v)
    return best


def main() -> None:
    # --- raw noul table ---
    print("**Raw model scores (NDCG / F1@5):**\n")
    print("| model | en sharp | en sharp+rules | fi sharp | fi sharp+rules |")
    print("|---|---|---|---|---|")
    print(
        "| Annif (baseline, no model) | 0.796 / 0.502 | 0.796 / 0.502 "
        "| 0.804 / 0.485 | 0.804 / 0.485 |"
    )
    all_cells = {}
    for label, slug in MODELS:
        cells = []
        for lang in ("en", "fi"):
            for cfg in CONFIGS:
                docs = load(f"rerank-{lang}-{slug}-{cfg}.csv")
                ndc, f1 = evaluate(docs, lambda g: rank_by(g, "noul_score"))
                cells.append((f"{ndc:.3f} / {f1:.3f}", ndc))
        all_cells[label] = cells
    top = max(ndc for cells in all_cells.values() for _, ndc in cells)
    for label, cells in all_cells.items():
        out = [f"**{text}**" if ndc == top else text for text, ndc in cells]
        print(f"| {label} | {out[0]} | {out[1]} | {out[2]} | {out[3]} |")

    # --- best blend table (incl. odds), best across the two configs ---
    print(
        "\n**Best blend per model (vs Annif: en 0.796 / 0.502, fi 0.804 / 0.485):**\n"
    )
    print("| model | en best blend | fi best blend |")
    print("|---|---|---|")
    all_cells = {}
    for label, slug in MODELS:
        cells = []
        for lang in ("en", "fi"):
            best = None
            for cfg in CONFIGS:
                docs = load(f"rerank-{lang}-{slug}-{cfg}.csv")
                r = best_blend(docs)
                if best is None or r[0] > best[0]:
                    best = r
            ndc, f1, method, v = best
            param = f"{v:g}" if isinstance(v, float) else v
            cell = f"{ndc:.3f} / {f1:.3f} ({method} {param})"
            if method == "Annif":
                cell = f"{ndc:.3f} / {f1:.3f} (≈Annif)"
            cells.append((cell, ndc))
        all_cells[label] = cells
    top = max(ndc for cells in all_cells.values() for _, ndc in cells)
    for label, cells in all_cells.items():
        out = [f"**{text}**" if ndc == top else text for text, ndc in cells]
        print(f"| {label} | {out[0]} | {out[1]} |")

    # --- per-language best (for the per-language selection recommendation) ---
    print("\nPer-language best (either config):")
    for lang in ("en", "fi"):
        best = (0, 0, "")
        for _label, slug in MODELS:
            for cfg in CONFIGS:
                docs = load(f"rerank-{lang}-{slug}-{cfg}.csv")
                r = best_blend(docs)
                if r[0] > best[0]:
                    best = (r[0], r[1], f"{slug} {cfg} ({r[2]} {r[3]})")
        print(f"  {lang}: {best[0]:.3f} / {best[1]:.3f}  {best[2]}")

    # --- win tally: which method is best on how many of the datasets ---
    from collections import Counter

    n_datasets = len(MODELS) * 2 * len(CONFIGS)
    tally: Counter = Counter()
    for _label, slug in MODELS:
        for lang in ("en", "fi"):
            for cfg in CONFIGS:
                docs = load(f"rerank-{lang}-{slug}-{cfg}.csv")
                _ndc, _f1, method, _v = best_blend(docs)
                tally[method] += 1
    print(
        f"\nBest-method win tally over the {n_datasets} "
        f"model x lang x config datasets:"
    )
    for method, n in tally.most_common():
        print(f"  {method:8s} {n}/{n_datasets}")

    # --- fixed-b average for the production recommendation ---
    print(
        f"\nFixed odds-b (no per-dataset tuning), "
        f"mean NDCG over the {n_datasets} datasets:"
    )
    for b in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        acc = []
        for _label, slug in MODELS:
            for lang in ("en", "fi"):
                for cfg in CONFIGS:
                    docs = load(f"rerank-{lang}-{slug}-{cfg}.csv")
                    acc.append(evaluate(docs, lambda g, b=b: strat_odds(g, b))[0])
        print(f"  odds b={b:g}: {np.mean(acc):.3f}")


if __name__ == "__main__":
    main()
