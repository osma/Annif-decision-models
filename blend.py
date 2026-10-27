#!/usr/bin/env python3
"""Experiment: combining Annif scores with decision-model (noul) scores.

Reads one or more rerank CSVs (produced by rerank.py) and, per document,
compares several ways of ranking the Annif-suggested candidates (the subjects
Annif actually produced, each of which also has a noul score) against a set
of baselines. For every method we report the mean NDCG (full list) and F1@5
over all documents, using binary gold relevance.

Baselines
    Annif   ranked by annif_score
    Noul    ranked by noul_score

Combining strategies (each has a swept hyperparameter)
    gate    keep candidates with noul >= gate, then rank survivors by annif_score
            (excluded ones go to the bottom, in annif order)        [gate]
    geom    score = norm(annif)^a * norm(noul)^(1-a), a = annif weight [a]
    linear  score = alpha*norm(annif) + (1-alpha)*norm(noul)          [alpha]
    prod    score = norm(annif) * noul^beta                           [beta]
    rrf     score = 1/(k + rank_annif) + 1/(k + rank_noul)            [k]
    odds    the dm_rerank backend formula: norm(annif) *
            ((noul+eps)/(1-noul+eps))^b * sigmoid((noul-tau)/0.1),
            squashed by x/(1+x); b > 1 can boost (see metrics.strat_odds)
                                                                             [b]

norm() is per-document min-max (a degenerate set maps to 0.5, matching
clm.py). The best hyperparameter per strategy is the one with the highest
mean NDCG.

Usage:
    python blend.py results/rerank-en-lev-sharp.csv results/rerank-fi-lev-sharp.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from metrics import (
    K,
    f1_at_k,
    ndcg,
    rank_by,
    strat_gate,
    strat_geom,
    strat_linear,
    strat_odds,
    strat_prod,
    strat_rrf,
)


# ---------------------------------------------------------------------------
# Metric aggregation over documents
# ---------------------------------------------------------------------------


def evaluate(docs: list[pd.DataFrame], rank_fn) -> tuple[float, float]:
    """Return (mean NDCG, mean F1@5) across docs for a ranking function."""
    ndcgs, f1s = [], []
    for df in docs:
        if df.empty:
            continue
        order = rank_fn(df)
        rel = [bool(df.at[i, "in_gold"]) for i in order]
        ndcgs.append(ndcg(rel))
        f1s.append(f1_at_k(rel, K))
    m_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    m_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return m_ndcg, m_f1


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_docs(csv_path: Path) -> list[pd.DataFrame]:
    """Return a list of per-document DataFrames of Annif-sourced candidates
    that have a noul score (the rerankable base list)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["noul_score"])
    df = df[df["source"] == "annif"].copy()
    df["in_gold"] = df["in_gold"].astype(str).str.lower().eq("true")
    docs = [g for _, g in df.groupby("isbn")]
    return docs


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------


def build_methods():
    """Yield (label, rank_fn_factory). Each factory takes a per-doc df and
    returns a ranking function (no extra params needed for the sweep loop —
    the swept parameter is baked in per row below)."""
    methods = [
        ("Annif (base)", lambda g: rank_by(g, "annif_score")),
        ("Noul   (base)", lambda g: rank_by(g, "noul_score")),
    ]
    return methods


def sweep(csv_path: Path) -> None:
    docs = load_docs(csv_path)
    n = len(docs)
    if not docs:
        print(f"No annif candidates with noul_score in {csv_path}", file=sys.stderr)
        return

    # Baselines (no param).
    results = []
    for label, fn in build_methods():
        ndc, f1 = evaluate(docs, fn)
        results.append((label, "—", ndc, f1))

    # Strategies with a swept parameter.
    def add(label, param_name, values, factory):
        rows = []
        for v in values:
            ndc, f1 = evaluate(docs, lambda g, f=factory, pv=v: f(g, pv))
            rows.append((label, f"{param_name}={v:g}", ndc, f1))
        results.extend(rows)

    add("1. gate  ", "gate", [0.3, 0.35, 0.4, 0.45, 0.5], strat_gate)
    add("2. geom  ", "a", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], strat_geom)
    add("3. linear", "alpha", [0.8, 0.85, 0.87, 0.9, 0.92, 0.95], strat_linear)
    add("4. prod  ", "beta", [0.1, 0.2, 0.3, 0.4, 0.5], strat_prod)
    add("5. rrf   ", "k", [1, 3, 5, 10, 20, 30], strat_rrf)
    # gate off (tau=-1.0); b covers the demote-only (b<1) and boost (b>1)
    # regimes of the odds-ratio term
    add(
        "6. odds  ",
        "b",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.45, 1.5, 1.7, 2.0],
        lambda g, b: strat_odds(g, b),
    )

    # Print the table for this dataset.
    print(f"\n{csv_path.name}   ({n} docs)")
    print(f"{'method':<12}{'param':<10}{'NDCG':>8}{'F1@5':>8}")
    print("-" * 38)
    # Group: keep baselines first, then strategies; highlight best param per strategy.
    best_by_group: dict[str, int] = {}
    for i, (_label, _p, ndc, _f1) in enumerate(results):
        group = _label.split(".")[0].strip() if _label[0].isdigit() else _label
        if group not in best_by_group or ndc > best_by_group[group][1]:
            best_by_group[group] = (i, ndc)

    for i, (label, param, ndc, f1) in enumerate(results):
        group = label.split(".")[0].strip() if label[0].isdigit() else label
        star = " *" if best_by_group[group][0] == i else ""
        print(f"{label:<12}{param:<10}{ndc:>8.3f}{f1:>8.3f}{star}")
    print("  (* = best NDCG within that method)")

    # Return the best row per strategy (by NDCG) for the cross-dataset summary.
    summary = []
    for i, (label, param, ndc, f1) in enumerate(results):
        group = label.split(".")[0].strip() if label[0].isdigit() else label
        if best_by_group[group][0] == i:
            name = label.strip()
            if name.startswith(("1.", "2.", "3.", "4.", "5.", "6.")):
                name = name[2:].strip()
            summary.append((name, param, ndc, f1))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Compare Annif+noul score-combining strategies.",
    )
    p.add_argument("csv", nargs="+", help="One or more rerank CSVs")
    args = p.parse_args(argv)

    per_dataset_best = {}
    for name in args.csv:
        path = Path(name)
        if not path.is_file():
            print(f"Error: not found: {name}", file=sys.stderr)
            sys.exit(1)
        per_dataset_best[path.name] = sweep(path)

    # Cross-dataset "best per strategy" summary.
    print("\n=== Best configuration per method (by NDCG) ===")
    print(f"{'method':<12}{'param':<10}")
    for ds in args.csv:
        print(f"\n{Path(ds).name}")
        print(f"{'method':<12}{'param':<10}{'NDCG':>8}{'F1@5':>8}")
        for name, param, ndc, f1 in per_dataset_best[Path(ds).name]:
            print(f"{name:<12}{param:<10}{ndc:>8.3f}{f1:>8.3f}")


if __name__ == "__main__":
    main()
