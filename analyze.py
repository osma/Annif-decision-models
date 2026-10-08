#!/usr/bin/env python3
"""Phase 2 analysis: scatter Annif score vs. decision-model score.

Reads a single rerank CSV (produced by rerank.py) and draws a scatterplot of
the Annif suggestion score (x) against the decision model's noul probability
(y). Subjects are colored by their relationship to the gold standard:

  * red    — Annif suggestion that IS a gold-standard subject (true positive)
  * grey   — Annif suggestion that is NOT in the gold standard
  * purple — gold subject that was NOT in the Annif list ("gold_extra");
             these have no Annif score, so they are plotted at a fixed x = 0

Per-document ranking quality (full-list NDCG and F1@5) is computed for two
rankings
and shown in a box on the plot:

  * Annif   — the candidate subjects ranked by their Annif score
  * noul    — all candidates (Annif suggestions plus gold_extra) ranked by
              their decision-model noul score

Both use binary gold relevance (a subject is relevant iff it is in the gold
standard). Averages are taken over all documents.

The figure is written to a PNG named after the input CSV (same path,
`.csv` replaced with `.png`).

Usage:
    python analyze.py results/rerank-en-lev-sharp.csv
    python analyze.py results/rerank-en-lev-sharp-state.csv --alpha 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must be set before pyplot is imported
import matplotlib.pyplot as plt
import pandas as pd

from metrics import K, f1_at_k, minmax, ndcg, strat_prod

# Color / style constants.
COLOR_GOLD = "red"
COLOR_NON_GOLD = "grey"
COLOR_GOLD_EXTRA = "purple"
GOLD_EXTRA_X = 0.0  # fixed x for gold_extra (no Annif score)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scatter Annif score vs. decision-model score for a rerank CSV.",
    )
    p.add_argument(
        "csv",
        help="Input rerank CSV (e.g. results/rerank-en-lev-sharp.csv)",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dot transparency, 0..1 (default: %(default)s)",
    )
    p.add_argument(
        "--size", type=float, default=18,
        help="Dot size in points^2 (default: %(default)s)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    in_path = Path(args.csv)
    if not in_path.is_file():
        print(f"Error: input CSV not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = in_path.with_suffix(".png")

    df = pd.read_csv(in_path)

    # Only rows with a usable decision-model score can be plotted.
    df = df.dropna(subset=["noul_score"])
    if df.empty:
        print(f"Error: no rows with a noul_score in {in_path}", file=sys.stderr)
        sys.exit(1)

    in_gold = df["in_gold"].astype(str).str.lower().eq("true")
    is_annif = df["source"] == "annif"
    is_gold_extra = df["source"] == "gold_extra"

    annif_gold = df[is_annif & in_gold]
    annif_nongold = df[is_annif & ~in_gold]
    gold_extra = df[is_gold_extra]

    # Per-document ranking quality for the two rankings (binary gold relevance).
    annif_ndcg, annif_f1, noul_ndcg, noul_f1 = [], [], [], []
    for _, doc in df.groupby("isbn"):
        annif_rows = doc[doc["source"] == "annif"].sort_values(
            "annif_score", ascending=False
        )
        annif_rel = (
            annif_rows["in_gold"].astype(str).str.lower().eq("true").tolist()
        )
        annif_ndcg.append(ndcg(annif_rel))
        annif_f1.append(f1_at_k(annif_rel, K))

        noul_rows = doc.sort_values("noul_score", ascending=False)
        noul_rel = (
            noul_rows["in_gold"].astype(str).str.lower().eq("true").tolist()
        )
        noul_ndcg.append(ndcg(noul_rel))
        noul_f1.append(f1_at_k(noul_rel, K))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Best product blend norm(annif) * noul^beta (swept over beta), blend.py.
    beta_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    prod_scores = []
    for beta in beta_grid:
        p_ndcg, p_f1 = [], []
        for _, doc in df.groupby("isbn"):
            doc = doc[doc["source"] == "annif"]
            if doc.empty:
                continue
            order = strat_prod(doc, beta)
            rel = [bool(doc.at[i, "in_gold"]) for i in order]
            p_ndcg.append(ndcg(rel))
            p_f1.append(f1_at_k(rel, K))
        prod_scores.append((beta, mean(p_ndcg), mean(p_f1)))
    best_beta, prod_ndcg, prod_f1 = max(prod_scores, key=lambda t: t[1])

    n_docs = len(df["isbn"].unique())
    metrics_text = (
        f"Per-doc mean over {n_docs} docs (binary gold relevance)\n"
        f"{'rank by':<8}{'NDCG':>10}{'F1@5':>10}\n"
        f"{'Annif':<8}{mean(annif_ndcg):>10.3f}{mean(annif_f1):>10.3f}\n"
        f"{'noul':<8}{mean(noul_ndcg):>10.3f}{mean(noul_f1):>10.3f}\n"
        f"{'prod':<8}{prod_ndcg:>10.3f}{prod_f1:>10.3f}\n"
        f"  (prod = beta={best_beta:g})"
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw non-gold first so the "signal" points sit on top.
    if len(annif_nongold):
        ax.scatter(
            annif_nongold["annif_score"], annif_nongold["noul_score"],
            c=COLOR_NON_GOLD, alpha=args.alpha, s=args.size,
            label="Annif suggestion, not gold",
        )
    if len(annif_gold):
        ax.scatter(
            annif_gold["annif_score"], annif_gold["noul_score"],
            c=COLOR_GOLD, alpha=args.alpha, s=args.size,
            label="Annif suggestion, gold",
        )
    if len(gold_extra):
        ax.scatter(
            [GOLD_EXTRA_X] * len(gold_extra), gold_extra["noul_score"],
            c=COLOR_GOLD_EXTRA, alpha=args.alpha, s=args.size,
            label="Gold, not suggested (no Annif score, x=0)",
        )

    ax.set_xlabel("Annif score")
    ax.set_ylabel("Decision model score (noul)")
    ax.set_title(in_path.stem)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(
        0.99, 0.99, metrics_text,
        transform=ax.transAxes,
        ha="right", va="top",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(
        f"Wrote {out_path} "
        f"({len(annif_gold)} gold, {len(annif_nongold)} non-gold, "
        f"{len(gold_extra)} gold-extra points)",
        file=sys.stderr,
    )
    print(
        f"  Annif  NDCG={mean(annif_ndcg):.3f}  F1@5={mean(annif_f1):.3f}\n"
        f"  noul   NDCG={mean(noul_ndcg):.3f}  F1@5={mean(noul_f1):.3f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
