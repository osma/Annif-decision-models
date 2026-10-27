#!/usr/bin/env python3
"""Offline diagnostics and blend-formula sweep over dump_scores.py output.

Two subcommands:

    python blend_zoo.py diagnostics scores.csv [--best-b 0.9783 --best-tau 0.4969]
    python blend_zoo.py sweep scores.csv

diagnostics answers the research questions before changing the blend
formula:
  * gold coverage: what fraction of gold subjects are in the candidate
    pool at all (the ceiling no blend can break)
  * AUC of the source score vs gold, and of the noul score vs gold
    (is the model a better discriminator than the source?)
  * NDCG@10 of pure-source and pure-noul rankings
  * correlation between source score and noul (redundant or new info?)
  * "rescuable by a boost": for the golds the currently tuned blend
    misses in its top-10, their noul and source-rank distributions

sweep compares, on the same table, a zoo of 2-parameter blend families:
  * current : score = source * noul**b * sigmoid((noul - tau)/0.1)
  * boost   : score = source * max(0, 1 + gamma*(sigmoid((noul - tau)/0.1) - 0.5))
              factor is exactly 1 at noul = tau; gamma = 0 is the exact
              pure-source point; allows the model to lift or demote.
  * odds    : score = source * ((noul + eps)/(1 - noul + eps))**b
              * sigmoid((noul - tau)/0.1); unbounded boost as noul -> 1;
              (b = 0, tau = -1.0) reproduces the source order to within
              1e-4, as the current formula does.
  * linear  : score = (1 - alpha)*minmax(source) + alpha*noul
              * sigmoid((noul - tau)/0.1); alpha = 0 is pure source.
  * rrf     : score = 1/(k + rank_source) + lambda/(k + rank_noul);
              lambda = 0 is pure source; rank-based and scale-free.

Candidates without a noul score keep their source score (in the rank-
based families they simply do not participate in the model term),
matching the production backend's degradation behaviour.

The winning family must beat `current`'s best on the validation table by
a clear margin AND hold on the held-out test table before it is
considered for the backend.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from metrics import f1_at_k, ndcg

EPS = 0.01  # odds-ratio damping
WIDTH = 0.1  # sigmoid gate width (fixed, as in the backend)
K10 = 10
K5 = 5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(path: str) -> list[pd.DataFrame]:
    """Read one dump CSV and return a list of per-document DataFrames with
    columns: subject_id, label, source (float), noul (float, NaN if the
    question was not asked), gold (0/1 int)."""
    df = pd.read_csv(path)
    df["noul"] = pd.to_numeric(df["noul_score"], errors="coerce")
    docs = [
        g[["subject_id", "label", "source_score", "noul", "gold"]]
        .rename(columns={"source_score": "source"})
        .reset_index(drop=True)
        for _, g in df.groupby("doc_id", sort=True)
    ]
    return docs


# ---------------------------------------------------------------------------
# Metrics over documents
# ---------------------------------------------------------------------------


def ndcg_at_k(relevant: list[bool], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(relevant[:k]) if rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(sum(relevant), k)))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def rank_list(docs: list[pd.DataFrame], scores: list[np.ndarray]) -> list[list[bool]]:
    """For every document, the gold-relevance flags in the order given by
    the corresponding score array."""
    out = []
    for df, s in zip(docs, scores):
        order = np.argsort(-s, kind="stable")
        out.append([bool(g) for g in df["gold"].to_numpy()[order]])
    return out


def evaluate(docs: list[pd.DataFrame], scores: list[np.ndarray]) -> dict[str, float]:
    """Aggregate the ranking given by the per-document scores into the
    standard metric set used throughout the project."""
    ranked = rank_list(docs, scores)
    ndcg_full = float(np.mean([ndcg(r) for r in ranked]))
    ndcg10 = float(np.mean([ndcg_at_k(r, K10) for r in ranked]))
    f15 = float(np.mean([f1_at_k(r, K5) for r in ranked]))
    p1 = float(np.mean([r[0] if r else 0.0 for r in ranked]))
    return {
        "NDCG": round(ndcg_full, 4),
        "NDCG@10": round(ndcg10, 4),
        "F1@5": round(f15, 4),
        "P@1": round(p1, 4),
    }


# ---------------------------------------------------------------------------
# Blend families. Each takes the document list and returns one score array
# per document, aligned with the document's rows.
# ---------------------------------------------------------------------------


def _sig(u: np.ndarray, tau: float) -> np.ndarray:
    """The sigmoid gate, NaN-safe (NaN in -> NaN out)."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-(u - tau) / WIDTH))


def _arrays(docs: list[pd.DataFrame]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(d["source"].to_numpy(), d["noul"].to_numpy()) for d in docs]


def _keep_source_where_nan(s: np.ndarray, u: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Candidates without a noul score keep their plain source score."""
    return np.where(np.isnan(u), s, sc)


def blend_current(docs, b: float, tau: float) -> list[np.ndarray]:
    out = []
    for s, u in _arrays(docs):
        sc = _keep_source_where_nan(s, u, s * np.power(u, b) * _sig(u, tau))
        out.append(sc)
    return out


def blend_boost(docs, gamma: float, tau: float) -> list[np.ndarray]:
    out = []
    for s, u in _arrays(docs):
        factor = 1.0 + gamma * (_sig(u, tau) - 0.5)
        sc = _keep_source_where_nan(s, u, s * np.clip(factor, 0.0, None))
        out.append(sc)
    return out


def blend_odds(docs, b: float, tau: float) -> list[np.ndarray]:
    out = []
    for s, u in _arrays(docs):
        odds = (u + EPS) / (1.0 - u + EPS)
        sc = _keep_source_where_nan(s, u, s * np.power(odds, b) * _sig(u, tau))
        out.append(sc)
    return out


def blend_linear(docs, alpha: float, tau: float) -> list[np.ndarray]:
    out = []
    for s, u in _arrays(docs):
        lo, hi = s.min(), s.max()
        norm = 0.5 * np.ones_like(s) if hi == lo else (s - lo) / (hi - lo)
        gate = _sig(u, tau)
        model = u * gate
        sc = _keep_source_where_nan(s, u, (1.0 - alpha) * norm + alpha * model)
        out.append(sc)
    return out


def blend_rrf(docs, k: int, lam: float) -> list[np.ndarray]:
    out = []
    for s, u in _arrays(docs):
        n = len(s)
        rank_s = np.empty(n, dtype=float)
        rank_s[np.argsort(-s, kind="stable")] = np.arange(1, n + 1)
        if lam == 0 or np.isnan(u).all():
            out.append(1.0 / (k + rank_s))
            continue
        mask = ~np.isnan(u)
        rank_u = np.full(n, float(n + 1))  # missing noul: no participation
        order = np.argsort(-np.where(mask, u, -np.inf), kind="stable")
        for rank, idx in enumerate(order, start=1):
            rank_u[idx] = rank
        score = 1.0 / (k + rank_s) + lam * np.where(mask, 1.0 / (k + rank_u), 0.0)
        out.append(score)
    return out


def pure_source(docs) -> list[np.ndarray]:
    return [d["source"].to_numpy() for d in docs]


def pure_noul(docs) -> list[np.ndarray]:
    return [d["noul"].to_numpy() for d in docs]


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def auc(scores: np.ndarray, gold: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC of scores vs binary gold for ONE
    document's candidates. NaN when the doc has no positive or no
    negative candidate (not scorable)."""
    n_pos = int(gold.sum())
    if n_pos == 0 or n_pos == len(gold):
        return float("nan")
    # ascending ranks: the lowest score gets rank 1 (Mann-Whitney
    # convention); a higher-scoring gold yields a larger rank and a
    # higher AUC
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average tie ranks
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sums = np.zeros(len(counts))
    np.add.at(tie_sums, inv, ranks)
    avg_rank = tie_sums[inv] / counts[inv]
    sum_pos = avg_rank[gold == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * (len(gold) - n_pos))


def mean_auc_per_doc(docs: list[pd.DataFrame], col: str) -> float:
    """Mean over documents of the per-document AUC of ``col`` vs gold
    (unscorable documents are skipped). For the noul column, candidates
    with a missing noul are dropped from that document before scoring."""
    vals = []
    for d in docs:
        s = d[col].to_numpy()
        g = d["gold"].to_numpy()
        if col == "noul":
            m = ~np.isnan(s)
            s, g = s[m], g[m]
        a = auc(s, g)
        if not math.isnan(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def cmd_diagnostics(args) -> None:
    docs = load(args.scores)
    n_docs = len(docs)

    pool_ids = set()
    for d in docs:
        pool_ids.update(d["subject_id"])
    total_golds = int(sum(int(d["gold"].sum()) for d in docs))
    covered = int(
        sum(
            len(set(d.loc[d["gold"] == 1, "subject_id"]) & set(d["subject_id"]))
            for d in docs
        )
    )
    print(
        f"documents: {n_docs}, candidates: {sum(len(d) for d in docs)}, "
        f"distinct pool subjects: {len(pool_ids)}"
    )
    print(
        f"gold instances: {total_golds}, in pool (coverage): {covered} "
        f"({100.0 * covered / total_golds:.1f}%)"
    )

    # AUC is a rank statistic within a document's candidate list, so it is
    # computed per document and averaged; pooling all candidates across
    # documents would mix the score scales.
    print(f"\nper-doc AUC source score vs gold: {mean_auc_per_doc(docs, 'source'):.4f}")
    print(f"per-doc AUC noul vs gold:          {mean_auc_per_doc(docs, 'noul'):.4f}")
    all_s = np.concatenate([d["source"].to_numpy() for d in docs])
    all_u = np.concatenate([d["noul"].to_numpy() for d in docs])
    m = ~np.isnan(all_u)
    corr = np.corrcoef(all_s[m], all_u[m])[0, 1]
    print(f"Pearson correlation source vs noul: {corr:.4f}")

    print("\nbaseline rankings:")
    for name, fn in (("source", pure_source), ("noul", pure_noul)):
        print(f"  {name:8s} {evaluate(docs, fn(docs))}")

    b, tau = args.best_b, args.best_tau
    best = evaluate(docs, blend_current(docs, b, tau))
    print(f"  current  b={b}, tau={tau}  {best}")

    # rescuable-by-boost analysis under the current tuned blend
    print(f"\nrescuable-by-boost analysis (current blend b={b}, tau={tau}):")
    missed = []  # (noul, source_rank) for golds in pool but not in top-10
    for d in docs:
        s = d["source"].to_numpy()
        u = d["noul"].to_numpy()
        g = d["gold"].to_numpy()
        blended = blend_current([d], b, tau)[0]
        top10 = set(np.argsort(-blended, kind="stable")[:K10].tolist())
        src_order = np.argsort(-s, kind="stable")
        src_rank = np.empty(len(s), dtype=int)
        src_rank[src_order] = np.arange(1, len(s) + 1)
        for i in np.where(g == 1)[0]:
            if i not in top10:
                missed.append((u[i], int(src_rank[i])))
    if not missed:
        print("  no in-pool golds missed in top-10")
        return
    nouls = np.array([n for n, _ in missed])
    ranks = np.array([r for _, r in missed])
    with_noul = ~np.isnan(nouls)
    print(f"  in-pool golds missed in top-10: {len(missed)}")
    print(
        f"    source rank: mean {ranks.mean():.1f}, "
        f"median {np.median(ranks):.0f}, max {ranks.max()}"
    )
    print(
        f"    with noul: {int(with_noul.sum())}; "
        f"noul >= 0.5: {int((nouls[with_noul] >= 0.5).sum())}, "
        f"noul >= 0.7: {int((nouls[with_noul] >= 0.7).sum())}, "
        f"mean noul {np.nanmean(nouls):.3f}"
    )
    strong = with_noul & (nouls >= 0.5)
    if strong.any():
        print(
            f"    of those with noul >= 0.5: {int(strong.sum())} have "
            f"source rank > {K10} (boost-relevant)"
        )


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def _sweep_grid(
    docs: list[pd.DataFrame], name: str, grid: list[tuple[dict, callable]]
) -> list[tuple[float, dict, dict]]:
    """Run each (params, blend_fn) pair; return (NDCG@10, params, metrics)
    sorted best-first."""
    results = []
    for params, fn in grid:
        m = evaluate(docs, fn(docs, **params))
        results.append((m["NDCG@10"], params, m))
    results.sort(key=lambda r: -r[0])
    print(f"\n[{name}] top 5 of {len(results)} trials (NDCG@10):")
    for score, params, m in results[:5]:
        print(
            f"  {score:.4f}  {params}   "
            f"NDCG={m['NDCG']} F1@5={m['F1@5']} P@1={m['P@1']}"
        )
    return results


def cmd_sweep(args) -> None:
    docs = load(args.scores)
    print(f"loaded {len(docs)} documents from {args.scores}")

    print("\nreference points:")
    for name, fn in (("source", pure_source), ("noul", pure_noul)):
        print(f"  {name:8s} {evaluate(docs, fn(docs))}")

    grids = [
        (
            "current",
            [
                (
                    {"b": round(float(b), 3), "tau": round(float(t), 3)},
                    lambda docs, b, tau: blend_current(docs, b, tau),
                )
                for b in np.arange(0.0, 2.0 + 1e-9, 0.05)
                for t in np.arange(-1.0, 0.5 + 1e-9, 0.05)
            ],
        ),
        (
            "boost",
            [
                (
                    {"gamma": round(float(g), 3), "tau": round(float(t), 3)},
                    lambda docs, gamma, tau: blend_boost(docs, gamma, tau),
                )
                for g in np.arange(0.0, 4.0 + 1e-9, 0.1)
                for t in np.arange(0.0, 0.5 + 1e-9, 0.02)
            ],
        ),
        (
            "odds",
            [
                (
                    {"b": round(float(b), 3), "tau": round(float(t), 3)},
                    lambda docs, b, tau: blend_odds(docs, b, tau),
                )
                for b in np.arange(0.0, 2.0 + 1e-9, 0.05)
                for t in np.arange(-1.0, 0.5 + 1e-9, 0.05)
            ],
        ),
        (
            "linear",
            [
                (
                    {"alpha": round(float(a), 3), "tau": round(float(t), 3)},
                    lambda docs, alpha, tau: blend_linear(docs, alpha, tau),
                )
                for a in np.arange(0.0, 1.0 + 1e-9, 0.02)
                for t in np.arange(-1.0, 0.5 + 1e-9, 0.05)
            ],
        ),
        (
            "rrf",
            [
                (
                    {"k": int(k), "w": round(float(l), 3)},
                    lambda docs, k, w: blend_rrf(docs, k, w),
                )
                for k in (1, 2, 5, 10, 20, 40)
                for l in np.arange(0.0, 2.0 + 1e-9, 0.05)
            ],
        ),
    ]

    all_best = {}
    for name, grid in grids:
        all_best[name] = _sweep_grid(docs, name, grid)

    print("\n=== summary: best of each family ===")
    for name, res in all_best.items():
        score, params, m = res[0]
        print(f"  {name:8s} {score:.4f}  {params}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline blend diagnostics and sweep")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_diag = sub.add_parser("diagnostics")
    p_diag.add_argument("scores", help="dump_scores.py output CSV")
    p_diag.add_argument(
        "--best-b",
        type=float,
        default=0.9783,
        help="currently tuned model-strength (for the rescuable analysis)",
    )
    p_diag.add_argument(
        "--best-tau",
        type=float,
        default=0.4969,
        help="currently tuned gate-threshold (for the rescuable analysis)",
    )
    p_diag.set_defaults(func=cmd_diagnostics)

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("scores", help="dump_scores.py output CSV")
    p_sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
