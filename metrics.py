"""Shared ranking metrics and score-combining helpers for analyze.py / blend.py.

Kept free of any plotting code so it can be imported from both scripts without
circular imports.
"""

from __future__ import annotations

import math

import pandas as pd

K = 5  # cut-off for F1@5


# ---------------------------------------------------------------------------
# Metrics (per document, binary gold relevance)
# ---------------------------------------------------------------------------


def ndcg(relevant: list[bool]) -> float:
    """NDCG over the whole ranked list, binary relevance.

    ``relevant`` is ordered best-first: ``relevant[i]`` is True iff the item at
    rank i (1-based) is a gold subject. The DCG is computed over the entire
    list (no @k cutoff). Returns 0.0 when there is no relevant item at all.
    """
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(relevant) if rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(sum(relevant)))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def f1_at_k(relevant: list[bool], k: int) -> float:
    """F1@k for a ranked list, binary relevance.

    ``relevant[i]`` is True iff the item at rank i (1-based) is a gold subject.
    Only the top ``k`` are considered as predictions.
    """
    top = relevant[:k]
    tp = sum(top)
    gold_total = sum(relevant)
    precision = tp / len(top) if top else 0.0
    recall = tp / gold_total if gold_total else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Per-document score helpers
# ---------------------------------------------------------------------------


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def ranks_desc(values: list[float]) -> list[int]:
    """1-based rank of each element in descending order (ties share the min
    rank), i.e. ranks_desc([3,1,2]) == [1,3,2]."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    out = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        out[idx] = rank
    return out


# ---------------------------------------------------------------------------
# Ranking strategies. Each takes a per-document DataFrame (annif-sourced rows
# with annif_score and noul_score columns) and returns the candidate order as
# a list of row indices, best first.
# ---------------------------------------------------------------------------


def rank_by(df: pd.DataFrame, col: str) -> list[int]:
    return df.sort_values(col, ascending=False).index.tolist()


def strat_gate(df: pd.DataFrame, gate: float) -> list[int]:
    keep = df.index[df["noul_score"] >= gate].tolist()
    drop = df.index[df["noul_score"] < gate].tolist()
    keep_sorted = sorted(keep, key=lambda i: df.at[i, "annif_score"], reverse=True)
    drop_sorted = sorted(drop, key=lambda i: df.at[i, "annif_score"], reverse=True)
    return keep_sorted + drop_sorted


def _order_by_score(df: pd.DataFrame, score: list[float]) -> list[int]:
    """Return the df's row indices ordered by score (best first)."""
    idx = list(df.index)
    return [
        idx[i] for i in sorted(range(len(idx)), key=lambda i: score[i], reverse=True)
    ]


def strat_geom(df: pd.DataFrame, a: float) -> list[int]:
    na = minmax(df["annif_score"].tolist())
    nn = minmax(df["noul_score"].tolist())
    score = [
        (na[i] ** a) * (nn[i] ** (1 - a)) if nn[i] > 0 else 0.0 for i in range(len(df))
    ]
    return _order_by_score(df, score)


def strat_linear(df: pd.DataFrame, alpha: float) -> list[int]:
    na = minmax(df["annif_score"].tolist())
    nn = minmax(df["noul_score"].tolist())
    score = [alpha * na[i] + (1 - alpha) * nn[i] for i in range(len(df))]
    return _order_by_score(df, score)


def strat_prod(df: pd.DataFrame, beta: float) -> list[int]:
    na = minmax(df["annif_score"].tolist())
    nn = df["noul_score"].tolist()
    score = [na[i] * (nn[i] ** beta) for i in range(len(df))]
    return _order_by_score(df, score)


def strat_rrf(df: pd.DataFrame, k: float) -> list[int]:
    ra = ranks_desc(df["annif_score"].tolist())
    rn = ranks_desc(df["noul_score"].tolist())
    score = [1.0 / (k + ra[i]) + 1.0 / (k + rn[i]) for i in range(len(df))]
    return _order_by_score(df, score)


# Damping constant of the odds-ratio term, same as the dm_rerank backend.
ODDS_EPS = 0.01


def strat_odds(df: pd.DataFrame, b: float, tau: float = -1.0) -> list[int]:
    """The dm_rerank backend's odds-ratio blend:

        raw   = norm(annif) * ((noul + eps) / (1 - noul + eps))**b
                * sigmoid((noul - tau) / 0.1)
        score = raw / (1 + raw)

    Unlike ``strat_prod`` (``norm(annif) * noul**b``), whose noul factor is
    always <= 1, the odds ratio can exceed 1 for confident "yes" answers,
    so with b > 1 this strategy can *boost* a candidate above its source
    score. tau defaults to -1.0 (gate off), matching the gateless design
    of the other single-parameter strategies. The minmax normalization and
    the squash are both order-preserving, so this reproduces the backend's
    ranking exactly under the prototype's rank-based metrics; the squash
    is kept for fidelity to the backend formula (it matters for
    score-based metrics, where it prevents clamping at 1.0)."""
    na = minmax(df["annif_score"].tolist())
    nn = df["noul_score"].tolist()
    score = []
    for i in range(len(df)):
        noul = nn[i]
        odds = (noul + ODDS_EPS) / (1.0 - noul + ODDS_EPS)
        gate = 1.0 / (1.0 + math.exp(-(noul - tau) / 0.1))
        raw = na[i] * (odds**b) * gate
        score.append(raw / (1.0 + raw) if raw > 0 else 0.0)
    return _order_by_score(df, score)
