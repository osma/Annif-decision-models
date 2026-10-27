#!/usr/bin/env python3
"""Compare blend families under Annif's *actual* metric: per-doc
sklearn dcg_score (score-based NDCG, binary relevance) on the
full-vocabulary score matrix, with scores clamped to 1.0 — exactly what
annif.eval.ndcg_score / SuggestionBatch.from_sequence do.

Two modes:

    # quick comparison of fixed configs
    python annif_ndcg.py scores.csv [scores2.csv ...]

    # fine grid sweep of the product and (squashed) odds families
    python annif_ndcg.py sweep scores.csv [scores2.csv ...]

The clamped-metric comparison is the definitive one: the backend's
ranking only reaches the evaluation (and the API) through clamped [0, 1]
scores, so a blend whose raw scores saturate at 1.0 for many top
candidates is only as good as the tie-breaking that score-based NDCG
applies among the saturated ones (sklearn assigns tied scores the mean
of their ranks, so the ordering *within* a tie group does not count).
The backend therefore maps raw scores through the monotonic squash
x / (1 + x) before returning them.
"""

from __future__ import annotations

import sys

import numpy as _np
from sklearn.metrics import dcg_score

import blend_zoo as bz


def _squash(x: float) -> float:
    """The backend's monotonic [0, 1) squash: x / (1 + x)."""
    if x <= 0.0:
        return 0.0
    return x / (1.0 + x)


def _squashed(scores: list) -> list:
    """Apply the backend squash to every raw score (the from_sequence
    clamp then never triggers)."""
    return [_np.array([_squash(v) for v in s]) for s in scores]


CASES = [
    ("source", lambda d: bz.pure_source(d)),
    ("current prod b=.9783 tau=.4969", lambda d: bz.blend_current(d, 0.9783, 0.4969)),
    ("current valbest b=2.0 tau=.3", lambda d: bz.blend_current(d, 2.0, 0.3)),
    ("odds raw b=1.45 tau=.25", lambda d: bz.blend_odds(d, 1.45, 0.25)),
    ("odds+SQUASH b=1.45 tau=.25", lambda d: _squashed(bz.blend_odds(d, 1.45, 0.25))),
]


# ---------------------------------------------------------------------------
# per-document score matrices (full subject universe)
# ---------------------------------------------------------------------------


def _matrices(docs: list) -> tuple[_np.ndarray, _np.ndarray, list[int]]:
    """Return (y_true, y_raw, scorable): binary gold and raw-source score
    matrices over the full subject universe, and the indices of docs with
    at least one gold (idcg > 0, the only docs that count)."""
    all_ids = sorted({int(x) for d in docs for x in d["subject_id"]})
    idx = {v: i for i, v in enumerate(all_ids)}
    y_true = _np.zeros((len(docs), len(all_ids)))
    y_raw = _np.zeros((len(docs), len(all_ids)))
    scorable = []
    for di, df in enumerate(docs):
        gold = df["gold"].to_numpy()
        if gold.sum() > 0:
            scorable.append(di)
        for i in range(len(df)):
            j = idx[int(df["subject_id"].iloc[i])]
            y_true[di, j] = gold[i]
            y_raw[di, j] = float(df["source"].iloc[i])
    return y_true, y_raw, scorable


def _ndcg10(y_true: _np.ndarray, y_pred: _np.ndarray) -> float:
    """Annif's NDCG@10, exactly as annif.eval.ndcg_score computes it:
    per-doc dcg/idcg, docs with idcg <= 0 count as 1.0, mean over ALL docs.

    Uses sklearn's per-sample DCG (the primitive annif.eval.ndcg_score
    builds on), so the result is identical to running the metric through
    the Annif backend."""
    from sklearn.metrics._ranking import _dcg_sample_scores

    scores = _np.ones(y_true.shape[0])
    dcg = _dcg_sample_scores(y_true, y_pred, k=10)
    idcg = _dcg_sample_scores(y_true, y_true, k=10)
    for i in range(y_true.shape[0]):
        if idcg[i] > 0:
            scores[i] = dcg[i] / idcg[i]
    return float(scores.mean())


def _scores_matrix(docs: list, fn) -> _np.ndarray:
    """Place per-doc raw scores (one list per doc, aligned with the doc's
    rows) into the full-universe matrix."""
    all_ids = sorted({int(x) for d in docs for x in d["subject_id"]})
    idx = {v: i for i, v in enumerate(all_ids)}
    scores = fn(docs)
    y_pred = _np.zeros((len(docs), len(all_ids)))
    for di, df in enumerate(docs):
        for i in range(len(df)):
            y_pred[di, idx[int(df["subject_id"].iloc[i])]] = float(scores[di][i])
    return y_pred


def _clamp1(y_pred: _np.ndarray) -> _np.ndarray:
    return _np.minimum(y_pred, 1.0)


def _squash_mat(y_pred: _np.ndarray) -> _np.ndarray:
    return y_pred / (1.0 + y_pred)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_compare(paths: list[str]) -> None:
    for path in paths:
        docs = bz.load(path)
        y_true, _, _ = _matrices(docs)
        print(f"--- {path} ({len(docs)} docs) ---")
        for name, fn in CASES:
            y_pred = _clamp1(_scores_matrix(docs, fn))
            print(f"  {name:34s} NDCG@10 = {_ndcg10(y_true, y_pred):.4f}")
        print()


def cmd_sweep(paths: list[str]) -> None:
    for path in paths:
        docs = bz.load(path)
        y_true, y_raw, _ = _matrices(docs)
        print(f"--- {path} ({len(docs)} docs) ---")
        print(f"  reference: source  {_ndcg10(y_true, _clamp1(y_raw)):.4f}")
        ref = _clamp1(
            _scores_matrix(docs, lambda d: bz.blend_current(d, 0.9783, 0.4969))
        )
        print(
            "  reference: current prod b=.9783 tau=.4969  "
            f"{_ndcg10(y_true, ref):.4f}"
        )
        for name, fn, use_squash in (
            ("product", bz.blend_current, False),
            ("odds+squash", bz.blend_odds, True),
            ("odds(raw, old)", bz.blend_odds, False),
        ):
            best = (0.0, None)
            for b in _np.arange(0.0, 2.0 + 1e-9, 0.05):
                for t in _np.arange(-1.0, 0.5 + 1e-9, 0.05):
                    y_pred = _scores_matrix(docs, lambda d: fn(d, float(b), float(t)))
                    if use_squash:
                        y_pred = _squash_mat(y_pred)
                    m = _ndcg10(y_true, _clamp1(y_pred))
                    if m > best[0]:
                        best = (m, (round(float(b), 2), round(float(t), 2)))
            print(
                f"  {name:16s} best {best[0]:.4f}  " f"b={best[1][0]} tau={best[1][1]}"
            )
        print()


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "sweep":
        cmd_sweep(args[1:])
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
