#!/usr/bin/env python3
"""Honest cross-validated selection of blend parameters.

For each model x dataset we run 5-fold CV over documents:
  * candidate set = the 5 existing methods (their standard grids) +
    the 168-point GPB family from unified_sweep.py;
  * in each fold, the candidate with the best training-fold mean NDCG is
    selected;
  * the PROTOCOL estimate (the honest number) = mean over folds of the
    held-out performance of the candidate selected in that fold.  This is
    what you would actually get running this selection procedure in
    production; it has no selection bias.
  * also reported: the per-candidate CV estimate (held-out docs of the
    folds where that candidate was selected) and the optimistic
    best-of-grid on the full data (upper bound, selection-biased).

The optimistic - protocol gap estimates how much of the per-dataset
"best-of-grid" numbers was pure sweep noise on 50-63 docs.
"""
import random
from pathlib import Path

from blend import load_docs
from metrics import ndcg, f1_at_k, rank_by
from unified_sweep import (
    GRID,
    EXISTING,
    MODELS,
    strat_gpb,
    fmt_p,
)

NFOLDS = 5
SEED = 0


def build_candidates():
    """Return list of (label, fn) where fn(df) -> ordered index list."""
    cands = [("Annif", lambda g: rank_by(g, "annif_score"))]
    for name, pname, values, fn in EXISTING:
        for v in values:
            cands.append((f"{name} {pname}={v:g}", (lambda g, fn=fn, v=v: fn(g, v))))
    for p in GRID:
        cands.append((f"GPB {fmt_p(p)}", (lambda g, p=p: strat_gpb(g, *p))))
    return cands


def doc_metric(df, order):
    rel = [bool(df.at[i, "in_gold"]) for i in order]
    return ndcg(rel), f1_at_k(rel, 5)


def per_doc_scores(docs, cands):
    """scores[c][d] = (ndcg, f1) of candidate c on doc d."""
    scores = []
    for _label, fn in cands:
        row = []
        for df in docs:
            row.append(doc_metric(df, fn(df)))
        scores.append(row)
    return scores


def cv_select(scores, n_docs, n_folds=NFOLDS, seed=SEED):
    """Run fold selection on precomputed per-doc scores.

    Returns (proto_ndcg, proto_f1, per_fold, sel_counts, held_ndcg,
             held_f1, held_n) where proto_* is the honest protocol
    estimate (mean over folds of the held-out performance of the
    candidate selected in that fold) and per_fold =
    [(sel_c, fold_ndcg, fold_f1), ...].
    """
    rng = random.Random(seed)
    n = n_docs
    idx = list(range(n))
    rng.shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]

    n_cand = len(scores)
    fold_train = []
    for f in range(n_folds):
        held = set(folds[f])
        train = [d for d in range(n) if d not in held]
        fold_train.append(
            [sum(scores[c][d][0] for d in train) / len(train) for c in range(n_cand)]
        )

    sel_counts = [0] * n_cand
    held_ndcg = [0.0] * n_cand
    held_f1 = [0.0] * n_cand
    held_n = [0] * n_cand
    per_fold = []
    for f in range(n_folds):
        best_c = max(range(n_cand), key=lambda c: fold_train[f][c])
        sel_counts[best_c] += 1
        fold_ndcg = fold_f1 = 0.0
        for d in folds[f]:
            held_ndcg[best_c] += scores[best_c][d][0]
            held_f1[best_c] += scores[best_c][d][1]
            held_n[best_c] += 1
            fold_ndcg += scores[best_c][d][0]
            fold_f1 += scores[best_c][d][1]
        per_fold.append((best_c, fold_ndcg / len(folds[f]), fold_f1 / len(folds[f])))
    proto_ndcg = sum(x[1] for x in per_fold) / n_folds
    proto_f1 = sum(x[2] for x in per_fold) / n_folds
    return (proto_ndcg, proto_f1, per_fold, sel_counts, held_ndcg, held_f1, held_n)


def main():
    cands = build_candidates()
    print(f"{len(cands)} candidates, {NFOLDS}-fold CV, seed={SEED}\n")

    agg = []
    for model, files in MODELS.items():
        for k in ("en-sharp", "en-state", "fi-sharp", "fi-state"):
            docs = load_docs(Path("results") / files[k])
            n = len(docs)
            scores = per_doc_scores(docs, cands)
            (p_ndcg, p_f1, per_fold, sel_counts, held_ndcg, held_f1, held_n) = (
                cv_select(scores, n)
            )

            # per-candidate CV estimate (held-out docs where selected)
            cv = [
                (c, held_ndcg[c] / held_n[c], held_f1[c] / held_n[c], sel_counts[c])
                for c in range(len(cands))
                if held_n[c] > 0
            ]
            cv_sorted = sorted(cv, key=lambda t: -t[1])
            top = cv_sorted[0]

            # optimistic best-of-grid on full data (selection-biased)
            full = [
                (c, sum(s[0] for s in scores[c]) / n, sum(s[1] for s in scores[c]) / n)
                for c in range(len(cands))
            ]
            opt = max(full, key=lambda t: t[1])

            annif_ndcg = sum(s[0] for s in scores[0]) / n
            annif_f1 = sum(s[1] for s in scores[0]) / n

            print(f"=== {model} / {k}  ({n} docs) ===")
            print(f"  Annif                  {annif_ndcg:.3f}/{annif_f1:.3f}")
            print(
                f"  PROTOCOL (honest CV)   {p_ndcg:.3f}/{p_f1:.3f}"
                f"   +{p_ndcg - annif_ndcg:+.3f} vs Annif"
            )
            print(
                "    folds: "
                + "  ".join(f"{cands[c][0]}({v:.3f})" for c, v, _ in per_fold)
            )
            print(
                f"  best candidate by CV   {top[1]:.3f}/{top[2]:.3f}"
                f"   ({top[3]}/{NFOLDS} folds)  {cands[top[0]][0]}"
            )
            print(
                f"  best-of-grid(full)     {opt[1]:.3f}/{opt[2]:.3f}"
                f"   noise-gap {opt[1] - p_ndcg:+.3f}  {cands[opt[0]][0]}"
            )
            print()
            agg.append((model, k, annif_ndcg, p_ndcg, opt[1]))

    print("=" * 72)
    print(
        f"{'model/dataset':30s}{'Annif':>8}{'PROTO':>8}{'full-best':>11}"
        f"{'  PR-An':>8}{'  noise':>8}"
    )
    for model, k, an, p, opt in agg:
        print(
            f"{model + '/' + k:30s}{an:>8.3f}{p:>8.3f}{opt:>11.3f}"
            f"{p - an:>+8.3f}{opt - p:>+8.3f}"
        )
    m_an = sum(a[2] for a in agg) / len(agg)
    m_p = sum(a[3] for a in agg) / len(agg)
    m_o = sum(a[4] for a in agg) / len(agg)
    print(
        f"{'mean':30s}{m_an:>8.3f}{m_p:>8.3f}{m_o:>11.3f}"
        f"{m_p - m_an:>+8.3f}{m_o - m_p:>+8.3f}"
    )


if __name__ == "__main__":
    main()
