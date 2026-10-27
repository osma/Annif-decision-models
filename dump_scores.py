#!/usr/bin/env python3
"""Dump per-candidate source scores, decision model (noul) scores and gold
labels for the blend research experiments.

Replays the exact production pipeline of a dm_rerank project (source
project suggestions on the project-transformed text, merge,
max-candidates truncation, and one decision model query per document)
by calling the backend's own hyperparameter-optimizer preparation step,
and writes one CSV with a row per candidate:

    doc_id, subject_id, label, source_score, noul_score, gold

* doc_id:        0-based position in the corpus (stable, in corpus order)
* subject_id:    the vocabulary subject index
* label:         the label in the project language (fallback: any label,
                 then the notation; empty string if none)
* source_score:  the merged source score, as returned by the sources
* noul_score:    the raw yes-probability from the decision model, or
                 empty if the question was not asked (no label)
* gold:          1 if the subject is a gold label of the document, else 0

The noul scores come from the live reranking service, so each corpus
needs one service pass (~the time of one annif hyperopt model phase).
The resulting table is static, so any blend formula can afterwards be
evaluated on it offline without any further service calls.

Usage:
    python dump_scores.py -p yso-dm-en \
        --out results/scores-eng-validate.csv \
        ../Annif-corpora/fulltext/jyu-theses/eng-validate/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the main Annif repository importable and behave as if the CLI were
# run from its root (project configuration discovery is CWD-relative).
# This script lives in the prototype repository, which is a subdirectory of
# the main repository; override with ANNIF_ROOT if needed.
ANNIF_ROOT = os.environ.get("ANNIF_ROOT", str(Path(__file__).resolve().parents[1]))
_ORIG_CWD = os.getcwd()
sys.path.insert(0, ANNIF_ROOT)
os.chdir(ANNIF_ROOT)
# mimic the CLI, which does not eagerly initialize every project (avoids
# loading models for unrelated projects that may be unavailable)
os.environ["FLASK_RUN_FROM_CLI"] = "true"

import annif  # noqa: E402
import annif.cli_util  # noqa: E402
import annif.registry  # noqa: E402,F401  (needed by cli_util.get_project)


def label_of(subjects, subject_id: int, language: str) -> str:
    """The subject's label in the project language, falling back to any
    available label, then the notation, then an empty string."""
    try:
        subject = subjects[subject_id]
    except IndexError:
        return ""
    if subject.labels:
        if language in subject.labels:
            return subject.labels[language]
        return next(iter(subject.labels.values()))
    if subject.notation:
        return subject.notation
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        "-p",
        required=True,
        help="the dm_rerank project id to replay (e.g. yso-dm-en)",
    )
    parser.add_argument(
        "--out", "-o", required=True, help="output CSV path (one row per candidate)"
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=8, help="jobs for the source suggestions"
    )
    parser.add_argument(
        "corpus", help="evaluation corpus path (TSV/CSV/JSONL file or directory)"
    )
    args = parser.parse_args()
    # resolve the corpus path against the original working directory, since
    # the CWD was already changed to the Annif root at import time
    args.corpus = str(Path(_ORIG_CWD, args.corpus).resolve())

    app = annif.create_flask_app()
    with app.app_context():
        proj = annif.cli_util.get_project(args.project)
        documents = annif.cli_util.open_documents(
            (args.corpus,), proj.subjects, proj.vocab_lang, None
        )

        # The optimizer's _prepare runs exactly the production pipeline:
        # project transform -> source suggestions (full text for the
        # sources) -> merge -> max-candidates -> one decision model query
        # per document. Only _prepare is called; no trials are run.
        optimizer = proj.backend.get_hp_optimizer(documents, "NDCG@10")
        cached = optimizer._prepare(n_jobs=args.jobs)

        source_suggestions = cached["source_suggestions"]
        noul_scores = cached["noul_scores"]
        gold_batches = cached["gold_batches"]
        subjects = proj.subjects
        language = proj.language

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows_written = 0
        with open(out_path, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(
                ["doc_id", "subject_id", "label", "source_score", "noul_score", "gold"]
            )
            for doc_id, (cands, noul, gold) in enumerate(
                zip(source_suggestions, noul_scores, gold_batches)
            ):
                # gold is a SubjectSet: an iterable of subject IDs
                gold_ids = set(gold)
                for cand in cands:
                    writer.writerow(
                        [
                            doc_id,
                            cand.subject_id,
                            label_of(subjects, cand.subject_id, language),
                            f"{cand.score:.6f}",
                            noul.get(cand.subject_id, ""),
                            1 if cand.subject_id in gold_ids else 0,
                        ]
                    )
                    rows_written += 1

        # a sidecar JSON with the run metadata, for reproducibility
        backend = proj.backend
        metadata = {
            "project": args.project,
            "corpus": args.corpus,
            "endpoint": backend.params["endpoint"],
            "model": backend.params["model"],
            "model-strength": backend.params["model-strength"],
            "gate-threshold": backend.params["gate-threshold"],
            "max-candidates": backend.params["max-candidates"],
            "state-transform": backend.params["state-transform"],
            "limit": backend.params["limit"],
            "docs": len(source_suggestions),
            "rows": rows_written,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = out_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")

        print(
            f"wrote {rows_written} candidate rows for "
            f"{len(source_suggestions)} documents to {out_path}"
        )


if __name__ == "__main__":
    main()
