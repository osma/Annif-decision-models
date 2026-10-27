#!/usr/bin/env python3
"""Recompute the gold column of a dump_scores.py output CSV without any
model calls.

The original dump compared candidate subject IDs against a gold
SubjectSet via its URI, but SubjectSet holds subject *IDs*, so the gold
column was all zeros. This script re-derives the per-document gold
subject-ID sets from the corpus (same corpus, same order as the dump)
and rewrites the CSV's gold column in place.

Usage:
    python patch_gold.py --project yso-dm-en \
        --csv results/scores-eng-validate.csv \
        /home/oisuomin/git/Annif-corpora/fulltext/jyu-theses/eng-validate/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Mirror dump_scores.py's bootstrap: importable annif, CWD = Annif root,
# CLI-style (lazy) project initialization.
ANNIF_ROOT = os.environ.get("ANNIF_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ANNIF_ROOT)
os.chdir(ANNIF_ROOT)
os.environ["FLASK_RUN_FROM_CLI"] = "true"

import annif  # noqa: E402
import annif.cli_util  # noqa: E402
import annif.registry  # noqa: E402,F401  (needed by cli_util.get_project)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--csv", "-c", required=True)
    parser.add_argument("corpus")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    corpus_path = str(Path(os.getcwd(), args.corpus).resolve())

    app = annif.create_flask_app()
    with app.app_context():
        proj = annif.cli_util.get_project(args.project)
        documents = annif.cli_util.open_documents(
            (corpus_path,), proj.subjects, proj.vocab_lang, None
        )
        # gold subject-ID sets in corpus order, exactly as the dump saw
        # them (doc_batches yields the same Documents, in the same order,
        # that the optimizer's _prepare consumes)
        gold_sets = [
            set(doc.subject_set) if doc.subject_set is not None else set()
            for batch in documents.doc_batches
            for doc in batch
        ]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    doc_col = header.index("doc_id")
    subj_col = header.index("subject_id")
    gold_col = header.index("gold")
    n_docs = len(gold_sets)
    fixed = 0
    for row in rows:
        doc_id = int(row[doc_col])
        if doc_id >= n_docs:
            sys.exit(f"doc_id {doc_id} out of range (corpus has {n_docs})")
        new_gold = 1 if int(row[subj_col]) in gold_sets[doc_id] else 0
        if new_gold != int(row[gold_col]):
            fixed += 1
        row[gold_col] = str(new_gold)

    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, csv_path)

    total_gold = sum(len(g) for g in gold_sets)
    print(
        f"patched {csv_path}: {fixed} rows changed "
        f"({total_gold} gold instances in corpus of {n_docs} docs)"
    )


if __name__ == "__main__":
    main()
