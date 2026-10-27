#!/usr/bin/env python3
"""Rerank Annif subject-indexing suggestions using a Jev-like decision model.

Reads document text (.txt), Annif suggestions (.annif), and gold-standard
subjects (.tsv) from a dataset directory.  For each book the script asks a
decision model (via the llama.cpp /v1/systemone endpoint) a noul question
per candidate subject and stores the raw probabilities to CSV for later
analysis (phase 2).

Usage:
    python rerank.py --base-url http://other-machine:8080 --model ggml-org/lev-GGUF
    python rerank.py --question-style sharp --state-prefix "..." --output out.csv
    python rerank.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "isbn",
    "lang",
    "yso_uri",
    "label",
    "source",
    "annif_score",
    "annif_rank",
    "noul_score",
    "in_gold",
    "instruction",
    "model",
]

# Question templates per style, each with a {label} slot.
INSTRUCTIONS = {
    # Plain: bare topically-about check (the original).
    "plain": "This document is about {label}.",
    # Sharp: force the model to judge CENTRALITY, not mere adjacency — a
    # passing or incidental mention should be answered "no".  This phrasing
    # lifted noul NDCG noticeably on the en test set (0.604 -> 0.678).
    "sharp": (
        "Is '{label}' a central subject of this document - one a "
        "librarian would assign as a primary heading - not merely a "
        "passing or incidental mention?"
    ),
}

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "ggml-org/lev-GGUF"
REQUEST_TIMEOUT = float(os.environ.get("ANNIF_JEV_TIMEOUT", "60"))  # seconds

# Best state-prefix found in the experiments (format/genre rule only).
# On the en test set this lifted the best blend from 0.832 to 0.838 NDCG
# (prod blend) and was the only query-side change that beat Annif on both
# NDCG and F1@5. Enable with --state-rules.
STATE_RULES = (
    "You are a librarian performing topical subject indexing. "
    "A document can have several central subjects. "
    "Do NOT assign the document's own format or genre as a subject "
    "(a book of recipes is not about 'cookbooks', a pattern book is not "
    "about 'handicraft patterns', a travel book is not about "
    "'travelogues').\n\nDocument to be indexed:\n"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A single subject candidate for one document."""

    uri: str
    label: str
    source: str  # "annif" or "gold_extra"
    annif_score: float | None = None
    annif_rank: int | None = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rerank Annif suggestions with a Jev-like decision model.",
    )
    p.add_argument(
        "--dataset-dir",
        default="jev-test-sets/en",
        help="Directory containing .txt/.annif/.tsv files (default: %(default)s)",
    )
    p.add_argument(
        "--lang",
        default="en",
        help="Language tag (default: %(default)s)",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("ANNIF_JEV_URL", DEFAULT_BASE_URL),
        help=f"llama.cpp server URL (env ANNIF_JEV_URL, default: {DEFAULT_BASE_URL})",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("ANNIF_JEV_MODEL", DEFAULT_MODEL),
        help=f"Model id (env ANNIF_JEV_MODEL, default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--output",
        default="-",
        help="Output CSV path, or '-' for stdout (default: %(default)s)",
    )
    p.add_argument(
        "--question-style",
        default="sharp",
        choices=list(INSTRUCTIONS),
        help="noul question phrasing: 'plain' (topically about) or 'sharp' "
        "(central subject, rejects passing mentions; the better of the "
        "two) (default: %(default)s)",
    )
    p.add_argument(
        "--state-rules",
        action="store_true",
        help="Prepend the built-in indexing-rules prefix to the document "
        "state for every question (best result found on the en set; "
        "see README). Takes effect unless --state-prefix is also given.",
    )
    p.add_argument(
        "--state-prefix",
        default="",
        help="Custom text prepended to the document state for every "
        "question. Overrides --state-rules if both are given.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max Annif suggestions per doc (default: %(default)s)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries on API failure (default: %(default)s)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first request payload and exit without calling the API",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Process only the first N documents (0 = all, default)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def find_books(dataset_dir: Path) -> list[Path]:
    """Return sorted list of .txt paths that also have a matching .annif."""
    return sorted(
        txt for txt in dataset_dir.glob("*.txt") if txt.with_suffix(".annif").exists()
    )


def parse_annif(path: Path, limit: int) -> list[Candidate]:
    """Parse a .annif file into ranked candidates (tab: uri, label, score)."""
    candidates: list[Candidate] = []
    with open(path, encoding="utf-8") as f:
        for rank, line in enumerate(f):
            if rank >= limit:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            candidates.append(
                Candidate(
                    uri=parts[0].strip("<>"),
                    label=parts[1],
                    source="annif",
                    annif_score=float(parts[2]),
                    annif_rank=rank,
                )
            )
    return candidates


def parse_tsv(path: Path) -> list[dict[str, str]]:
    """Parse a .tsv gold file into [{uri, label}, …]. Empty list if missing."""
    entries: list[dict[str, str]] = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            entries.append({"uri": parts[0].strip("<>"), "label": parts[1]})
    return entries


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_candidates(
    annif_suggestions: list[Candidate],
    gold: list[dict[str, str]],
) -> list[Candidate]:
    """Merge Annif suggestions with gold subjects not already in the list."""
    candidates = list(annif_suggestions)
    annif_uris = {c.uri for c in candidates}
    for entry in gold:
        if entry["uri"] not in annif_uris:
            candidates.append(
                Candidate(
                    uri=entry["uri"],
                    label=entry["label"],
                    source="gold_extra",
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def build_instructions(candidates: list[Candidate], style: str) -> list[str]:
    """Build the instruction string for each candidate using the given style."""
    tpl = INSTRUCTIONS[style]
    return [tpl.format(label=c.label) for c in candidates]


def build_payload(
    model: str,
    state: str,
    instructions: list[str],
) -> dict[str, Any]:
    questions = {
        str(i): {"type": "noul", "instructions": instr}
        for i, instr in enumerate(instructions)
    }
    return {"model": model, "state": state, "questions": questions}


def query_systemone(base_url: str, payload: dict, retries: int) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/v1/systemone"
    attempt = 0
    while True:
        try:
            resp = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as err:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(
                    f"API request to {endpoint} failed after "
                    f"{retries + 1} attempts: {err}"
                ) from err
            delay = 2**attempt
            print(
                f"  request failed ({err}); retrying in {delay}s "
                f"({attempt}/{retries})",
                file=sys.stderr,
            )
            time.sleep(delay)


def extract_noul_scores(response: dict[str, Any]) -> dict[str, float | None]:
    """Map question-id → noul probability from the API response."""
    scores: dict[str, float | None] = {}
    for qid, answer in response.get("answers", {}).items():
        scores[qid] = answer.get("noul")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_book(
    txt_path: Path,
    args: argparse.Namespace,
) -> tuple[str, str, list[Candidate], list[str], set[str]]:
    """Load one book and return (isbn, state, candidates, instructions, gold_uris)."""
    isbn = txt_path.stem
    state = load_text(txt_path)
    prefix = args.state_prefix or (STATE_RULES if args.state_rules else "")
    if prefix:
        state = prefix + state
    annif_path = txt_path.with_suffix(".annif")
    tsv_path = txt_path.with_suffix(".tsv")

    annif_suggestions = parse_annif(annif_path, args.limit)
    gold = parse_tsv(tsv_path)
    gold_uris = {g["uri"] for g in gold}
    candidates = build_candidates(annif_suggestions, gold)
    instructions = build_instructions(candidates, args.question_style)
    return isbn, state, candidates, instructions, gold_uris


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    to_stdout = args.output == "-"
    output_path = None if to_stdout else Path(args.output)

    if not dataset_dir.is_dir():
        print(f"Error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    books = find_books(dataset_dir)
    if args.max_docs > 0:
        books = books[: args.max_docs]
    if not books:
        print(
            f"Error: no books found (need .txt + .annif) in {dataset_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Books:      {len(books)}", file=sys.stderr)
    print(f"Model:      {args.model}", file=sys.stderr)
    print(f"Base URL:   {args.base_url}", file=sys.stderr)
    print(f"Question:   {args.question_style}", file=sys.stderr)
    prefix = args.state_prefix or (STATE_RULES if args.state_rules else "")
    if prefix:
        kind = "custom prefix" if args.state_prefix else "built-in rules"
        print(f"State prefix: {len(prefix)} chars ({kind})", file=sys.stderr)
    print(file=sys.stderr)

    # --dry-run: build payload for the first book and print it
    if args.dry_run:
        isbn, state, candidates, instructions, _ = process_book(books[0], args)
        payload = build_payload(args.model, state, instructions)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        n_annif = sum(1 for c in candidates if c.source == "annif")
        n_extra = len(candidates) - n_annif
        print(
            f"\n# {len(candidates)} questions "
            f"({n_annif} annif + {n_extra} gold_extra) for {isbn}",
            file=sys.stderr,
        )
        return

    # Normal run
    total = len(books)
    if to_stdout:
        fh = sys.stdout
        write_header = True
    else:
        file_exists = output_path.exists() and output_path.stat().st_size > 0
        write_header = not file_exists
        mode = "a" if file_exists else "w"
        fh = open(output_path, mode, newline="", encoding="utf-8")

    try:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for doc_idx, txt_path in enumerate(books, 1):
            isbn, state, candidates, instructions, gold_uris = process_book(
                txt_path, args
            )
            payload = build_payload(args.model, state, instructions)

            try:
                response = query_systemone(args.base_url, payload, args.retries)
                noul_scores = extract_noul_scores(response)
            except RuntimeError as err:
                print(f"  WARNING: {err}", file=sys.stderr)
                noul_scores = {}

            vals = [v for v in noul_scores.values() if v is not None]
            if vals:
                print(
                    f"[{doc_idx}/{total}] {isbn}: "
                    f"noul min={min(vals):.3f} "
                    f"mean={sum(vals) / len(vals):.3f} "
                    f"max={max(vals):.3f} "
                    f"({len(candidates)} q)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[{doc_idx}/{total}] {isbn}: no noul scores",
                    file=sys.stderr,
                )

            for i, cand in enumerate(candidates):
                qid = str(i)
                writer.writerow(
                    {
                        "isbn": isbn,
                        "lang": args.lang,
                        "yso_uri": cand.uri,
                        "label": cand.label,
                        "source": cand.source,
                        "annif_score": (
                            cand.annif_score if cand.annif_score is not None else ""
                        ),
                        "annif_rank": (
                            cand.annif_rank if cand.annif_rank is not None else ""
                        ),
                        "noul_score": noul_scores.get(qid, ""),
                        "in_gold": "true" if cand.uri in gold_uris else "false",
                        "instruction": instructions[i],
                        "model": args.model,
                    }
                )

        dest = "stdout" if to_stdout else str(output_path)
        print(f"\nDone. Results written to {dest}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    finally:
        if not to_stdout:
            fh.close()


if __name__ == "__main__":
    main()
