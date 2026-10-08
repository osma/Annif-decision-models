#!/bin/sh
# Unattended experiment runner for a Jev-like decision model.
#
# Performs the full pipeline for all four sharp configurations
# (en/fi x with/without --state-rules), then blend sweeps, plots and a
# final comparison summary against the lev baselines.
#
# Usage:
#   ./run_model.sh [MODEL]
#     MODEL   model name used in request payloads and output filenames
#             (default: clefflash). e.g. ./run_model.sh laya
#
# Other settings via environment (all optional):
#   BASE_URL          llama.cpp server (default http://localhost:8080)
#   ANNIF_JEV_TIMEOUT per-request timeout seconds (default 600; slow models)
#   RETRIES           retries on API failure (default 3)
#
# Designed to be launched unmonitored (e.g. nohup ./run_model.sh mymodel &).
# All progress is written to logs/<model>/run-<timestamp>.log and to the
# final summary.

set -u

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="${1:-clefflash}"
BASE_URL="${BASE_URL:-http://localhost:8080}"
# Slow model: evaluate per-request timeout generously (50+ questions/request).
export ANNIF_JEV_TIMEOUT="${ANNIF_JEV_TIMEOUT:-600}"
RETRIES="${RETRIES:-3}"
PY="./.venv/bin/python"

LOG_DIR="logs/$MODEL"
mkdir -p "$LOG_DIR" results
MAIN_LOG="$LOG_DIR/run-$(date +%Y%m%d-%H%M%S).log"

# Per-run output files, derived from the model name:
#   rerank-<lang>-<model>-sharp{,-state}.csv
CSV_EN_PLAIN="results/rerank-en-${MODEL}-sharp.csv"
CSV_EN_STATE="results/rerank-en-${MODEL}-sharp-state.csv"
CSV_FI_PLAIN="results/rerank-fi-${MODEL}-sharp.csv"
CSV_FI_STATE="results/rerank-fi-${MODEL}-sharp-state.csv"

# Baselines for the final summary (model NDCG / F1@5, same configs).
# Lev baselines:
#   en sharp 0.678/0.362, en sharp+state 0.698/0.318
#   fi sharp 0.691/0.329, fi sharp+state 0.694/0.333
# Annif baselines:
#   en 0.796/0.502, fi 0.804/0.485

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

# Wait until the endpoint answers a trivial noul request.
# Returns 0 if the endpoint is up, 1 after MAX_PROBES failed probes.
wait_for_endpoint() {
    max_probes="${1:-60}"
    i=0
    while [ "$i" -lt "$max_probes" ]; do
        if curl -s -m 60 -X POST "$BASE_URL/v1/systemone" \
                -H 'Content-Type: application/json' \
                -d '{"model":"'"$MODEL"'","state":"test.","questions":{"q1":{"type":"noul","instructions":"Is this about cats?"}}}' \
            | grep -q '"noul"'; then
            return 0
        fi
        i=$((i + 1))
        log "endpoint not up yet (probe $i/$max_probes), waiting 60s..."
        sleep 60
    done
    return 1
}

# Run one rerank configuration.
# Args: <dataset-dir> <lang> <output-csv> <state-arg> <tag>
run_rerank() {
    dataset_dir="$1"
    lang="$2"
    out="$3"
    state_arg="$4"
    tag="$5"
    logfile="$LOG_DIR/${lang}-${tag}.log"

    # Start clean: never append to a stale/partial file.
    rm -f "$out"

    log "RUN  ${lang} sharp ${tag} -> $out"
    if "$PY" rerank.py \
            --dataset-dir "$dataset_dir" \
            --lang "$lang" \
            --base-url "$BASE_URL" \
            --model "$MODEL" \
            --question-style sharp \
            --retries "$RETRIES" \
            $state_arg \
            --output "$out" > "$logfile" 2>&1; then
        log "OK   $out ($(wc -l < "$out") lines)"
        return 0
    else
        log "FAIL ${lang} ${tag} run; removing partial $out (see $logfile)"
        # Keep the partial for debugging, but mark it so re-runs start clean.
        [ -f "$out" ] && mv "$out" "$out.partial"
        return 1
    fi
}

analyze_one() {
    out="$1"
    [ -f "$out" ] || { log "SKIP analyze (missing $out)"; return 0; }
    log "BLEND $out"
    "$PY" blend.py "$out" 2>&1 | tee -a "$MAIN_LOG" | tail -9
    log "PLOT  $out"
    "$PY" analyze.py "$out" 2>&1 | tee -a "$MAIN_LOG" | tail -4
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "=== experiment run started ==="
log "model=$MODEL base_url=$BASE_URL timeout=${ANNIF_JEV_TIMEOUT}s retries=$RETRIES"
log "main log: $MAIN_LOG"

# 1. Make sure the endpoint is serving.
if ! wait_for_endpoint 60; then
    log "FATAL: endpoint $BASE_URL never came up; aborting."
    exit 1
fi
log "endpoint is up."

# 2. The four rerank runs (sequentially -- parallel runs have crashed servers).
run_rerank "jev-test-sets/en" "en" "$CSV_EN_PLAIN" "" "plain"
run_rerank "jev-test-sets/en" "en" "$CSV_EN_STATE" "--state-rules" "state"
run_rerank "jev-test-sets/fi" "fi" "$CSV_FI_PLAIN" "" "plain"
run_rerank "jev-test-sets/fi" "fi" "$CSV_FI_STATE" "--state-rules" "state"

# 3. Blends + plots for every CSV we have.
for f in "$CSV_EN_PLAIN" "$CSV_EN_STATE" "$CSV_FI_PLAIN" "$CSV_FI_STATE"; do
    analyze_one "$f"
done

# 4. Final comparison summary.
log "=== SUMMARY ($MODEL vs lev, model NDCG / F1@5) ==="
"$PY" - "$CSV_EN_PLAIN" "$CSV_EN_STATE" "$CSV_FI_PLAIN" "$CSV_FI_STATE" <<'PYEOF' | tee -a "$MAIN_LOG"
import sys
import pandas as pd
import statistics as st
from metrics import ndcg, f1_at_k

en_plain, en_state, fi_plain, fi_state = sys.argv[1:5]

def doc_metrics(d, score_col):
    nds, f1s = [], []
    for _, g in d.groupby("isbn", sort=False):
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        rel = (g.in_gold == 1).tolist()
        nds.append(ndcg(rel))
        f1s.append(f1_at_k(rel, 5))
    return st.mean(nds), st.mean(f1s)

rows = [
    ("en sharp",      en_plain,   "0.678 0.362", "0.796 0.502"),
    ("en sharp+state", en_state,  "0.698 0.318", "0.796 0.502"),
    ("fi sharp",      fi_plain,   "0.691 0.329", "0.804 0.485"),
    ("fi sharp+state", fi_state,  "0.694 0.333", "0.804 0.485"),
]
print(f"{'config':16} {'model NDCG/F1':>14} {'lev NDCG/F1':>14} {'annif NDCG/F1':>14}")
for name, path, lev, annif in rows:
    try:
        d = pd.read_csv(path, dtype={"isbn": str})
        nd, f1 = doc_metrics(d, "noul_score")
        print(f"{name:16} {nd:.3f} / {f1:.3f}   {lev}    {annif}")
    except FileNotFoundError:
        print(f"{name:16}  (no CSV -- run failed)")
PYEOF

log "=== experiment run finished ==="
exit 0
