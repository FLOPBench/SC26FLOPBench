#!/usr/bin/env bash
# reproduce_paper_results.sh
#
# Reproduces all paper results end-to-end without requiring a GPU or live
# LLM API access.  The script:
#
#   1. Pulls the required LFS files (gpuFLOPBench.json, all-NCU-GPU-Data.csv,
#      gpuflops_db.dump, code_features_db.dump).
#   2. Restores the feature-voting PostgreSQL database from its dump.
#   3. Restores the direct-prompting PostgreSQL database from its dump.
#   4. Restores the request-metadata PostgreSQL database from its dump.
#   5. Generates all paper figures and listings.
#   6. Runs the artifact-evaluation unit tests to verify SHA-256 hashes.
#
# Prerequisites:
#   - The conda environment 'gpuflopbench-updated' must be active, or Python
#     must already have all required packages installed.
#   - Git LFS must be installed and authenticated (e.g. via GITHUB_TOKEN in CI).
#
# Usage:
#   conda activate gpuflopbench-updated
#   ./reproduce_paper_results.sh
#
# Pre-collected request_metadata.dump is committed in the repo and restored in
# step 4 so that Figures 9 & 10 can be reproduced without a live OPENROUTER_API_KEY.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
# Step 1 — Pull required LFS files
# ---------------------------------------------------------------------------
log_info "[1/6] Pulling required LFS files..."

# In CI the files are already present via actions/checkout lfs:true + volume
# mount.  For local runs, git lfs pull fetches any missing content.
git lfs fetch --all

# Verify required files are present at full content (not LFS pointers)
REQUIRED_FILES=(
    "dataset-creation/gpuFLOPBench.json"
    "cuda-profiling/collected-data/all-NCU-GPU-Data.csv"
    "experiments/direct-prompting/gpuflops_db.dump"
    "experiments/direct-prompting/request_metadata.dump"
    "experiments/feature-voting/code_features_db.dump"
)
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        log_error "Required file not found after LFS pull: $f"
        exit 1
    fi
    # A raw LFS pointer is < 200 bytes; real content is much larger.
    SIZE=$(wc -c < "$f")
    if [ "$SIZE" -lt 200 ]; then
        log_error "$f appears to be an LFS pointer (${SIZE} bytes). Run 'git lfs pull' manually."
        exit 1
    fi
done

log_info "LFS files verified."

# ---------------------------------------------------------------------------
# Step 2 — Restore feature-voting database
# ---------------------------------------------------------------------------
log_info "[2/6] Restoring feature-voting database from committed dump..."

cd "$SCRIPT_DIR/experiments/feature-voting"
python run_voting_queries.py --importAndExit
cd "$SCRIPT_DIR"

log_info "code_features_db restored."

# ---------------------------------------------------------------------------
# Step 3 — Restore direct-prompting database
# ---------------------------------------------------------------------------
log_info "[3/6] Restoring direct-prompting database from committed dump..."

cd "$SCRIPT_DIR/experiments/direct-prompting"
python run_queries.py --importAndExit
cd "$SCRIPT_DIR"

log_info "gpuflops_db restored."

# ---------------------------------------------------------------------------
# Step 4 — Restore request-metadata database
# ---------------------------------------------------------------------------
log_info "[4/6] Restoring request-metadata database from committed dump..."

cd "$SCRIPT_DIR/experiments/direct-prompting"
python fetch_openrouter_request_metadata.py --importAndExit
cd "$SCRIPT_DIR"

log_info "request_metadata restored."

# ---------------------------------------------------------------------------
# Step 5 — Artifact evaluation tests (against committed reference files)
# ---------------------------------------------------------------------------
log_info "[5/6] Running artifact evaluation tests..."

cd "$SCRIPT_DIR/unit-tests"
pytest test_artifact_evaluation.py -m slow -v
cd "$SCRIPT_DIR"

log_info "Artifact evaluation tests passed."

# ---------------------------------------------------------------------------
# Step 6 — Generate paper figures and listings
# ---------------------------------------------------------------------------
log_info "[6/6] Generating paper figures and listings..."
# Outputs go to paper-figure-output-reproduced/ to avoid overwriting the
# committed reference files that step 5 validated against.

# Listing 1 and Listing 3
cd "$SCRIPT_DIR/experiments/direct-prompting"
python print_prompt_for_paper_listing_1.py \
    --listing1Path paper-figure-output-reproduced/listing1.txt \
    --listing2Path paper-figure-output-reproduced/listing2.txt
log_info "  listing1.txt and listing2.txt written."

# Direct-prompting figures (Figures 2, 6, 7; Table 3)
python make_plots_for_paper.py \
    --onlySharedSamples \
    --outputDir paper-figure-output-reproduced
log_info "  Direct-prompting figures written to paper-figure-output-reproduced/."

# Request-metadata figures (Figures 9 & 10)
python fetch_openrouter_request_metadata.py \
    --makePlotsForPaper \
    --onlySharedSamples \
    --plotOutputDir paper-figure-output-reproduced/request-metadata
log_info "  Request-metadata figures written to paper-figure-output-reproduced/request-metadata/."

# Error-analysis figure (Figure 8)
cd "$SCRIPT_DIR/experiments/error-analysis"
python make_plots_for_paper.py \
    --outputDir paper-figure-output-reproduced
log_info "  Error-analysis figure written to paper-figure-output-reproduced/."

cd "$SCRIPT_DIR"

echo ""
log_info "All steps completed successfully."
log_info "Paper figures are in:"
log_info "  experiments/direct-prompting/paper-figure-output-reproduced/"
log_info "  experiments/error-analysis/paper-figure-output-reproduced/"
