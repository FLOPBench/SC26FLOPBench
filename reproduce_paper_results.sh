#!/usr/bin/env bash
# reproduce_paper_results.sh
#
# Reproduces all paper results end-to-end without requiring a GPU or live
# LLM API access.  The script:
#
#   1. Pulls the required LFS files (gpuFLOPBench.json, sass_files.zip,
#      gpuflops_db.dump, code_features_db.dump).
#   2. Unzips the SASS archive into scraped-sass/.
#   3. Restores the feature-voting PostgreSQL database from its dump.
#   4. Restores the direct-prompting PostgreSQL database from its dump.
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
# Note: The fetch_openrouter_request_metadata.py step (Figures 9 & 10) requires
# a live OPENROUTER_API_KEY and is intentionally skipped here. The corresponding
# test_artifact_evaluation.py test skips gracefully when the key is absent.

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
git lfs pull --include \
    "dataset-creation/gpuFLOPBench.json,\
cuda-profiling/collected-data/scraped-sass/sass_files.zip,\
experiments/direct-prompting/gpuflops_db.dump,\
experiments/feature-voting/code_features_db.dump" \
    || log_warn "git lfs pull encountered an error; continuing (files may already be present)."

# Verify required files are present at full content (not LFS pointers)
REQUIRED_FILES=(
    "dataset-creation/gpuFLOPBench.json"
    "cuda-profiling/collected-data/scraped-sass/sass_files.zip"
    "experiments/direct-prompting/gpuflops_db.dump"
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
# Step 2 — Unzip SASS archive
# ---------------------------------------------------------------------------
log_info "[2/6] Extracting SASS files from sass_files.zip..."

python cuda-profiling/collected-data/unzip_collected_data.py --extract

log_info "SASS extraction complete."

# ---------------------------------------------------------------------------
# Step 3 — Restore feature-voting database
# ---------------------------------------------------------------------------
log_info "[3/6] Restoring feature-voting database from committed dump..."

cd "$SCRIPT_DIR/experiments/feature-voting"
python run_voting_queries.py --importAndExit
cd "$SCRIPT_DIR"

log_info "code_features_db restored."

# ---------------------------------------------------------------------------
# Step 4 — Restore direct-prompting database
# ---------------------------------------------------------------------------
log_info "[4/6] Restoring direct-prompting database from committed dump..."

cd "$SCRIPT_DIR/experiments/direct-prompting"
python run_queries.py --importAndExit
cd "$SCRIPT_DIR"

log_info "gpuflops_db restored."

# ---------------------------------------------------------------------------
# Step 5 — Generate paper figures and listings
# ---------------------------------------------------------------------------
log_info "[5/6] Generating paper figures and listings..."

# Listing 1 and Listing 3
cd "$SCRIPT_DIR/experiments/direct-prompting"
python print_prompt_for_paper_listing_1.py \
    --listing1Path listing1.txt \
    --listing2Path listing2.txt
log_info "  listing1.txt and listing2.txt written."

# Direct-prompting figures (Figures 2, 6, 7; Table 3)
python make_plots_for_paper.py \
    --onlySharedSamples \
    --outputDir paper-figure-output
log_info "  Direct-prompting figures written to paper-figure-output/."

# Error-analysis figure (Figure 8)
cd "$SCRIPT_DIR/experiments/error-analysis"
python make_plots_for_paper.py \
    --outputDir paper-figure-output
log_info "  Error-analysis figure written to paper-figure-output/."

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Step 6 — Artifact evaluation tests
# ---------------------------------------------------------------------------
log_info "[6/6] Running artifact evaluation tests..."

cd "$SCRIPT_DIR/unit-tests"
pytest test_artifact_evaluation.py -m slow -v
cd "$SCRIPT_DIR"

echo ""
log_info "All steps completed successfully."
log_info "Paper figures are in:"
log_info "  experiments/direct-prompting/paper-figure-output/"
log_info "  experiments/error-analysis/paper-figure-output/"
