#!/usr/bin/env bash

# runTests.sh - Test script for gpuFLOPBench-updated unit tests
#
# This script runs pytest on the unit-tests directory with optional
# filtering for GPU tests.
#
# Usage:
#   ./runTests.sh           # Run all tests
#   ./runTests.sh --noGPU   # Run tests excluding GPU tests
#
# Options:
#   --noGPU   Exclude tests that require GPU access (marked with 'gpu' marker)
#   --help    Show this help message

set -e  # Exit on error

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
TESTS_DIR="$PROJECT_ROOT/unit-tests"

# Default configuration
EXCLUDE_GPU=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help message
show_help() {
    cat << EOF
Usage: ./runTests.sh [OPTIONS]

Run unit tests for gpuFLOPBench-updated.

Options:
    --noGPU   Exclude tests that require GPU access (marked with 'gpu' marker)
    --help    Show this help message

Examples:
    ./runTests.sh           # Run all tests
    ./runTests.sh --noGPU   # Run only tests that don't require GPU
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --noGPU)
            EXCLUDE_GPU=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if tests directory exists
if [ ! -d "$TESTS_DIR" ]; then
    log_error "Tests directory not found: $TESTS_DIR"
    exit 1
fi

# Check for pytest
if ! command -v pytest &> /dev/null; then
    log_error "pytest not found. Please install pytest."
    exit 1
fi

# Build pytest command
cd "$TESTS_DIR"

if [ "$EXCLUDE_GPU" = true ]; then
    log_info "Running tests (excluding GPU and slow tests)..."
    pytest -v -s -m 'not gpu and not slow'
else
    log_info "Running all tests..."
    pytest -v -s
fi

# pytest will return its exit code, which we propagate
exit $?
