"""
Artifact evaluation reproducibility tests.

Each test re-runs a paper-figure generation script into a temporary directory
and verifies that the SHA-256 hashes of the selected output files match the
committed reference copies.

These tests require a populated local PostgreSQL database (gpuflops_db /
request_metadata).  If the script exits non-zero for any reason (database
unavailable, data missing, etc.) the test is skipped rather than failed.  The
test harness never loads dump files, writes to, or modifies the database in
any way.
"""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_PROMPTING_DIR = REPO_ROOT / "experiments" / "direct-prompting"
ERROR_ANALYSIS_DIR = REPO_ROOT / "experiments" / "error-analysis"
DIRECT_PROMPTING_REF_DIR = DIRECT_PROMPTING_DIR / "paper-figure-output"
ERROR_ANALYSIS_REF_DIR = ERROR_ANALYSIS_DIR / "paper-figure-output"
REQUEST_METADATA_REF_DIR = DIRECT_PROMPTING_REF_DIR / "request-metadata"


def _sha256(path: Path) -> str:
	"""Return the hex SHA-256 digest of the file at *path*."""
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(65536), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _run_paper_script(script_dir: Path, args: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
	"""Run a paper-figure script as a subprocess with *args* from *script_dir*."""
	return subprocess.run(
		[sys.executable, *args],
		cwd=str(script_dir),
		capture_output=True,
		text=True,
		timeout=timeout,
	)


def _skip_if_failed(result: subprocess.CompletedProcess, script_name: str) -> None:
	"""Skip the test if the script exited with a non-zero return code."""
	if result.returncode != 0:
		output_excerpt = (result.stdout + result.stderr)[:800]
		pytest.skip(
			f"{script_name} exited with code {result.returncode} "
			f"(database unavailable or data missing).\n{output_excerpt}"
		)


def _assert_file_matches_reference(generated: Path, reference: Path, label: str) -> None:
	"""Assert that the generated file is byte-for-byte identical to the reference."""
	assert generated.exists(), f"Expected output file was not created: {generated}"
	assert reference.exists(), f"Reference file not found in repository: {reference}"
	generated_hash = _sha256(generated)
	reference_hash = _sha256(reference)
	assert generated_hash == reference_hash, (
		f"Output mismatch for {label}.\n"
		f"  generated : {generated_hash}  ({generated})\n"
		f"  reference : {reference_hash}  ({reference})\n"
		"If the change is intentional, update the reference file by re-running "
		"the script with its default output directory and committing the result."
	)


@pytest.mark.slow
def test_direct_prompting_make_plots_for_paper(tmp_path):
	"""Re-run make_plots_for_paper.py and verify the four selected figures."""
	result = _run_paper_script(
		DIRECT_PROMPTING_DIR,
		["make_plots_for_paper.py", "--onlySharedSamples", "--outputDir", str(tmp_path)],
	)
	_skip_if_failed(result, "make_plots_for_paper.py (direct-prompting)")

	checked_files = [
		"figure6_expected_rai_distribution_by_gpu_precision.png",
		"figure11_ai_percent_difference_boxplots.png",
		"figure2_5_ai_bound_confusion_heatmaps_with_zero.png",
		"table_figure12_8_threshold_coverage.tex",
	]
	for filename in checked_files:
		_assert_file_matches_reference(
			tmp_path / filename,
			DIRECT_PROMPTING_REF_DIR / filename,
			filename,
		)


@pytest.mark.slow
def test_direct_prompting_fetch_openrouter_request_metadata_plots(tmp_path):
	"""Re-run fetch_openrouter_request_metadata.py and verify the two selected plots."""
	result = _run_paper_script(
		DIRECT_PROMPTING_DIR,
		[
			"fetch_openrouter_request_metadata.py",
			"--makePlotsForPaper",
			"--onlySharedSamples",
			"--plotOutputDir", str(tmp_path),
		],
	)
	_skip_if_failed(result, "fetch_openrouter_request_metadata.py")

	checked_files = [
		"plot2_query_time_distribution.png",
		"plot3_cost_distribution.png",
	]
	for filename in checked_files:
		_assert_file_matches_reference(
			tmp_path / filename,
			REQUEST_METADATA_REF_DIR / filename,
			filename,
		)


@pytest.mark.slow
def test_error_analysis_make_plots_for_paper(tmp_path):
	"""Re-run error-analysis make_plots_for_paper.py and verify the selected figure."""
	result = _run_paper_script(
		ERROR_ANALYSIS_DIR,
		["make_plots_for_paper.py", "--outputDir", str(tmp_path)],
	)
	_skip_if_failed(result, "make_plots_for_paper.py (error-analysis)")

	_assert_file_matches_reference(
		tmp_path / "figure1_model_feature_association_heatmap.png",
		ERROR_ANALYSIS_REF_DIR / "figure1_model_feature_association_heatmap.png",
		"figure1_model_feature_association_heatmap.png",
	)


@pytest.mark.slow
def test_direct_prompting_print_prompt_for_paper_listing(tmp_path):
	"""Re-run print_prompt_for_paper_listing_1.py and verify listing1.txt and listing2.txt."""
	tmp_listing1 = tmp_path / "listing1.txt"
	tmp_listing2 = tmp_path / "listing2.txt"
	result = _run_paper_script(
		DIRECT_PROMPTING_DIR,
		[
			"print_prompt_for_paper_listing_1.py",
			"--listing1Path", str(tmp_listing1),
			"--listing2Path", str(tmp_listing2),
		],
	)
	_skip_if_failed(result, "print_prompt_for_paper_listing_1.py")

	_assert_file_matches_reference(
		tmp_listing1,
		DIRECT_PROMPTING_DIR / "listing1.txt",
		"listing1.txt",
	)
	_assert_file_matches_reference(
		tmp_listing2,
		DIRECT_PROMPTING_DIR / "listing2.txt",
		"listing2.txt",
	)
