#!/usr/bin/env python3
"""
unzip_collected_data.py

Helper script to unzip pre-collected profiling archives into the GPU
subdirectories expected by condense_perf_counter_data.py, and to restore
SASS files from the SASS archive into scraped-sass/.

By default the script runs in DRY-RUN mode and only prints what would be
extracted without touching any files.  Pass --extract to perform the
actual extraction.  Files that already exist at the destination are
skipped unless --overwrite is also given.

Archives handled:
  NVIDIA_*_profiling-results-*.zip   (one per GPU, e.g. collected from
                                       runProfiling.sh runs on each machine)
        └─ extracted to: ./{short_gpu_name}/
           where short_gpu_name is one of: 3080, A10, A100, H100, ...

  scraped-sass/sass_files.zip         (produced by extact_sass_from_built_executables.py)
        └─ extracted to: ./scraped-sass/

Usage:
    # Preview what would be extracted (safe, no files written):
    python unzip_collected_data.py

    # Actually extract, skipping files that already exist:
    python unzip_collected_data.py --extract

    # Extract and overwrite any existing files:
    python unzip_collected_data.py --extract --overwrite
"""

import argparse
import re
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Ordered list of (substring_to_match, short_dir_name) pairs.
# A100 must be listed before A10 so the longer prefix matches first.
GPU_SHORT_NAME_MAP = [
    ("GeForce_RTX_3080", "3080"),
    ("RTX_3080",         "3080"),
    ("3080",             "3080"),
    ("A100",             "A100"),
    ("A10",              "A10"),
    ("H100",             "H100"),
    ("V100",             "V100"),
    ("A30",              "A30"),
    ("A40",              "A40"),
    ("A6000",            "A6000"),
    ("RTX_4090",         "4090"),
    ("4090",             "4090"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_model_to_short_name(gpu_model: str) -> str | None:
    """Return the short subdirectory name for a GPU model string, or None."""
    for pattern, short in GPU_SHORT_NAME_MAP:
        if pattern in gpu_model:
            return short
    return None


def _extract_gpu_model_from_zip_name(zip_name: str) -> str | None:
    """
    Parse the GPU model string from a zip filename of the form:
        NVIDIA_{GPU_MODEL}_profiling-results-{TIMESTAMP}.zip

    Returns the GPU model string (with underscores/dashes) or None.
    """
    m = re.match(r"NVIDIA_(.+?)_profiling-results-", zip_name)
    return m.group(1) if m else None


def _plan_profiling_zip(
    zip_path: Path,
) -> tuple[str | None, Path | None, list[tuple[str, Path]]]:
    """
    Return (short_name, dest_dir, [(member_name, dest_path), ...]) for a
    profiling-results zip.

    short_name is None when the GPU model cannot be identified; dest_dir and
    members will be None / empty in that case.
    """
    gpu_model = _extract_gpu_model_from_zip_name(zip_path.name)
    if gpu_model is None:
        return None, None, []

    short = _gpu_model_to_short_name(gpu_model)
    if short is None:
        return gpu_model, None, []

    dest_dir = SCRIPT_DIR / short
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            (m, dest_dir / m)
            for m in zf.namelist()
            if not m.endswith("/")
        ]
    return short, dest_dir, members


def _plan_sass_zip(zip_path: Path) -> tuple[Path, list[tuple[str, Path]]]:
    """
    Return (dest_dir, [(member_name, dest_path), ...]) for
    scraped-sass/sass_files.zip.

    SASS zips store files without a leading directory component, so each
    member is extracted directly into scraped-sass/.
    """
    dest_dir = SCRIPT_DIR / "scraped-sass"
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            (m, dest_dir / Path(m).name)
            for m in zf.namelist()
            if not m.endswith("/")
        ]
    return dest_dir, members


def _extract_member(
    zip_path: Path,
    member: str,
    dest: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> str:
    """
    Extract one member from a zip to dest.

    Returns a short status string:
      "would extract", "extracted", or "skipped (already exists)".
    """
    if dest.exists() and not overwrite:
        return "skipped (already exists)"
    if dry_run:
        return "would extract"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    dest.write_bytes(data)
    return "extracted"


def _summarise_counts(counts: dict[str, int]) -> str:
    parts = [f"{count} {status}" for status, count in sorted(counts.items()) if count]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unzip pre-collected profiling archives into the expected subdirectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help=(
            "Perform the actual extraction. "
            "Without this flag the script runs in dry-run mode and only "
            "prints what would happen."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist at the destination.",
    )
    args = parser.parse_args()

    dry_run = not args.extract
    if dry_run:
        print("DRY RUN — pass --extract to perform extraction.\n")

    any_archive_found = False

    # ------------------------------------------------------------------
    # 1. Profiling archives: NVIDIA_*_profiling-results-*.zip
    # ------------------------------------------------------------------
    profiling_zips = sorted(
        SCRIPT_DIR.glob("NVIDIA_*_profiling-results-*.zip")
    )

    for zip_path in profiling_zips:
        any_archive_found = True
        short, dest_dir, members = _plan_profiling_zip(zip_path)

        if short is None:
            print(
                f"WARNING: could not parse GPU model from {zip_path.name}; "
                "skipping."
            )
            continue

        if dest_dir is None:
            print(
                f"WARNING: GPU model '{short}' from {zip_path.name} is not in "
                "the known short-name map; skipping.  Add an entry to "
                "GPU_SHORT_NAME_MAP in this script to handle it."
            )
            continue

        counts: dict[str, int] = {}
        for member, dest in members:
            status = _extract_member(
                zip_path, member, dest, overwrite=args.overwrite, dry_run=dry_run
            )
            counts[status] = counts.get(status, 0) + 1

        rel_dest = dest_dir.relative_to(SCRIPT_DIR)
        print(
            f"{zip_path.name}\n"
            f"  → {rel_dest}/ ({len(members)} files)  "
            f"[{_summarise_counts(counts)}]"
        )

    if not profiling_zips:
        print("No NVIDIA_*_profiling-results-*.zip files found in this directory.")

    # ------------------------------------------------------------------
    # 2. SASS archive: scraped-sass/sass_files.zip
    # ------------------------------------------------------------------
    sass_zip = SCRIPT_DIR / "scraped-sass" / "sass_files.zip"

    if sass_zip.exists():
        any_archive_found = True
        dest_dir, members = _plan_sass_zip(sass_zip)

        counts = {}
        for member, dest in members:
            status = _extract_member(
                sass_zip, member, dest, overwrite=args.overwrite, dry_run=dry_run
            )
            counts[status] = counts.get(status, 0) + 1

        rel_dest = dest_dir.relative_to(SCRIPT_DIR)
        print(
            f"\n{sass_zip.relative_to(SCRIPT_DIR)}\n"
            f"  → {rel_dest}/ ({len(members)} files)  "
            f"[{_summarise_counts(counts)}]"
        )
    else:
        print(
            "\nscraped-sass/sass_files.zip not found "
            "(run extact_sass_from_built_executables.py first to generate it)."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if dry_run:
        if any_archive_found:
            print("Dry run complete.  Re-run with --extract to perform extraction.")
        else:
            print("No archives found.  Nothing to extract.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
