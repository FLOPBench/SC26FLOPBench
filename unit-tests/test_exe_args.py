"""
Tests for Makefile-based executable argument extraction.
"""

from pathlib import Path
import os
import sys
import warnings
import yaml
import pytest

GATHERDATA_DIR = Path(__file__).resolve().parents[1] / "cuda-profiling"
sys.path.insert(0, str(GATHERDATA_DIR))

import utils as gd


def _load_benchmarks_yaml():
    yaml_path = Path(__file__).resolve().parents[1] / "HeCBench" / "benchmarks.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _find_src_dirs(benchmark_name):
    src_root = Path(__file__).resolve().parents[1] / "HeCBench" / "src"
    candidates = [
        src_root / f"{benchmark_name}-cuda",
        src_root / f"{benchmark_name}-omp",
    ]
    return [p for p in candidates if p.is_dir()]


def _collect_makefile_args(benchmark_name):
    args_lists = []
    for src_dir in _find_src_dirs(benchmark_name):
        args_lists.extend(gd.get_makefile_run_args(str(src_dir)))
    return args_lists


def _normalize_args(args):
    return [str(a) for a in args]


def _iter_built_targets():
    root = Path(__file__).resolve().parents[1]
    bin_root = root / "build" / "bin"
    src_root = root / "HeCBench" / "src"

    for model in ["cuda", "omp"]:
        model_dir = bin_root / model
        if not model_dir.is_dir():
            continue

        for entry in model_dir.iterdir():
            if not entry.is_file():
                continue
            if not os.access(entry, os.X_OK):
                continue
            if entry.suffix in {".cpp", ".c", ".o", ".so", ".log"}:
                continue

            target_name = entry.name
            src_dir = src_root / f"{target_name}-{model}"
            if not src_dir.is_dir():
                fallback = src_root / target_name
                if fallback.is_dir():
                    src_dir = fallback
                else:
                    continue

            yield target_name, model, src_dir


def test_makefile_args_for_built_targets():
    targets = list(_iter_built_targets())
    assert targets, "No built targets found under build/bin"

    empty_args = {"cuda": [], "omp": []}

    for target_name, model, src_dir in targets:
        makefile_path = gd.find_makefile_for_target(str(src_dir))
        assert makefile_path, f"Makefile not found for {target_name} ({model})"

        #args_lists = gd.extract_run_args_from_makefile(makefile_path, src_dir=str(src_dir))
        args_lists = gd.get_makefile_run_args(src_dir=str(src_dir))
        assert isinstance(args_lists, list), f"Args extraction failed for {target_name} ({model})"

        if not args_lists or all(not args for args in args_lists):
            empty_args[model].append(target_name)

    for model, names in empty_args.items():
        if names:
            warnings.warn(
                f"Targets with empty input arguments ({model}): {', '.join(sorted(names))}",
                RuntimeWarning,
            )

