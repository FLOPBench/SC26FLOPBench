from __future__ import annotations

from importlib import util
import sys
from pathlib import Path
from textwrap import dedent


def _load_cst_utils():
    import types

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "langchain-tools" / "treesitter-tools" / "cst_utils.py"
    package_name = "treesitter_tools"
    spec = util.spec_from_file_location(f"{package_name}.cst_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load cst_utils from langchain-tools/treesitter-tools")
    package_module = sys.modules.get(package_name)
    if package_module is None:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(module_path.parent)]
        sys.modules[package_name] = package_module
    module = util.module_from_spec(spec)
    module.__spec__ = spec
    module.__package__ = package_name
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]

cst_utils = _load_cst_utils()


def _write_sample_source(tmp_path: Path) -> Path:
    content = dedent(
        """\
        #include <cstdio>

        __global__ void kernel(int value) {
            printf("%d\\n", value);
        }

        void host() {
            kernel<<<1, 1>>>(42);
        }
        """
    )
    path = tmp_path / "sample.cu"
    path.write_text(content)
    return path


def test_read_text_refreshes_when_file_updates(tmp_path: Path) -> None:
    path = _write_sample_source(tmp_path)
    initial = cst_utils.read_text(path)
    assert "__global__ void kernel" in initial
    addition = "// injected comment"
    path.write_text(initial + "\n" + addition)
    refreshed = cst_utils.read_text(path)
    assert addition in refreshed


def test_pick_language_for_file_handles_extensions() -> None:
    assert cst_utils.pick_language_for_file("foo.cu") == "cuda"
    assert cst_utils.pick_language_for_file("bar.cpp") == "cpp"
    assert cst_utils.pick_language_for_file("baz.txt", default="cpp") == "cpp"


def test_find_enclosing_function_locates_host(tmp_path: Path) -> None:
    path = _write_sample_source(tmp_path)
    text = cst_utils.read_text(path)
    tree = cst_utils.parse_file(path)
    launch_index = text.index("kernel<<<")
    byte_offset = len(text[:launch_index].encode("utf-8"))
    func_ref = cst_utils.find_enclosing_function(tree, byte_offset)
    assert func_ref is not None
    func_span = cst_utils.ref_to_span(path, func_ref, text)
    span_text = cst_utils.span_text(text, func_span)
    assert "void host" in span_text


def test_cuda_launch_detection_and_extraction(tmp_path: Path) -> None:
    path = _write_sample_source(tmp_path)
    text = cst_utils.read_text(path)
    tree = cst_utils.parse_file(path)
    lines = text.splitlines()
    target_line = next(i for i, line in enumerate(lines, start=1) if "kernel<<<" in line)
    launches = cst_utils.collect_cuda_launches_on_line(tree, text, target_line)
    assert len(launches) == 1
    launch_parts = cst_utils.extract_cuda_launch_parts(path, text, launches[0])
    assert launch_parts["kernel_name_text"] == "kernel"
    cfg_span = launch_parts["launch_cfg_span"]
    assert cfg_span is not None
    cfg_text = cst_utils.span_text(text, cfg_span)
    assert "<<<" in cfg_text and ">>>" in cfg_text
    arg_span = launch_parts["arg_span"]
    assert arg_span is not None
    arg_text = cst_utils.span_text(text, arg_span)
    assert "(42)" in arg_text


def test_find_cuda_launches_on_line_helper(tmp_path: Path) -> None:
    path = _write_sample_source(tmp_path)
    text = cst_utils.read_text(path)
    line = next(i for i, val in enumerate(text.splitlines(), start=1) if "kernel<<<" in val)
    launches = cst_utils.find_cuda_launches_on_line(str(path), line=line)
    assert len(launches) == 1
    entry = launches[0]
    assert entry["kernel_name_text"] == "kernel"
    assert entry["full_launch_span"]["text"].strip().startswith("kernel")
    assert "launch_cfg_span" in entry and entry["launch_cfg_span"] is not None


def _write_openmp_source(tmp_path: Path) -> Path:
    content = dedent(
        """\
        #include <omp.h>

        void host_loop() {
        #pragma omp parallel for num_threads(4)
            for (int i = 0; i < 16; ++i) {
                printf("%d\\n", i);
            }
        }
        """
    )
    path = tmp_path / "openmp.cu"
    path.write_text(content)
    return path


def test_find_omp_pragmas_and_region(tmp_path: Path) -> None:
    path = _write_openmp_source(tmp_path)
    text = cst_utils.read_text(path)
    line = next(i for i, val in enumerate(text.splitlines(), start=1) if "#pragma omp" in val)
    spans = cst_utils.find_omp_pragmas_near_line(path, text, line)
    assert spans
    span = spans[0]
    assert "#pragma omp" in cst_utils.span_text(text, span)
    tree = cst_utils.parse_file(path)
    region = cst_utils.collect_omp_region(path, text, tree, span)
    assert region is not None
    assert "parallel" in region.kind_text
    assert "for" in region.clauses
    stmt_text = cst_utils.span_text(text, region.associated_stmt_span)
    assert "for (int i" in stmt_text


def test_parse_omp_pragma_raw() -> None:
    kind, clauses = cst_utils.parse_omp_pragma("#pragma omp parallel for num_threads(4)")
    assert kind == "parallel"
    assert "for" in clauses
    assert "num_threads(4)" in clauses


def test_build_omp_region_helper(tmp_path: Path) -> None:
    path = _write_openmp_source(tmp_path)
    text = cst_utils.read_text(path)
    line = next(i for i, val in enumerate(text.splitlines(), start=1) if "#pragma omp" in val)
    results = cst_utils.build_omp_region(str(path), line=line)
    assert results
    entry = results[0]
    assert entry["kind_text"] == "parallel"
    assert entry["clauses"] and "for" in entry["clauses"]
    assert "pragma_text" in entry and "#pragma omp" in entry["pragma_text"]


def test_lulesh_cuda_launch_detection_real_code() -> None:
    cuda_file = (
        REPO_ROOT / ".." / "HeCBench" / "src" / "lulesh-cuda" / "lulesh.cu"
    )
    text = cst_utils.read_text(cuda_file)
    line = next(
        i
        for i, line_text in enumerate(text.splitlines(), start=1)
        if "fill_sig<<<" in line_text
    )
    tree = cst_utils.parse_file(cuda_file)
    launches = cst_utils.collect_cuda_launches_on_line(tree, text, line)
    assert launches, "fill_sig launch not detected in lulesh.cu"
    launch_ref = launches[0]
    parts = cst_utils.extract_cuda_launch_parts(cuda_file, text, launch_ref)
    assert parts["kernel_name_text"] == "fill_sig"
    full_text = cst_utils.span_text(text, parts["full_launch_span"])
    assert "fill_sig<<<gws_elem, lws>>>" in full_text
    cfg_span = parts["launch_cfg_span"]
    assert cfg_span is not None
    cfg_text = cst_utils.span_text(text, cfg_span)
    assert "<<<gws_elem, lws>>>" in cfg_text
    arg_span = parts["arg_span"]
    assert arg_span is not None
    arg_text = cst_utils.span_text(text, arg_span)
    assert "d_sigxx" in arg_text


def test_find_cuda_launches_on_line_real_code() -> None:
    cuda_file = (
        REPO_ROOT / ".." / "HeCBench" / "src" / "lulesh-cuda" / "lulesh.cu"
    )
    text = cst_utils.read_text(cuda_file)
    line = next(
        i
        for i, line_text in enumerate(text.splitlines(), start=1)
        if "fill_sig<<<" in line_text
    )
    launches = cst_utils.find_cuda_launches_on_line(str(cuda_file), line=line)
    assert launches, "Expected find_cuda_launches_on_line to return entries"
    entry = launches[0]
    assert entry["kernel_name_text"] == "fill_sig"
    assert entry["launch_cfg_span"] is not None
    assert "<<<" in entry["launch_cfg_span"]["text"]


def test_lulesh_omp_region_detection_real_code() -> None:
    omp_file = REPO_ROOT / ".." / "HeCBench" / "src" / "lulesh-omp" / "lulesh.cc"
    text = cst_utils.read_text(omp_file)
    target_directive = "#pragma omp target teams distribute parallel for thread_limit(THREADS)"
    line = next(
        i
        for i, line_text in enumerate(text.splitlines(), start=1)
        if target_directive in line_text
    )
    regions = cst_utils.build_omp_region(str(omp_file), line=line)
    assert regions, "Expected build_omp_region to find at least one region"
    entry = regions[0]
    assert entry["kind_text"] == "target"
    assert "teams" in entry["clauses"]
    assert "parallel" in entry["clauses"]
    assert "for" in entry["clauses"]
    assert target_directive in entry["pragma_text"]
    stmt_text = entry["associated_stmt_span"]["text"].strip()
    assert stmt_text.startswith("for (Index_t i")
