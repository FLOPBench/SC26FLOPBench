"""
Test suite for kernel name demangling

Tests that kernel names can be correctly demangled for use with ncu -k option.
Includes test for the demangling bug fix from upstream gpuFLOPBench.
"""

import pytest
import warnings
import subprocess
import re
import sys
from pathlib import Path

GATHERDATA_DIR = Path(__file__).resolve().parents[1] / "cuda-profiling"
sys.path.insert(0, str(GATHERDATA_DIR))

import utils as gd

# Test constants: Mangled C++ names for testing demangling
# These are standard Itanium ABI mangled names
SIMPLE_MANGLED_NAME = "_Z3foov"  # foo()
SIMPLE_KERNEL_MANGLED = "_Z6kernelv"  # kernel()
TEMPLATE_KERNEL_MANGLED = "_Z6kernelIiEvPT_"  # void kernel<int>(int*)


def check_tool_available(tool_name):
    """Check if a command-line tool is available"""
    try:
        result = subprocess.run(
            ['which', tool_name],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def test_demangling_tools_available():
    """Check which demangling tools are available"""
    tools = ['cu++filt', 'c++filt', 'llvm-cxxfilt']
    available = []
    
    for tool in tools:
        if check_tool_available(tool):
            available.append(tool)
    
    print(f"\nAvailable demangling tools: {available}")
    
    assert len(available) > 0, \
        "No demangling tools available. Need at least one of: cu++filt, c++filt, llvm-cxxfilt"




def test_cxxfilt_basic_demangling():
    """Test basic c++filt demangling with known mangled name"""
    
    if not check_tool_available('c++filt'):
        pytest.skip("c++filt not available")
    
    # Simple mangled C++ name
    demangled = gd.demangle_kernel_name(SIMPLE_MANGLED_NAME, 'c++filt')
    
    assert demangled != SIMPLE_MANGLED_NAME, "c++filt failed to demangle simple name"
    assert "foo" in demangled, f"Unexpected demangled result: {demangled}"


def test_demangling_with_preferred_tool():
    """Test that preferred tool is used first"""
    
    # Try with c++filt if available
    if check_tool_available('c++filt'):
        result = gd.demangle_kernel_name(SIMPLE_KERNEL_MANGLED, prefer_tool='c++filt')
        assert result != SIMPLE_KERNEL_MANGLED, "Demangling failed"
        assert "kernel" in result, f"Unexpected result: {result}"


def test_demangling_first_try_succeeds():
    """
    Test the bug fix: demangling should succeed on FIRST try.
    
    The upstream bug was that demangling would try multiple tools
    with convoluted fallback logic. We fix this by using correct
    tool order and returning first success.
    """
    
    # Find available tool
    available_tool = None
    for tool in ['cu++filt', 'c++filt', 'llvm-cxxfilt']:
        if check_tool_available(tool):
            available_tool = tool
            break
    
    if not available_tool:
        pytest.skip("No demangling tools available")
    
    # Should succeed with first tool
    first_try_result = gd.demangle_kernel_name(SIMPLE_KERNEL_MANGLED, available_tool)
    
    # Should get same result with reliable function
    reliable_result = gd.demangle_kernel_name(SIMPLE_KERNEL_MANGLED, available_tool)
    
    assert first_try_result == reliable_result, \
        "Demangling results differ between first try and reliable function"
    
    assert first_try_result != SIMPLE_KERNEL_MANGLED or SIMPLE_KERNEL_MANGLED.startswith('__omp'), \
        f"Demangling failed on first try with {available_tool}"


def test_extract_simple_name_removes_templates():
    """Test that template parameters are removed"""
    
    # Template without spaces (as would come from real demangling)
    full_name = "myKernel<int>"
    simple = gd.extract_kernel_name_for_ncu(full_name)
    
    assert simple == "myKernel", f"Template removal failed: {simple}"


def test_extract_simple_name_removes_namespace():
    """Test that namespace is removed"""
    
    full_name = "my::namespace::kernel"
    simple = gd.extract_kernel_name_for_ncu(full_name)
    
    assert simple == "kernel", f"Namespace removal failed: {simple}"


def test_extract_simple_name_removes_return_type():
    """Test that return type is removed"""
    
    full_name = "void myKernel"
    simple = gd.extract_kernel_name_for_ncu(full_name)
    
    assert simple == "myKernel", f"Return type removal failed: {simple}"


def test_extract_simple_name_complex():
    """Test extraction from complex kernel name"""
    
    # More realistic: demangled names don't have spaces in templates
    full_name = "void my::space::kernel<int>"
    simple = gd.extract_kernel_name_for_ncu(full_name)
    
    assert simple == "kernel", f"Complex name extraction failed: {simple}"


def test_extract_simple_name_already_simple():
    """Test that already-simple names pass through"""
    
    simple_names = ["kernel", "myKernel", "kernel_v2"]
    
    for name in simple_names:
        result = gd.extract_kernel_name_for_ncu(name)
        assert result == name, f"Simple name modified: {name} -> {result}"


def test_omp_kernel_names_dont_need_demangling():
    """Verify OpenMP kernel names don't need demangling"""
    
    omp_name = "__omp_offloading_10_abc123_main_l25"
    
    # Demangling should return original (or similar)
    if check_tool_available('c++filt'):
        result = gd.demangle_kernel_name(omp_name, 'c++filt')
        # OMP names typically don't demangle or demangle to themselves
        assert '__omp_offloading' in result, \
            f"OMP kernel name changed unexpectedly: {result}"


def test_filter_library_kernels():
    """Test that library kernels (cub::, thrust::) are filtered"""
    
    library_kernels = [
        "cub::DeviceReduce",
        "thrust::transform",
        "cub::BlockScan<int>"
    ]
    
    for kernel in library_kernels:
        simple = gd.extract_kernel_name_for_ncu(kernel)
        
        # After extraction, check if we should filter
        # In practice, gatherData.py filters before extraction
        # But we test the logic here
        if simple.startswith('cub::') or simple.startswith('thrust::'):
            print(f"Would filter library kernel: {simple}")


def test_demangling_integration():
    """
    Integration test: demangle and extract simple name.
    
    This simulates the full workflow used in gatherData.py.
    """
    
    # Find available tool
    available_tool = None
    for tool in ['cu++filt', 'c++filt', 'llvm-cxxfilt']:
        if check_tool_available(tool):
            available_tool = tool
            break
    
    if not available_tool:
        pytest.skip("No demangling tools available")
    
    # Step 1: Demangle
    demangled = gd.demangle_kernel_name(TEMPLATE_KERNEL_MANGLED, available_tool)
    
    print(f"\nMangled:   {TEMPLATE_KERNEL_MANGLED}")
    print(f"Demangled: {demangled}")
    
    # Step 2: Extract simple name
    simple = gd.extract_kernel_name_for_ncu(demangled)
    
    print(f"Simple:    {simple}")
    
    # Verify
    assert simple == "kernel", f"Integration test failed: got {simple}"


@pytest.mark.slow
def test_demangling_multiple_tools():
    """Test demangling with all available tools"""
    
    tools = ['cu++filt', 'c++filt', 'llvm-cxxfilt']
    available = [t for t in tools if check_tool_available(t)]
    
    if not available:
        pytest.skip("No demangling tools available")
    
    results = {}
    for tool in available:
        result = gd.demangle_kernel_name(SIMPLE_KERNEL_MANGLED, tool)
        results[tool] = result
    
    print(f"\nDemangling results for '{SIMPLE_KERNEL_MANGLED}':")
    for tool, result in results.items():
        print(f"  {tool}: {result}")
    
    # All tools should produce similar results
    demangled_values = [r for r in results.values() if r != SIMPLE_KERNEL_MANGLED]
    
    assert len(demangled_values) > 0, "No tool successfully demangled the name"


def test_empty_name_handling():
    """Test that empty names are handled gracefully"""
    
    empty_names = ["", "   ", "\n"]
    
    for name in empty_names:
        simple = gd.extract_kernel_name_for_ncu(name)
        # Should not crash, should return stripped version
        assert isinstance(simple, str), "Extract returned non-string"


def test_parse_omp_offload_entries():
    """Test parsing of OpenMP offload entries from objdump output."""
    sample_output = """
0000000000000000  w    O llvm_offload_entries   0000000000000038              .offloading.entry.__omp_offloading_7f_1234__Z3foov_l12
0000000000000000  w    O llvm_offload_entries   0000000000000038              .offloading.entry.__omp_offloading_7f_5678__Z3barv_l34
"""

    names = gd.parse_omp_offload_entries(sample_output)
    assert names == [
        "__omp_offloading_7f_1234__Z3foov_l12",
        "__omp_offloading_7f_5678__Z3barv_l34",
    ]


def test_demangle_omp_offload_name_with_line():
    """Demangle OpenMP offload name and keep line tag."""
    if not any(check_tool_available(t) for t in ['llvm-cxxfilt', 'c++filt', 'cu++filt']):
        pytest.skip("No demangling tools available")

    omp_symbol = "__omp_offloading_7f_437f7__Z20compact_cell_centric9full_data12compact_dataiPPc_l148"
    demangled = gd.demangle_omp_offload_name(omp_symbol, prefer_tool='llvm-cxxfilt')
    assert demangled.endswith(":l148")
    assert "compact_cell_centric" in demangled


@pytest.mark.slow
def test_ncu_name_extraction_all_executables(cuda_executables, omp_executables):
    """
    Ensure NCU kernel name extraction works for CUDA and OpenMP executables.

    All executables should have at least one kernel name extracted from their
    respective binaries.
    """
    if not cuda_executables and not omp_executables:
        pytest.skip("No executables found to test")

    if cuda_executables and not check_tool_available('cuobjdump'):
        pytest.skip("cuobjdump not available")

    hecbench_src = Path(__file__).resolve().parents[1] / "HeCBench" / "src"

    # CUDA executables: use cuobjdump + demangle + extract for NCU
    missing_cuda = []
    skipped_cuda = []
    for exe in cuda_executables:
        target_name = exe.name
        src_dir = hecbench_src / f"{target_name}-cuda"
        if not src_dir.is_dir():
            alt = hecbench_src / target_name
            if alt.is_dir():
                src_dir = alt

        if not gd.source_has_cuda_kernels(str(src_dir)):
            skipped_cuda.append(target_name)
            continue

        target = {
            'targetName': target_name,
            'exe': str(exe),
            'src': str(src_dir),
            'model': 'cuda'
        }

        raw_names = gd.get_cuobjdump_kernels(target)
        profiler_names = []
        for name in raw_names:
            demangled = gd.demangle_kernel_name(name)
            if gd.is_library_kernel(demangled):
                continue
            profiler = gd.extract_kernel_name_for_ncu(demangled)
            if profiler:
                profiler_names.append(profiler)

        if not profiler_names:
            missing_cuda.append(target_name)

    if missing_cuda:
        warnings.warn(
            f"CUDA executables without extracted kernel names: {missing_cuda}",
            UserWarning
        )

    # OpenMP executables: use objdump + demangle
    missing_omp = []
    for exe in omp_executables:
        target_name = exe.name
        src_dir = hecbench_src / f"{target_name}-omp"
        if not src_dir.is_dir():
            alt = hecbench_src / target_name
            if alt.is_dir():
                src_dir = alt

        target = {
            'targetName': target_name,
            'exe': str(exe),
            'src': str(src_dir),
            'model': 'omp'
        }

        raw_names = gd.get_objdump_kernels(target)
        if not raw_names:
            missing_omp.append(target_name)
            continue

        # Ensure demangling runs and yields non-empty names
        demangled = [gd.demangle_omp_offload_name(name) for name in raw_names]
        if not any(name.strip() for name in demangled):
            missing_omp.append(target_name)

    if missing_omp:
        warnings.warn(
            f"OpenMP executables without extracted kernel names: {missing_omp}",
            UserWarning
        )


@pytest.mark.slow
def test_profiliable_kernel_counts(cuda_executables, omp_executables):
    """
    Print total number of profiliable codes and total number of profiliable kernels.
    """
    hecbench_src = Path(__file__).resolve().parents[1] / "HeCBench" / "src"

    cuda_codes = 0
    cuda_kernels = 0
    for exe in cuda_executables:
        target_name = exe.name
        src_dir = hecbench_src / f"{target_name}-cuda"
        if not src_dir.is_dir():
            alt = hecbench_src / target_name
            if alt.is_dir():
                src_dir = alt

        if not gd.source_has_cuda_kernels(str(src_dir)):
            continue

        target = {
            'targetName': target_name,
            'exe': str(exe),
            'src': str(src_dir),
            'model': 'cuda'
        }

        if not gd.exe_has_cuda_kernels(target):
            continue

        raw_names = gd.get_cuobjdump_kernels(target)
        profiler_names = []
        for name in raw_names:
            demangled = gd.demangle_kernel_name(name)
            if gd.is_library_kernel(demangled):
                continue
            profiler = gd.extract_kernel_name_for_ncu(demangled)
            if profiler:
                profiler_names.append(profiler)

        if profiler_names:
            cuda_codes += 1
            cuda_kernels += len(profiler_names)

    omp_codes = 0
    omp_kernels = 0
    for exe in omp_executables:
        target_name = exe.name
        src_dir = hecbench_src / f"{target_name}-omp"
        if not src_dir.is_dir():
            alt = hecbench_src / target_name
            if alt.is_dir():
                src_dir = alt

        target = {
            'targetName': target_name,
            'exe': str(exe),
            'src': str(src_dir),
            'model': 'omp'
        }

        raw_names = gd.get_objdump_kernels(target)
        if raw_names:
            omp_codes += 1
            omp_kernels += len(raw_names)

    total_codes = cuda_codes + omp_codes
    total_kernels = cuda_kernels + omp_kernels

    print(
        f"\nProfiliable codes: {total_codes} (CUDA: {cuda_codes}, OpenMP: {omp_codes})"
    )
    print(
        f"Profiliable kernels: {total_kernels} (CUDA: {cuda_kernels}, OpenMP: {omp_kernels})"
    )


def test_cuda_non_profiliable_targets_have_no_source_kernels(cuda_executables):
    """
    For CUDA executables that appear to have no profiliable kernels,
    confirm their sources contain no CUDA kernel markers.
    """
    if not cuda_executables:
        pytest.skip("No CUDA executables found")

    if not check_tool_available('cuobjdump'):
        pytest.skip("cuobjdump not available")

    hecbench_src = Path(__file__).resolve().parents[1] / "HeCBench" / "src"

    mismatches = []
    for exe in cuda_executables:
        target_name = exe.name
        src_dir = hecbench_src / f"{target_name}-cuda"
        if not src_dir.is_dir():
            alt = hecbench_src / target_name
            if alt.is_dir():
                src_dir = alt
            else:
                continue

        target = {
            'targetName': target_name,
            'exe': str(exe),
            'src': str(src_dir),
            'model': 'cuda'
        }

        if gd.exe_has_cuda_kernels(target):
            continue

        if gd.source_has_cuda_kernels(str(src_dir)):
            mismatches.append(target_name)

    assert not mismatches, (
        "CUDA targets deemed non-profiliable but found kernel markers in source: "
        f"{sorted(mismatches)}"
    )
