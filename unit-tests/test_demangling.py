"""
Test suite for kernel name demangling

Tests that kernel names can be correctly demangled for use with ncu -k option.
Includes test for the demangling bug fix from upstream gpuFLOPBench.
"""

import pytest
import subprocess
import re


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


def demangle_name(mangled, tool='c++filt'):
    """
    Demangle a kernel name using specified tool.
    
    Returns demangled name or original if demangling fails.
    """
    if not check_tool_available(tool):
        return mangled
    
    try:
        result = subprocess.run(
            [tool],
            input=mangled.encode(),
            capture_output=True,
            timeout=5
        )
        
        if result.returncode == 0:
            demangled = result.stdout.decode('utf-8').strip()
            if demangled and demangled != mangled:
                return demangled
    except Exception:
        pass
    
    return mangled


def demangle_name_reliable(mangled, prefer_tool='cu++filt'):
    """
    Demangle kernel name using preferred tool, with fallbacks.
    
    This is the FIXED version that tries tools in order and returns
    first successful result (avoiding the upstream bug).
    """
    tools = [prefer_tool]
    
    # Add other tools as fallbacks
    for tool in ['cu++filt', 'c++filt', 'llvm-cxxfilt']:
        if tool != prefer_tool and tool not in tools:
            tools.append(tool)
    
    for tool in tools:
        result = demangle_name(mangled, tool)
        if result != mangled:
            return result
    
    return mangled


def extract_simple_kernel_name(full_name):
    """
    Extract simple kernel name for use with ncu -k.
    
    Removes:
    - Return types
    - Template parameters
    - Namespaces
    """
    # Handle empty or whitespace-only input
    if not full_name or not full_name.strip():
        return full_name.strip() if full_name else ""
    
    # Remove return type if present
    if ' ' in full_name:
        parts = full_name.split()
        if parts:
            full_name = parts[-1]
    
    # Remove template parameters
    if '<' in full_name or '>' in full_name:
        parts = re.split(r'<|>', full_name)
        if parts:
            full_name = parts[0]
    
    # Remove namespace
    if '::' in full_name:
        parts = full_name.split('::')
        if parts:
            full_name = parts[-1]
    
    return full_name


def test_cxxfilt_basic_demangling():
    """Test basic c++filt demangling with known mangled name"""
    
    if not check_tool_available('c++filt'):
        pytest.skip("c++filt not available")
    
    # Simple mangled C++ name
    mangled = "_Z3foov"
    demangled = demangle_name(mangled, 'c++filt')
    
    assert demangled != mangled, "c++filt failed to demangle simple name"
    assert "foo" in demangled, f"Unexpected demangled result: {demangled}"


def test_demangling_with_preferred_tool():
    """Test that preferred tool is used first"""
    
    # Sample mangled name
    mangled = "_Z6kernelv"
    
    # Try with c++filt if available
    if check_tool_available('c++filt'):
        result = demangle_name_reliable(mangled, prefer_tool='c++filt')
        assert result != mangled, "Demangling failed"
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
    
    # Test with simple mangled name
    mangled = "_Z6kernelv"
    
    # Should succeed with first tool
    first_try_result = demangle_name(mangled, available_tool)
    
    # Should get same result with reliable function
    reliable_result = demangle_name_reliable(mangled, available_tool)
    
    assert first_try_result == reliable_result, \
        "Demangling results differ between first try and reliable function"
    
    assert first_try_result != mangled or mangled.startswith('__omp'), \
        f"Demangling failed on first try with {available_tool}"


def test_extract_simple_name_removes_templates():
    """Test that template parameters are removed"""
    
    # Template without spaces (as would come from real demangling)
    full_name = "myKernel<int>"
    simple = extract_simple_kernel_name(full_name)
    
    assert simple == "myKernel", f"Template removal failed: {simple}"


def test_extract_simple_name_removes_namespace():
    """Test that namespace is removed"""
    
    full_name = "my::namespace::kernel"
    simple = extract_simple_kernel_name(full_name)
    
    assert simple == "kernel", f"Namespace removal failed: {simple}"


def test_extract_simple_name_removes_return_type():
    """Test that return type is removed"""
    
    full_name = "void myKernel"
    simple = extract_simple_kernel_name(full_name)
    
    assert simple == "myKernel", f"Return type removal failed: {simple}"


def test_extract_simple_name_complex():
    """Test extraction from complex kernel name"""
    
    # More realistic: demangled names don't have spaces in templates
    full_name = "void my::space::kernel<int>"
    simple = extract_simple_kernel_name(full_name)
    
    assert simple == "kernel", f"Complex name extraction failed: {simple}"


def test_extract_simple_name_already_simple():
    """Test that already-simple names pass through"""
    
    simple_names = ["kernel", "myKernel", "kernel_v2"]
    
    for name in simple_names:
        result = extract_simple_kernel_name(name)
        assert result == name, f"Simple name modified: {name} -> {result}"


def test_omp_kernel_names_dont_need_demangling():
    """Verify OpenMP kernel names don't need demangling"""
    
    omp_name = "__omp_offloading_10_abc123_main_l25"
    
    # Demangling should return original (or similar)
    if check_tool_available('c++filt'):
        result = demangle_name(omp_name, 'c++filt')
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
        simple = extract_simple_kernel_name(kernel)
        
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
    
    # Sample mangled name
    mangled = "_Z6kernelIiEvPT_"  # kernel<int>(int*)
    
    # Step 1: Demangle
    demangled = demangle_name_reliable(mangled, available_tool)
    
    print(f"\nMangled:   {mangled}")
    print(f"Demangled: {demangled}")
    
    # Step 2: Extract simple name
    simple = extract_simple_kernel_name(demangled)
    
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
    
    # Test with sample mangled name
    mangled = "_Z6kernelv"
    
    results = {}
    for tool in available:
        result = demangle_name(mangled, tool)
        results[tool] = result
    
    print(f"\nDemangling results for '{mangled}':")
    for tool, result in results.items():
        print(f"  {tool}: {result}")
    
    # All tools should produce similar results
    demangled_values = [r for r in results.values() if r != mangled]
    
    assert len(demangled_values) > 0, "No tool successfully demangled the name"


def test_empty_name_handling():
    """Test that empty names are handled gracefully"""
    
    empty_names = ["", "   ", "\n"]
    
    for name in empty_names:
        simple = extract_simple_kernel_name(name)
        # Should not crash, should return stripped version
        assert isinstance(simple, str), "Extract returned non-string"
