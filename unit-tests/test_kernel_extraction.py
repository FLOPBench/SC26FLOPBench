"""
Test suite for kernel extraction from binaries

Tests that GPU kernels can be extracted from CUDA and OpenMP binaries.
"""

import pytest
import subprocess
import re
from pathlib import Path


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


def test_cuobjdump_available():
    """Verify cuobjdump is available for CUDA kernel extraction"""
    if not check_tool_available('cuobjdump'):
        pytest.skip("cuobjdump not available (CUDA toolkit not installed)")
    
    # If available, verify it works
    result = subprocess.run(
        ['cuobjdump', '--help'],
        capture_output=True,
        timeout=5
    )
    assert result.returncode == 0, "cuobjdump exists but failed to run"


def test_objdump_available():
    """Verify objdump is available for OpenMP kernel extraction"""
    assert check_tool_available('objdump'), \
        "objdump not available (binutils not installed)"
    
    # Verify it works
    result = subprocess.run(
        ['objdump', '--version'],
        capture_output=True,
        timeout=5
    )
    assert result.returncode == 0, "objdump exists but failed to run"


def test_llvm_objdump_available_or_skip():
    """Check if llvm-objdump is available (optional fallback)"""
    available = check_tool_available('llvm-objdump')
    if available:
        print("\nllvm-objdump is available (good for CUDA fallback)")
    else:
        print("\nllvm-objdump not available (optional)")


def extract_cuda_kernels_with_cuobjdump(exe_path):
    """
    Extract CUDA kernel names from binary using cuobjdump.
    
    Returns list of kernel names or empty list if extraction fails.
    """
    try:
        result = subprocess.run(
            ['cuobjdump', '--list-text', str(exe_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return []
        
        output = result.stdout
        
        # Pattern to match kernel names in cuobjdump output
        # Pattern explanation:
        #   \.text\. - Match literal ".text." prefix
        #   [\w<>:,\s\*\-]+ - Match kernel name characters (word chars, templates, namespaces, pointers)
        #   (?=\s|$) - Positive lookahead for whitespace or end of line
        pattern = r'\.text\.[\w<>:,\s\*\-]+(?=\s|$)'
        matches = re.finditer(pattern, output, re.MULTILINE)
        
        kernel_names = []
        for match in matches:
            name = match.group().replace('.text.', '').strip()
            if name:
                kernel_names.append(name)
        
        return kernel_names
    
    except Exception as e:
        print(f"cuobjdump failed: {e}")
        return []


def extract_omp_kernels_with_objdump(exe_path):
    """
    Extract OpenMP kernel names from binary using objdump.
    
    Returns list of kernel names or empty list if extraction fails.
    """
    try:
        result = subprocess.run(
            ['objdump', '-t', '--section=omp_offloading_entries', str(exe_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return []
        
        output = result.stdout
        
        # Pattern to match OMP offloading entry names
        # Pattern explanation:
        #   (?<=\.omp_offloading\.entry\.) - Positive lookbehind for entry prefix
        #   (__omp_offloading[^\s]+) - Capture __omp_offloading followed by non-whitespace
        pattern = r'(?<=\.omp_offloading\.entry\.)(__omp_offloading[^\s]+)'
        matches = re.finditer(pattern, output, re.MULTILINE)
        
        kernel_names = [m.group(1) for m in matches if m.group(1)]
        
        return kernel_names
    
    except Exception as e:
        print(f"objdump failed: {e}")
        return []


def test_cuda_kernel_extraction_sample(cuda_executables):
    """Test kernel extraction from sample CUDA executables"""
    
    if not cuda_executables:
        pytest.skip("No CUDA executables found to test")
    
    if not check_tool_available('cuobjdump'):
        pytest.skip("cuobjdump not available")
    
    # Test on first few CUDA executables
    sample = cuda_executables[:5]
    
    results = {}
    for exe in sample:
        kernels = extract_cuda_kernels_with_cuobjdump(exe)
        results[exe.name] = kernels
    
    # At least some should have kernels
    total_kernels = sum(len(k) for k in results.values())
    
    print(f"\nKernel extraction results:")
    for name, kernels in results.items():
        print(f"  {name}: {len(kernels)} kernel(s)")
        if kernels:
            print(f"    Sample: {kernels[0]}")
    
    assert total_kernels > 0, \
        "No kernels extracted from any CUDA executable. Check extraction logic."


def test_omp_kernel_extraction_sample(omp_executables):
    """Test kernel extraction from sample OpenMP executables"""
    
    if not omp_executables:
        pytest.skip("No OpenMP executables found to test")
    
    # Test on first few OMP executables
    sample = omp_executables[:5]
    
    results = {}
    for exe in sample:
        kernels = extract_omp_kernels_with_objdump(exe)
        results[exe.name] = kernels
    
    # At least some should have kernels
    total_kernels = sum(len(k) for k in results.values())
    
    print(f"\nOpenMP kernel extraction results:")
    for name, kernels in results.items():
        print(f"  {name}: {len(kernels)} kernel(s)")
        if kernels:
            print(f"    Sample: {kernels[0]}")
    
    assert total_kernels > 0, \
        "No kernels extracted from any OpenMP executable. Check extraction logic."


def test_cuda_kernel_names_not_empty(cuda_executables):
    """Verify extracted CUDA kernel names are non-empty strings"""
    
    if not cuda_executables:
        pytest.skip("No CUDA executables found")
    
    if not check_tool_available('cuobjdump'):
        pytest.skip("cuobjdump not available")
    
    exe = cuda_executables[0]
    kernels = extract_cuda_kernels_with_cuobjdump(exe)
    
    if not kernels:
        pytest.skip(f"No kernels found in {exe.name}")
    
    for kernel in kernels:
        assert isinstance(kernel, str), f"Kernel name is not a string: {kernel}"
        assert len(kernel) > 0, "Empty kernel name extracted"
        assert kernel.strip() == kernel, f"Kernel name has leading/trailing whitespace: '{kernel}'"


def test_omp_kernel_names_valid(omp_executables):
    """Verify extracted OpenMP kernel names follow expected pattern"""
    
    if not omp_executables:
        pytest.skip("No OpenMP executables found")
    
    exe = omp_executables[0]
    kernels = extract_omp_kernels_with_objdump(exe)
    
    if not kernels:
        pytest.skip(f"No kernels found in {exe.name}")
    
    # OMP kernel names should start with __omp_offloading
    for kernel in kernels:
        assert kernel.startswith('__omp_offloading'), \
            f"OMP kernel name doesn't match expected pattern: {kernel}"


@pytest.mark.slow
def test_all_cuda_executables_have_kernels(cuda_executables):
    """Verify all CUDA executables contain kernels (slow - tests all)"""
    
    if not cuda_executables:
        pytest.skip("No CUDA executables found")
    
    if not check_tool_available('cuobjdump'):
        pytest.skip("cuobjdump not available")
    
    no_kernels = []
    
    for exe in cuda_executables:
        kernels = extract_cuda_kernels_with_cuobjdump(exe)
        if not kernels:
            no_kernels.append(exe.name)
    
    # Report results
    print(f"\nTested {len(cuda_executables)} CUDA executables")
    print(f"Without kernels: {len(no_kernels)}")
    
    if no_kernels:
        print(f"\nExecutables without extracted kernels (first 10):")
        for name in no_kernels[:10]:
            print(f"  - {name}")
    
    # Allow some executables without kernels (might use external libs)
    success_rate = (len(cuda_executables) - len(no_kernels)) / len(cuda_executables) * 100
    
    assert success_rate > 50, \
        f"Too many executables without kernels: {success_rate:.1f}% success rate"
    
    print(f"\nKernel extraction success rate: {success_rate:.1f}%")


@pytest.mark.slow  
def test_all_omp_executables_have_kernels(omp_executables):
    """Verify all OpenMP executables contain kernels (slow - tests all)"""
    
    if not omp_executables:
        pytest.skip("No OpenMP executables found")
    
    no_kernels = []
    
    for exe in omp_executables:
        kernels = extract_omp_kernels_with_objdump(exe)
        if not kernels:
            no_kernels.append(exe.name)
    
    # Report results
    print(f"\nTested {len(omp_executables)} OpenMP executables")
    print(f"Without kernels: {len(no_kernels)}")
    
    if no_kernels:
        print(f"\nExecutables without extracted kernels (first 10):")
        for name in no_kernels[:10]:
            print(f"  - {name}")
    
    # Allow some executables without kernels
    success_rate = (len(omp_executables) - len(no_kernels)) / len(omp_executables) * 100
    
    assert success_rate > 50, \
        f"Too many executables without kernels: {success_rate:.1f}% success rate"
    
    print(f"\nKernel extraction success rate: {success_rate:.1f}%")
