#!/usr/bin/env python3
"""
Utility helpers for kernel discovery and demangling.

These functions are shared by profiling and unit tests.
"""

import os
import re
import subprocess
import shlex


def demangle_kernel_name(mangled_name, prefer_tool='cu++filt'):
    """
    Demangle a kernel name using cu++filt, c++filt, or llvm-cxxfilt.

    This fixes the upstream bug by trying demangling tools in the correct order
    and returning the first successful result.

    Args:
        mangled_name: The mangled kernel name
        prefer_tool: Preferred demangling tool ('cu++filt', 'c++filt', 'llvm-cxxfilt')

    Returns:
        Demangled kernel name or original if demangling fails
    """
    tools = [prefer_tool]
    if prefer_tool != 'cu++filt':
        tools.append('cu++filt')
    if prefer_tool != 'c++filt':
        tools.append('c++filt')
    if prefer_tool != 'llvm-cxxfilt':
        tools.append('llvm-cxxfilt')

    for tool in tools:
        try:
            # Check if tool exists
            result = subprocess.run(['which', tool], capture_output=True, timeout=5)
            if result.returncode != 0:
                continue

            # Try to demangle
            result = subprocess.run(
                [tool],
                input=mangled_name.encode(),
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                demangled = result.stdout.decode('utf-8').strip()
                if demangled and demangled != mangled_name:
                    return demangled
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            continue

    # If all tools fail, return original
    return mangled_name


def demangle_omp_offload_name(omp_symbol, prefer_tool='llvm-cxxfilt'):
    """
    Demangle an OpenMP offload entry name and preserve line number.

    Example:
        __omp_offloading_7f_437f7__Z20compact_cell_centric9full_data12compact_dataiPPc_l148
        -> compact_cell_centric(full_data, compact_data, int, char**):l148
    """
    if not omp_symbol:
        return omp_symbol

    line_tag = None
    line_match = re.match(r'^(.*)_l(\d+)$', omp_symbol)
    if line_match:
        omp_symbol = line_match.group(1)
        line_tag = f"l{line_match.group(2)}"

    mangled_part = omp_symbol.split('__')[-1] if '__' in omp_symbol else omp_symbol
    demangled = demangle_kernel_name(mangled_part, prefer_tool=prefer_tool)

    if demangled == mangled_part:
        demangled = mangled_part

    if line_tag:
        return f"{demangled}:{line_tag}"

    return demangled


def is_library_kernel(demangled_name):
    """Return True if the demangled name is from common CUDA libraries."""
    if not demangled_name:
        return False

    library_markers = [
        'cub::',
        'thrust::',
        '__cuda_'
    ]

    return any(marker in demangled_name for marker in library_markers)


def get_cuobjdump_kernels(target):
    """Extract CUDA kernel names from executable using cuobjdump"""
    targetName = target['targetName']
    srcDir = target['src']
    exe_path = target['exe']

    # Try cuobjdump first
    try:
        result = subprocess.run(
            ['cuobjdump', '--list-text', exe_path],
            cwd=srcDir,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        if result.returncode == 0:
            output = result.stdout.decode('UTF-8')

            # Extract kernel names with regex
            # cuobjdump output formats vary across CUDA/clang versions. Common forms:
            #   "SASS text section 3 : x-_Z15accuracy_kerneliiiPKfPKiPi.sm_86.elf.bin"
            #   "SASS text section 1 : x-void myKernel(int*) (sm_80)"
            #   "SASS text section 1 : x-_Z6kernelv [clone]"
            raw_names = []

            patterns = [
                # Capture mangled name before .sm_XX.elf.bin or [clone]
                r': x-([^\s]+?)(?=\.sm_[0-9A-Za-z]+\.elf\.bin| \[clone\]|$)',
                # Capture name after "x-void" for verbose output formats
                r': x-void\s+([^\s\(]+)',
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, output, re.MULTILINE)
                for m in matches:
                    name = m.group(1).strip()
                    if name:
                        raw_names.append(name)

            # Fallback for older outputs with .text.<symbol>
            if not raw_names:
                text_pattern = r'\.text\.[\w<>:,\s\*\-]+(?=\s|$)'
                matches = re.finditer(text_pattern, output, re.MULTILINE)
                for m in matches:
                    name = m.group().replace('.text.', '').strip()
                    if name:
                        raw_names.append(name)

            return raw_names
    except Exception as e:
        print(f"cuobjdump failed for {targetName}: {e}")

    # Try llvm-objdump as fallback
    try:
        # Note: Using shell=True here for pipe, but inputs are from filesystem, not user input
        # exe_path is validated to be an existing file in our build directory
        result = subprocess.run(
            ['sh', '-c', f'llvm-objdump -t {shlex.quote(exe_path)} | c++filt'],
            cwd=srcDir,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        if result.returncode == 0:
            output = result.stdout.decode('UTF-8')
            # Look for kernel-like symbols
            kernel_pattern = r'\.text\.[\w<>:,\s\*]+(?=\s|$)'
            matches = re.finditer(kernel_pattern, output, re.MULTILINE)
            return [m.group().replace('.text.', '') for m in matches if m.group()]
    except Exception as e:
        print(f"llvm-objdump failed for {targetName}: {e}")

    return []


def parse_omp_offload_entries(objdump_output):
    """Parse OpenMP offload entry names from objdump output text."""
    patterns = [
        r'(?<=\.offloading\.entry\.)(__omp_offloading.*)(?=\n)',
        r'(?<=\.omp_offloading\.entry\.)(__omp_offloading.*)(?=\n)'
    ]

    kernel_names = []
    for pattern in patterns:
        matches = re.finditer(pattern, objdump_output, re.MULTILINE)
        kernel_names.extend([m.group() for m in matches if m.group()])

    return kernel_names


def get_objdump_kernels(target):
    """Extract OpenMP kernel names from executable using objdump"""
    targetName = target['targetName']
    srcDir = target['src']
    exe_path = target['exe']

    sections = ['llvm_offload_entries', 'omp_offloading_entries']

    for section in sections:
        try:
            result = subprocess.run(
                ['objdump', '-t', '--section', section, exe_path],
                cwd=srcDir,
                timeout=60,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )

            if result.returncode != 0:
                continue

            output = result.stdout.decode('UTF-8')
            raw_names = parse_omp_offload_entries(output)

            if not raw_names:
                continue

            return raw_names

        except Exception as e:
            print(f"objdump failed for {targetName}: {e}")
            continue

        print(f"WARNING: No OMP offload entries found in {targetName}")
    return []


def extract_kernel_name_for_ncu(full_kernel_name):
    """
    Extract the simple kernel name for use with ncu -k option.

    Handles templates and namespaces by extracting the core function name.
    """
    if not full_kernel_name:
        return ''

    name = full_kernel_name.strip()
    if not name:
        return ''

    # Drop parameter list if present: "void kernel<int>(T1 *)" -> "void kernel<int>"
    if '(' in name:
        name = name.split('(', 1)[0].strip()

    # Remove return type: "void my::kernel<int>" -> "my::kernel<int>"
    if ' ' in name:
        parts = [p for p in re.split(r'\s+', name) if p]
        name = parts[-1] if parts else ''

    if not name:
        return ''

    # Remove templates: "kernel<int>" -> "kernel"
    if '<' in name:
        name = name.split('<', 1)[0]

    # Remove namespaces: "my::space::kernel" -> "kernel"
    if '::' in name:
        name = name.split('::')[-1]

    return name.strip()


def source_has_cuda_kernels(src_dir):
    """
    Best-effort check whether CUDA sources define kernels.

    Returns True if any source file contains __global__ or __device__.
    """
    if not src_dir or not os.path.isdir(src_dir):
        return False

    patterns = ["__global__", "__device__"]
    extensions = (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp")

    try:
        for root, _, files in os.walk(src_dir):
            for name in files:
                if not name.endswith(extensions):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, 'r', errors='ignore') as f:
                        contents = f.read()
                        if any(pat in contents for pat in patterns):
                            return True
                except OSError:
                    continue
    except Exception:
        return False

    return False


def exe_has_cuda_kernels(target):
    """
    Check if CUDA executable contains any profiliable (non-library) kernels.
    """
    if not target:
        return False

    raw_names = get_cuobjdump_kernels(target)
    if not raw_names:
        return False

    for name in raw_names:
        demangled = demangle_kernel_name(name)
        if is_library_kernel(demangled):
            continue
        profiler = extract_kernel_name_for_ncu(demangled)
        if profiler:
            return True

    return False