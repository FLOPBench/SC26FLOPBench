#!/usr/bin/env python3
"""
Utility helpers for kernel discovery and demangling.

These functions are shared by profiling and unit tests.
"""

import os
import re
import subprocess
import shlex
from pathlib import Path


MAKEFILE_CANDIDATES = ("Makefile", "makefile", "GNUmakefile")


def find_makefile_for_target(src_dir):
    """
    Locate the Makefile for a benchmark source directory.

    Args:
        src_dir: Path to the benchmark source directory

    Returns:
        Absolute path to Makefile if found, else None
    """
    if not src_dir or not os.path.isdir(src_dir):
        return None

    for name in MAKEFILE_CANDIDATES:
        path = os.path.join(src_dir, name)
        if os.path.isfile(path):
            return path

    ignore_dirs = {".git", "build", "cmake", "CMakeFiles", "_deps"}
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for name in MAKEFILE_CANDIDATES:
            if name in files:
                return os.path.join(root, name)

    return None


def _file_has_run_target(lines):
    """Return True if the file contains a run target with at least one recipe line."""
    recipe = _collect_run_recipe_lines(lines)
    return len(recipe) > 0


def find_run_target_file(src_dir):
    """
    Search for a file containing a Make-style run target.

    This is a fallback for benchmarks that store run targets in non-Makefile
    include files (e.g., src/make_targets).
    """
    if not src_dir or not os.path.isdir(src_dir):
        return None

    ignore_dirs = {".git", "build", "cmake", "CMakeFiles", "_deps"}
    max_size = 2 * 1024 * 1024  # 2MB

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > max_size:
                    continue
            except OSError:
                continue

            try:
                with open(path, "r", errors="ignore") as f:
                    lines = f.readlines()
            except OSError:
                continue

            if _file_has_run_target(lines):
                return path

    return None


def _strip_make_comment(line):
    """Strip Makefile comments while respecting quotes."""
    if not line:
        return line

    in_single = False
    in_double = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line.rstrip()


def _collect_run_recipe_lines(lines):
    """Collect recipe lines under the run: target."""
    run_lines = []
    in_run = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not in_run:
            if re.match(r"^\s*run\s*:", line) and not stripped.startswith("#"):
                in_run = True
            i += 1
            continue

        if stripped == "":
            i += 1
            continue

        if re.match(r"^[^\s].*:\s*", line) and not re.match(r"^\s*run\s*:", line):
            break

        if re.match(r"^\s*#", line):
            i += 1
            continue

        if not re.match(r"^\s+", line):
            break

        run_lines.append(line)
        i += 1

    return run_lines


def _strip_recipe_prefix(line):
    """Remove recipe prefixes like @, -, + and leading whitespace."""
    cleaned = line.lstrip()
    while cleaned and cleaned[0] in ("@", "-", "+"):
        cleaned = cleaned[1:].lstrip()
    return cleaned


def _find_exec_index(tokens, exe_name=None):
    """Find the token index for the executable invocation."""
    for i, token in enumerate(tokens):
        if token.startswith("./"):
            return i
        if exe_name:
            if token == exe_name or token.endswith(f"/{exe_name}"):
                return i
        if token in ("$(program)", "$(PROGRAM)", "$(exe)", "$(EXE)", "$(target)", "$(TARGET)"):
            return i
        if token.startswith("./$(program)") or token.startswith("./$(PROGRAM)"):
            return i
        if token.endswith("/$(program)") or token.endswith("/$(PROGRAM)"):
            return i
    return None


def _find_hecbench_src_root(src_dir):
    """Locate the HeCBench/src root given a benchmark src directory."""
    if not src_dir:
        return None

    path = Path(src_dir).resolve()
    for parent in [path] + list(path.parents):
        if parent.name == "src":
            return str(parent)
    return None


def _is_numeric_token(token):
    return bool(re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", token))


def _resolve_arg_path(arg, src_dir, run_cwd=None):
    """Resolve input file arguments to paths relative to src_dir."""
    if not arg or arg.startswith("-"):
        return arg
    if _is_numeric_token(arg):
        return arg
    if "$(" in arg or "${" in arg:
        return arg

    if os.path.isabs(arg):
        return arg

    candidates = []
    if run_cwd:
        candidates.append(os.path.join(run_cwd, arg))
    if src_dir:
        candidates.append(os.path.join(src_dir, arg))

    for path in candidates:
        if os.path.exists(path):
            return os.path.relpath(path, src_dir)

    src_root = _find_hecbench_src_root(src_dir)
    if src_root:
        basename = os.path.basename(arg)
        for root, _, files in os.walk(src_root):
            if basename in files:
                path = os.path.join(root, basename)
                return os.path.relpath(path, src_dir)

    return arg


def extract_run_args_from_makefile(makefile_path, exe_name=None, src_dir=None):
    """
    Extract command-line arguments from the Makefile run target.

    Returns a list of argument lists (one per run command).
    """
    if not makefile_path or not os.path.isfile(makefile_path):
        return None

    try:
        with open(makefile_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return None

    recipe_lines = _collect_run_recipe_lines(lines)
    if not recipe_lines:
        return None

    args_list = []
    shell_ops = {"&&", ";", "|", "||", "&"}
    i = 0
    while i < len(recipe_lines):
        merged = recipe_lines[i].rstrip()
        while merged.endswith("\\") and i + 1 < len(recipe_lines):
            next_line = recipe_lines[i + 1].rstrip()
            merged = merged[:-1].rstrip() + " " + next_line.lstrip()
            i += 1
        i += 1

        merged = _strip_make_comment(merged)
        merged = _strip_recipe_prefix(merged)

        if not merged:
            continue

        try:
            tokens = shlex.split(merged)
        except ValueError:
            continue

        if not tokens:
            continue

        tokens = [tok for tok in tokens if tok != "\\"]

        run_cwd = None
        if len(tokens) >= 2 and tokens[0] == "cd":
            cd_path = tokens[1]
            run_cwd = os.path.normpath(os.path.join(src_dir or "", cd_path)) if not os.path.isabs(cd_path) else cd_path
            tokens = tokens[2:]
            if tokens and tokens[0] in shell_ops:
                tokens = tokens[1:]

        exec_index = _find_exec_index(tokens, exe_name=exe_name)
        if exec_index is None:
            continue

        args = []
        for token in tokens[exec_index + 1:]:
            if token in shell_ops:
                break
            if token == "\\":
                continue
            args.append(token)

        if src_dir:
            args = [_resolve_arg_path(arg, src_dir, run_cwd=run_cwd) for arg in args]

        args_list.append(args)

    return args_list


def get_makefile_run_args(src_dir, exe_name=None):
    """
    Convenience wrapper to find Makefile and extract run args.
    """
    makefile_path = find_makefile_for_target(src_dir)
    if makefile_path:
        args = extract_run_args_from_makefile(makefile_path, exe_name=exe_name, src_dir=src_dir)
        if args:
            return args

    fallback_path = find_run_target_file(src_dir)
    if fallback_path:
        return extract_run_args_from_makefile(fallback_path, exe_name=exe_name, src_dir=src_dir)

    return []


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