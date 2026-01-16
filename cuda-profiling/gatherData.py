#!/usr/bin/env python3
'''
gatherData.py - Profiling script for HeCBench CUDA and OpenMP benchmarks

This script profiles built executables using NVIDIA Nsight Compute (ncu) to
gather roofline performance data for each kernel.

The script:
1. Parses benchmarks.yaml to get benchmark metadata and execution arguments
2. Extracts kernel names from binaries using cuobjdump/objdump
3. Demangles kernel names reliably for use with `ncu -k`
4. Executes benchmarks with ncu for roofline profiling (first invocation of each kernel)
5. Extracts performance metrics from ncu-rep files
6. Downloads/unzips required input files
7. Writes results to cuda-profiling/gpuData.csv

Usage:
    python3 cuda-profiling/gatherData.py [OPTIONS]

Options:
    --buildDir PATH      Path to build directory (default: ../build)
    --srcDir PATH        Path to HeCBench src directory (default: ../HeCBench/src)
    --outfile PATH       Output CSV file (default: ./gpuData.csv)
    --skipRuns           Skip execution, only parse existing ncu-rep files
    --help               Show this message
'''

import signal
import os
import sys
import argparse
import pandas as pd
import glob
import yaml
from pprint import pprint
import re
from tqdm import tqdm
import subprocess
import shlex
from io import StringIO
import numpy as np
import csv
from pathlib import Path

# Global directory paths
DOWNLOAD_DIR = ''
THIS_DIR = ''
SRC_DIR = ''
BUILD_DIR = ''
HECBENCH_ROOT = ''

def setup_dirs(buildDir, srcDir):
    """Initialize global directory paths"""
    global DOWNLOAD_DIR, THIS_DIR, SRC_DIR, BUILD_DIR, HECBENCH_ROOT
    
    THIS_DIR = os.path.abspath(os.path.dirname(__file__))
    assert os.path.exists(THIS_DIR), f"Current directory not found: {THIS_DIR}"
    
    DOWNLOAD_DIR = os.path.abspath(f'{THIS_DIR}/downloads')
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, srcDir))
    BUILD_DIR = os.path.abspath(os.path.join(THIS_DIR, buildDir))
    HECBENCH_ROOT = os.path.dirname(SRC_DIR)
    
    print('Using the following directories:', flush=True)
    print(f'THIS_DIR        = [{THIS_DIR}]', flush=True)
    print(f'DOWNLOAD_DIR    = [{DOWNLOAD_DIR}]', flush=True)
    print(f'SRC_DIR         = [{SRC_DIR}]', flush=True)
    print(f'BUILD_DIR       = [{BUILD_DIR}]', flush=True)
    print(f'HECBENCH_ROOT   = [{HECBENCH_ROOT}]', flush=True)
    
    assert os.path.exists(SRC_DIR), f"Source directory not found: {SRC_DIR}"
    assert os.path.exists(BUILD_DIR), f"Build directory not found: {BUILD_DIR}"
    
    return

def load_benchmarks_yaml():
    """Load and parse benchmarks.yaml from HeCBench"""
    yaml_path = os.path.join(HECBENCH_ROOT, 'benchmarks.yaml')
    assert os.path.exists(yaml_path), f"benchmarks.yaml not found at {yaml_path}"
    
    with open(yaml_path, 'r') as f:
        benchmarks = yaml.safe_load(f)
    
    print(f"Loaded {len(benchmarks)} benchmarks from benchmarks.yaml")
    return benchmarks

def get_runnable_targets():
    """Gather list of executable files from build directory"""
    files = glob.glob(f'{BUILD_DIR}/*')
    execs = []
    
    for entry in files:
        # Check if file is executable
        if os.path.isfile(entry) and os.access(entry, os.X_OK):
            basename = os.path.basename(entry)
            
            # Skip non-executable files
            if any(ext in basename for ext in ['.cpp', '.c', '.o', '.so', '.log']):
                continue
            
            # Determine source directory
            execSrcDir = os.path.abspath(f'{SRC_DIR}/{basename}')
            
            # Check source directory exists
            if not os.path.isdir(execSrcDir):
                print(f"WARNING: No source dir for {basename} at {execSrcDir}")
                continue
            
            execDict = {
                'basename': basename,
                'exe': entry,
                'src': execSrcDir
            }
            execs.append(execDict)
    
    print(f"Found {len(execs)} executable targets")
    return execs

def get_exe_args_from_yaml(targets, benchmarks_data):
    """Extract execution arguments from benchmarks.yaml"""
    for target in targets:
        basename = target['basename']
        # Remove model suffix (-cuda, -omp, etc) to get benchmark name
        benchmark_name = re.sub(r'-(cuda|omp|hip|sycl)$', '', basename)
        
        target['exeArgs'] = ''
        
        if benchmark_name in benchmarks_data:
            bench_info = benchmarks_data[benchmark_name]
            if 'test' in bench_info and bench_info['test'] and 'args' in bench_info['test']:
                args = bench_info['test']['args']
                if args:
                    target['exeArgs'] = ' '.join(str(a) for a in args)
        
        if not target['exeArgs']:
            print(f"WARNING: No execution args found for {basename}")
    
    return targets

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

def get_cuobjdump_kernels(target):
    """Extract CUDA kernel names from executable using cuobjdump"""
    basename = target['basename']
    srcDir = target['src']
    exe_path = target['exe']
    
    # Try cuobjdump first
    try:
        result = subprocess.run(
            f'cuobjdump --list-text {exe_path}',
            shell=True,
            cwd=srcDir,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        if result.returncode == 0:
            output = result.stdout.decode('UTF-8')
            
            # Extract kernel names with regex
            # Pattern matches kernel names in cuobjdump output
            pattern = r'((?<= : x-)|(?<= : x-void ))[\w\-\:]*(?=[\(\<].*[\)\>](?:(?:(?:(?:\.sm_)|(?: \(\.sm_)).*\.elf\.bin)|(?: \[clone)))'
            matches = re.finditer(pattern, output, re.MULTILINE)
            
            raw_names = [m.group() for m in matches if m.group()]
            
            # Demangle each name
            demangled = []
            for name in raw_names:
                demangled_name = demangle_kernel_name(name)
                demangled.append(demangled_name)
            
            return demangled
    except Exception as e:
        print(f"cuobjdump failed for {basename}: {e}")
    
    # Try llvm-objdump as fallback
    try:
        result = subprocess.run(
            f'llvm-objdump -t {exe_path} | c++filt',
            shell=True,
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
        print(f"llvm-objdump failed for {basename}: {e}")
    
    return []

def get_objdump_kernels(target):
    """Extract OpenMP kernel names from executable using objdump"""
    basename = target['basename']
    srcDir = target['src']
    exe_path = target['exe']
    
    try:
        result = subprocess.run(
            f'objdump -t --section=omp_offloading_entries {exe_path}',
            shell=True,
            cwd=srcDir,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        if result.returncode == 0:
            output = result.stdout.decode('UTF-8')
            
            # Extract offloading entry names
            pattern = r'(?<=\.omp_offloading\.entry\.)(__omp_offloading.*)(?=\n)'
            matches = re.finditer(pattern, output, re.MULTILINE)
            
            kernel_names = [m.group() for m in matches if m.group()]
            
            # OpenMP targets should have at least one offload region
            if not kernel_names:
                print(f"WARNING: No OMP offload entries found in {basename}")
            
            return kernel_names
    except Exception as e:
        print(f"objdump failed for {basename}: {e}")
    
    return []

def extract_kernel_name_for_ncu(full_kernel_name):
    """
    Extract the simple kernel name for use with ncu -k option.
    
    Handles templates and namespaces by extracting the core function name.
    """
    # Remove return type if present
    if ' ' in full_kernel_name:
        parts = full_kernel_name.split()
        full_kernel_name = parts[-1]
    
    # Handle templates - remove angle brackets and contents
    if '<' in full_kernel_name or '>' in full_kernel_name:
        parts = re.split(r'<|>', full_kernel_name)
        full_kernel_name = parts[0]
    
    # Handle namespaces - take last component
    if '::' in full_kernel_name:
        full_kernel_name = full_kernel_name.split('::')[-1]
    
    # Filter out library kernels (like cub::)
    if full_kernel_name.startswith('cub::') or full_kernel_name.startswith('thrust::'):
        return None
    
    return full_kernel_name

def get_kernel_names(targets):
    """Extract kernel names from all targets"""
    for target in tqdm(targets, desc='Extracting kernel names'):
        basename = target['basename']
        
        kernel_names = []
        
        if '-cuda' in basename:
            raw_names = get_cuobjdump_kernels(target)
            
            # Process each kernel name
            for name in raw_names:
                simple_name = extract_kernel_name_for_ncu(name)
                if simple_name:
                    kernel_names.append(simple_name)
        
        elif '-omp' in basename:
            kernel_names = get_objdump_kernels(target)
        
        # Remove duplicates while preserving order
        kernel_names = list(dict.fromkeys(kernel_names))
        
        target['kernelNames'] = kernel_names
        
        if not kernel_names:
            print(f"WARNING: No kernels found for {basename}")
    
    return targets

def execute_target_with_ncu(target, kernelName):
    """
    Execute target with ncu profiling for specified kernel.
    
    Captures roofline metrics for the first invocation of the kernel.
    """
    basename = target['basename']
    exeArgs = target['exeArgs']
    srcDir = target['src']
    exe_path = target['exe']
    
    reportFileName = f'{basename}-[{kernelName}]-report'
    
    # NCU command for roofline profiling
    # -c 2: Capture first 2 invocations (use first, second as confirmation)
    # --set roofline: Enable roofline metrics
    # --metrics smsp__sass_thread_inst_executed_op_integer_pred_on: Add integer ops
    ncuCommand = (
        f'ncu -f -o {reportFileName} '
        f'--set roofline '
        f'--metrics smsp__sass_thread_inst_executed_op_integer_pred_on '
        f'-c 2 -k "{kernelName}"'
    )
    
    exeCommand = f'{ncuCommand} {exe_path} {exeArgs}'.strip()
    
    print(f'Profiling: {basename} - kernel: {kernelName}', flush=True)
    print(f'  Command: {exeCommand}', flush=True)
    
    try:
        # Execute with timeout (5 minutes)
        process = subprocess.Popen(
            shlex.split(exeCommand),
            cwd=srcDir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        
        stdout, _ = process.communicate(timeout=300)
        returncode = process.returncode
        
    except subprocess.TimeoutExpired:
        # Kill process group on timeout
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        stdout, _ = process.communicate()
        print(f'  TIMEOUT for {basename}-[{kernelName}]')
        return (None, None)
    except Exception as e:
        print(f'  ERROR executing {basename}-[{kernelName}]: {e}')
        return (None, None)
    
    if returncode != 0:
        print(f'  Execution failed with code {returncode}')
        return (None, None)
    
    stdout_str = stdout.decode('UTF-8')
    
    if '==WARNING== No kernels were profiled.' in stdout_str:
        print(f'  No kernels profiled (kernel not invoked)')
        return (None, None)
    
    # Read ncu-rep file
    rep_file = f'{srcDir}/{reportFileName}.ncu-rep'
    if not os.path.exists(rep_file):
        print(f'  No .ncu-rep file generated')
        return (None, None)
    
    # Parse ncu-rep to CSV
    try:
        result = subprocess.run(
            f'ncu --import {reportFileName}.ncu-rep --csv --print-units base --page raw',
            shell=True,
            cwd=srcDir,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        if result.returncode != 0:
            print(f'  Failed to parse ncu-rep file')
            return (None, None)
        
        return (stdout_str, result)
    
    except Exception as e:
        print(f'  Error parsing ncu-rep: {e}')
        return (None, None)

def roofline_results_to_df(ncuOutput):
    """Convert NCU CSV output to pandas DataFrame"""
    stringified = StringIO(ncuOutput.stdout.decode('UTF-8'))
    df = pd.read_csv(stringified, quotechar='"')
    return df

def str_to_float(x):
    """Convert string with commas to float"""
    if pd.isna(x) or x == '':
        return np.nan
    return np.float64(str(x).replace(',', ''))

def calc_roofline_data(df):
    """
    Calculate roofline metrics from raw NCU data.
    
    Formulas:
    - DP Performance: (DP_ADD + DP_MUL + DP_FMA*2) * cycles_per_sec
    - SP Performance: (SP_ADD + SP_MUL + SP_FMA*2) * cycles_per_sec
    - INT Performance: int_ops / xtime
    - Traffic: DRAM bytes/sec
    - Arithmetic Intensity: Performance / Traffic
    """
    # Skip header row (units)
    kdf = df.iloc[1:].copy(deep=True)
    
    if kdf.shape[0] == 0:
        return kdf
    
    # Cycles per second
    avgCyclesPerSecond = kdf['smsp__cycles_elapsed.avg.per_second'].apply(str_to_float)
    
    # Double precision ops
    sumDPAddOps = kdf['smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumDPMulOps = kdf['smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumDPfmaOps = kdf['derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2'].apply(str_to_float)
    kdf['dpPerf'] = (sumDPAddOps + sumDPMulOps + sumDPfmaOps) * avgCyclesPerSecond
    
    # Single precision ops
    sumSPAddOps = kdf['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumSPMulOps = kdf['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumSPfmaOps = kdf['derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2'].apply(str_to_float)
    kdf['spPerf'] = (sumSPAddOps + sumSPMulOps + sumSPfmaOps) * avgCyclesPerSecond
    
    # DRAM traffic
    kdf['traffic'] = kdf['dram__bytes.sum.per_second'].apply(str_to_float)
    
    # Arithmetic intensity
    kdf['dpAI'] = kdf['dpPerf'] / kdf['traffic']
    kdf['spAI'] = kdf['spPerf'] / kdf['traffic']
    
    # Execution time
    kdf['xtime'] = kdf['gpu__time_duration.sum'].apply(str_to_float)
    kdf['device'] = kdf['device__attribute_display_name']
    
    # Integer ops
    kdf['intops'] = kdf['smsp__sass_thread_inst_executed_op_integer_pred_on.sum'].apply(str_to_float)
    kdf['intPerf'] = kdf['intops'] / (1e-9 * kdf['xtime'])  # xtime is in nanoseconds
    kdf['intAI'] = kdf['intPerf'] / kdf['traffic']
    
    return kdf

def has_already_been_sampled(basename, kernelName, df):
    """Check if kernel has already been profiled"""
    if df.shape[0] == 0:
        return False
    
    subset = df[(df['targetName'] == basename) & (df['kernelName'] == kernelName)]
    return subset.shape[0] > 0

def execute_targets(targets, csvFilename, skipRuns=False):
    """Execute all targets and gather profiling data"""
    
    # Load existing data if available
    if os.path.isfile(csvFilename):
        df = pd.read_csv(csvFilename)
        print(f"Loaded existing data: {df.shape[0]} rows")
    else:
        df = pd.DataFrame()
    
    for target in tqdm(targets, desc='Profiling benchmarks'):
        basename = target['basename']
        kernelNames = target['kernelNames']
        exeArgs = target['exeArgs']
        
        if not kernelNames:
            print(f"Skipping {basename} - no kernels found")
            continue
        
        for kernelName in kernelNames:
            if has_already_been_sampled(basename, kernelName, df):
                print(f"Skipping {basename}:[{kernelName}] - already sampled")
                continue
            
            if skipRuns:
                print(f"Skipping {basename}:[{kernelName}] - skipRuns enabled")
                continue
            
            # Execute with NCU
            stdout, ncuResult = execute_target_with_ncu(target, kernelName)
            
            if ncuResult is None:
                continue
            
            # Parse results
            try:
                rawDF = roofline_results_to_df(ncuResult)
                roofDF = calc_roofline_data(rawDF)
                
                # Extract relevant columns
                subset = roofDF[[
                    'Kernel Name', 'traffic', 'dpAI', 'spAI', 'dpPerf', 'spPerf',
                    'xtime', 'Block Size', 'Grid Size', 'device',
                    'intops', 'intPerf', 'intAI'
                ]].copy()
                
                subset['targetName'] = basename
                subset['exeArgs'] = exeArgs
                subset['kernelName'] = kernelName
                
                # Append to dataframe
                df = pd.concat([df, subset], ignore_index=True)
                
                # Save incrementally
                df.to_csv(csvFilename, quoting=csv.QUOTE_NONNUMERIC, quotechar='"', 
                         index=False, na_rep='NULL')
                
                print(f"  Saved data for {basename}:[{kernelName}]")
                
            except Exception as e:
                print(f"  Error processing results: {e}")
                continue
    
    print(f"Profiling complete! Data saved to {csvFilename}")
    print(f"Total samples: {df.shape[0]}")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Profile HeCBench benchmarks with NVIDIA Nsight Compute'
    )
    
    parser.add_argument('--buildDir', type=str, default='../build',
                       help='Directory containing built executables')
    parser.add_argument('--srcDir', type=str, default='../HeCBench/src',
                       help='Directory containing source files')
    parser.add_argument('--outfile', type=str, default='./gpuData.csv',
                       help='Output CSV file for profiling data')
    parser.add_argument('--skipRuns', action='store_true',
                       help='Skip execution, only parse existing ncu-rep files')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_dirs(args.buildDir, args.srcDir)
    
    # Load benchmarks metadata
    benchmarks = load_benchmarks_yaml()
    
    # Get executable targets
    targets = get_runnable_targets()
    
    if not targets:
        print("ERROR: No executable targets found!")
        sys.exit(1)
    
    # Get execution arguments from YAML
    targets = get_exe_args_from_yaml(targets, benchmarks)
    
    # Extract kernel names
    targets = get_kernel_names(targets)
    
    # Execute and profile
    results = execute_targets(targets, args.outfile, args.skipRuns)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
