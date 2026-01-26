#!/usr/bin/env python3
'''
gatherData.py - Profiling script for HeCBench CUDA and OpenMP benchmarks

This script profiles built CUDA/OpenMP executables using NVIDIA Nsight Compute (ncu)
to gather roofline performance data for each kernel.

The script:
1. Parses benchmarks.yaml to get benchmark metadata and execution arguments
2. Extracts kernel names from binaries using cuobjdump/objdump
3. Executes benchmarks with ncu for roofline profiling (first invocation per kernel)
4. Repeats profiling per target for a configurable number of samples
5. Extracts performance metrics from ncu-rep files and writes to CSV (with sample index)
6. Logs stdout/stderr per run to a timestamped JSON log file
7. Zips the CSV, JSON log, and NCU reports into a timestamped archive

Usage:
    python3 cuda-profiling/gatherData.py [OPTIONS]

Options:
    --buildDir PATH      Path to build directory (default: ../build)
    --srcDir PATH        Path to HeCBench src directory (default: ../HeCBench/src)
    --outfile PATH       Output CSV file (default: ./gpuData.csv)
    --skipRuns           Skip execution, only parse existing ncu-rep files
    --samples N          Number of samples per target (default: 3)
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
import time
import json
from datetime import datetime
import zipfile
from dataclasses import dataclass
from typing import Optional, List

from utils import (
    demangle_kernel_name,
    demangle_omp_offload_name,
    is_library_kernel,
    get_cuobjdump_kernels,
    parse_omp_offload_entries,
    get_objdump_kernels,
    source_has_cuda_kernels,
    exe_has_cuda_kernels,
    get_makefile_run_args,
    get_gpu_info,
)

# Global directory paths
THIS_DIR = ''
SRC_DIR = ''
BUILD_DIR = ''
HECBENCH_ROOT = ''
GPU_INFO = None
GPU_PREFIX = ''


@dataclass
class NcuRunResult:
    stdout_capture: Optional[str]
    ncu_result: Optional[subprocess.CompletedProcess]
    kernel_executed: Optional[bool]
    elapsed_sec: float
    timed_out: bool
    stdout: Optional[str]
    stderr: Optional[str]
    returncode: Optional[int]
    cmd: List[str]
    cwd: str
    parse_error: Optional[str]
    status: str


def _sanitize_gpu_name(name):
    if not name:
        return ''
    safe = re.sub(r"\s+", "_", name.strip())
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", safe)
    return safe


def _apply_gpu_prefix_to_path(path, prefix):
    if not path or not prefix:
        return path
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    if basename.startswith(f"{prefix}_"):
        return path
    return os.path.join(directory, f"{prefix}_{basename}")

def setup_dirs(buildDir, srcDir):
    """Initialize global directory paths"""
    global THIS_DIR, SRC_DIR, BUILD_DIR, HECBENCH_ROOT
    
    THIS_DIR = os.path.abspath(os.path.dirname(__file__))
    assert os.path.exists(THIS_DIR), f"Current directory not found: {THIS_DIR}"
    
    SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, srcDir))
    BUILD_DIR = os.path.abspath(os.path.join(THIS_DIR, buildDir))
    HECBENCH_ROOT = os.path.dirname(SRC_DIR)
    
    print('Using the following directories:', flush=True)
    print(f'THIS_DIR        = [{THIS_DIR}]', flush=True)
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
    execs = []

    bin_dir = os.path.join(BUILD_DIR, 'bin')
    if os.path.isdir(bin_dir):
        for model in ['cuda', 'omp']:
            model_dir = os.path.join(bin_dir, model)
            if not os.path.isdir(model_dir):
                continue

            for entry in glob.glob(f'{model_dir}/*'):
                if os.path.isfile(entry) and os.access(entry, os.X_OK):
                    targetName = os.path.basename(entry)

                    if any(ext in targetName for ext in ['.cpp', '.c', '.o', '.so', '.log']):
                        continue

                    execSrcDir = os.path.abspath(f'{SRC_DIR}/{targetName}-{model}')
                    if not os.path.isdir(execSrcDir):
                        fallback_src = os.path.abspath(f'{SRC_DIR}/{targetName}')
                        if os.path.isdir(fallback_src):
                            execSrcDir = fallback_src
                        else:
                            print(f"WARNING: No source dir for {targetName} at {execSrcDir}")
                            continue

                    execDict = {
                        'targetName': targetName,
                        'exe': entry,
                        'src': execSrcDir,
                        'model': model
                    }
                    execs.append(execDict)
    else:
        files = glob.glob(f'{BUILD_DIR}/*')
        for entry in files:
            # Check if file is executable
            if os.path.isfile(entry) and os.access(entry, os.X_OK):
                targetName = os.path.basename(entry)

                # Skip non-executable files
                if any(ext in targetName for ext in ['.cpp', '.c', '.o', '.so', '.log']):
                    continue

                # Determine source directory
                execSrcDir = os.path.abspath(f'{SRC_DIR}/{targetName}')

                # Check source directory exists
                if not os.path.isdir(execSrcDir):
                    print(f"WARNING: No source dir for {targetName} at {execSrcDir}")
                    continue

                execDict = {
                    'targetName': targetName,
                    'exe': entry,
                    'src': execSrcDir
                }
                execs.append(execDict)
    
    print(f"Found {len(execs)} executable targets")
    return execs

def get_exe_args_from_yaml(targets, benchmarks_data):
    """Extract execution arguments from benchmarks.yaml"""
    for target in targets:
        targetName = target['targetName']
        # Remove model suffix (-cuda, -omp, etc) to get benchmark name
        benchmark_name = re.sub(r'-(cuda|omp|hip|sycl)$', '', targetName)
        
        target['exeArgs'] = ''
        
        yaml_args = None
        have_args = False

        if benchmark_name in benchmarks_data:
            bench_info = benchmarks_data[benchmark_name]
            if 'test' in bench_info and bench_info['test'] and 'args' in bench_info['test']:
                yaml_args = bench_info['test']['args']
                if yaml_args is not None:
                    have_args = True
                if yaml_args:
                    target['exeArgs'] = ' '.join(str(a) for a in yaml_args)

        if yaml_args is None:
            make_args = get_makefile_run_args(target.get('src'), exe_name=None)
            if make_args:
                have_args = True
                last_args = make_args[-1]
                target['exeArgs'] = ' '.join(str(a) for a in last_args) if last_args else ''

        if not target['exeArgs'] and not have_args:
            print(f"WARNING: No execution args found for {targetName}")
    
    return targets


def get_kernel_names(targets):
    """Extract kernel names from all targets"""
    for target in tqdm(targets, desc='Extracting kernel names'):
        targetName = target['targetName']
        model = target.get('model')
        
        kernels = []

        if model == 'cuda' or '-cuda' in targetName:
            raw_names = get_cuobjdump_kernels(target)
            
            # Process each kernel name
            for name in raw_names:
                demangled = demangle_kernel_name(name)

                if is_library_kernel(demangled):
                    continue

                profiler_name = demangled

                if profiler_name:
                    kernels.append({
                        'mangled': name,
                        'demangled': demangled,
                        'profiler': profiler_name
                    })

        elif model == 'omp' or '-omp' in targetName:
            raw_names = get_objdump_kernels(target)
            for name in raw_names:
                demangled = demangle_omp_offload_name(name)
                kernels.append({
                    'mangled': name,
                    'demangled': demangled,
                    'profiler': name
                })
        
        # Remove duplicates while preserving order
        unique = []
        seen = set()
        for kernel in kernels:
            key = (kernel.get('mangled'), kernel.get('profiler'))
            if key in seen:
                continue
            seen.add(key)
            unique.append(kernel)
        kernels = unique
        
        target['kernels'] = kernels

        if not kernels:
            if model == 'cuda' or '-cuda' in targetName:
                if not source_has_cuda_kernels(target.get('src')):
                    print(f"INFO: No profiliable CUDA kernels found in sources for {targetName}; skipping.")
                elif not exe_has_cuda_kernels(target):
                    print(f"INFO: No profiliable CUDA kernels found in executable for {targetName}; skipping.")
                else:
                    print(f"WARNING: No CUDA kernels found for {targetName} (kernels expected from source)")
            else:
                print(f"WARNING: No kernels found for {targetName}")
    
    return targets


def summarize_profiliable_kernels(targets):
    """
    Summarize profiliable codes/kernels and list targets without kernels.
    """
    cuda_missing = []
    omp_missing = []
    cuda_profiliable = 0
    omp_profiliable = 0
    cuda_kernels = 0
    omp_kernels = 0

    for target in targets:
        target_name = target.get('targetName')
        model = target.get('model')
        kernels = target.get('kernels') or []

        if model == 'cuda' or '-cuda' in (target_name or ''):
            if kernels:
                cuda_profiliable += 1
                cuda_kernels += len(kernels)
            else:
                cuda_missing.append(target_name)
        elif model == 'omp' or '-omp' in (target_name or ''):
            if kernels:
                omp_profiliable += 1
                omp_kernels += len(kernels)
            else:
                omp_missing.append(target_name)

    summary = {
        'cuda_profiliable': cuda_profiliable,
        'omp_profiliable': omp_profiliable,
        'cuda_kernels': cuda_kernels,
        'omp_kernels': omp_kernels,
        'cuda_missing': cuda_missing,
        'omp_missing': omp_missing,
    }

    return summary


def summarize_existing_sampling_progress(targets, outfile, summary):
    """Summarize already-sampled CUDA/OpenMP codes and kernels from existing CSV."""
    sampled = {
        'cuda_codes': 0,
        'omp_codes': 0,
        'cuda_kernels': 0,
        'omp_kernels': 0,
    }

    if not os.path.isfile(outfile):
        return sampled

    try:
        existing_df = pd.read_csv(outfile)
        if existing_df.empty or 'targetName' not in existing_df.columns or 'kernelName' not in existing_df.columns:
            return sampled

        target_model = {}
        valid_pairs = set()
        for t in targets:
            tname = t.get('targetName')
            model = t.get('model')
            if not model:
                if '-cuda' in (tname or ''):
                    model = 'cuda'
                elif '-omp' in (tname or ''):
                    model = 'omp'
            target_model[tname] = model
            for k in t.get('kernels') or []:
                valid_pairs.add((tname, k.get('profiler')))

        sampled_df = existing_df[['targetName', 'kernelName']]
        if 'kernel_executed' in existing_df.columns:
            sampled_df = existing_df[existing_df['kernel_executed'] == 'normal'][['targetName', 'kernelName']]

        sampled_pairs = set(
            (row['targetName'], row['kernelName'])
            for _, row in sampled_df.dropna().iterrows()
        )
        sampled_pairs = sampled_pairs.intersection(valid_pairs)

        sampled_codes_by_model = {'cuda': set(), 'omp': set()}
        sampled_kernels_by_model = {'cuda': 0, 'omp': 0}

        for tname, _ in sampled_pairs:
            model = target_model.get(tname)
            if model in sampled_codes_by_model:
                sampled_codes_by_model[model].add(tname)
                sampled_kernels_by_model[model] += 1

        sampled['cuda_codes'] = len(sampled_codes_by_model['cuda'])
        sampled['omp_codes'] = len(sampled_codes_by_model['omp'])
        sampled['cuda_kernels'] = sampled_kernels_by_model['cuda']
        sampled['omp_kernels'] = sampled_kernels_by_model['omp']
        return sampled
    except Exception as e:
        print(f"WARNING: Failed to read existing data from {outfile}: {e}")
        return sampled


def _get_report_basename(target, sample_idx):
    targetName = target['targetName']
    model = target.get('model')
    if not model:
        if '-cuda' in (targetName or ''):
            model = 'cuda'
        elif '-omp' in (targetName or ''):
            model = 'omp'
    model_tag = 'cuda' if model == 'cuda' else 'omp' if model == 'omp' else 'UNKNOWN'
    results_dir = os.path.join(THIS_DIR, 'ncu-rep-results')
    os.makedirs(results_dir, exist_ok=True)
    report_base = f'{targetName}-{model_tag}-s{sample_idx}-report'
    if GPU_PREFIX:
        report_base = f'{GPU_PREFIX}_{report_base}'
    return os.path.join(results_dir, report_base)


def _build_ncu_command(target, report_basename):
    exeArgs = target['exeArgs']
    exe_path = target['exe']
    ncu_args = [
        'ncu', '-f', '-o', report_basename,
        '--set', 'roofline',
        '--metrics', 'smsp__sass_thread_inst_executed_op_integer_pred_on,dram__bytes_read.sum,dram__bytes_write.sum',
        '--kernel-name-base', 'demangled',
        '--kernel-id', ':::1',
        exe_path
    ]
    if exeArgs:
        ncu_args.extend(shlex.split(exeArgs))
    return ncu_args


def _run_ncu_process(ncu_args, srcDir, timeout_sec):
    start_time = time.monotonic()
    process = subprocess.Popen(
        ncu_args,
        cwd=srcDir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
        returncode = process.returncode
        elapsed = time.monotonic() - start_time
        return stdout, stderr, returncode, elapsed, False
    except subprocess.TimeoutExpired:
        print('  TIMEOUT; sending SIGINT to allow report flush')
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, stderr = process.communicate()
        except Exception:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, stderr = process.communicate()
        returncode = process.returncode
        elapsed = time.monotonic() - start_time
        return stdout, stderr, returncode, elapsed, True


def _try_parse_ncu_report(report_basename, srcDir, stdout_str):
    if stdout_str is None:
        stdout_str = ''
    rep_file = f'{report_basename}.ncu-rep'
    if not os.path.exists(rep_file):
        print('  No .ncu-rep file generated')
        return None, 'missing .ncu-rep report'
    if os.path.getsize(rep_file) == 0:
        print('  .ncu-rep file is empty')
        return None, 'empty .ncu-rep report'

    try:
        result = subprocess.run(
            [
                'ncu', '--import', rep_file,
                '--csv', '--print-units', 'base', '--page', 'raw',
                '--print-kernel-base', 'mangled'
            ],
            cwd=srcDir,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        if result.returncode != 0:
            print('  Failed to parse ncu-rep file')
            return None, f'ncu --import failed (code {result.returncode})'

        return (stdout_str, result, True), None
    except Exception as e:
        print(f'  Error parsing ncu-rep: {e}')
        return None, f'exception parsing ncu-rep: {e}'


def _build_run_result(ncu_args, srcDir, stdout_capture, ncu_result, kernel_executed,
                      elapsed, timed_out, stdout_str, stderr_str, returncode,
                      parse_error, status):
    return NcuRunResult(
        stdout_capture=stdout_capture,
        ncu_result=ncu_result,
        kernel_executed=kernel_executed,
        elapsed_sec=elapsed,
        timed_out=timed_out,
        stdout=stdout_str,
        stderr=stderr_str,
        returncode=returncode,
        cmd=ncu_args,
        cwd=srcDir,
        parse_error=parse_error,
        status=status,
    )


def _classify_status(returncode, timed_out, kernel_executed, parse_error, combined_str):
    if timed_out:
        return 'timeout'
    if kernel_executed is False:
        return 'no_kernels'
    if returncode not in (None, 0):
        return 'error'
    if parse_error:
        return 'parse_error'
    if combined_str and '==WARNING== No kernels were profiled.' in combined_str:
        return 'no_kernels'
    return 'normal'



def execute_target_with_ncu(target, sample_idx, timeout_sec=120):
    """
    Execute target with ncu profiling for all kernels (first invocation).

    Captures roofline metrics for the first invocation of each kernel.
    """
    targetName = target['targetName']
    srcDir = target['src']
    report_basename = _get_report_basename(target, sample_idx)
    ncu_args = _build_ncu_command(target, report_basename)

    print(f'\nProfiling: {targetName} sample {sample_idx} (all kernels, first invocations)', flush=True)
    print(f'  Command: {" ".join(ncu_args)}', flush=True)

    try:
        stdout, stderr, returncode, elapsed, timed_out = _run_ncu_process(
            ncu_args,
            srcDir,
            timeout_sec
        )
    except Exception as e:
        print(f'  ERROR executing {targetName}: {e}')
        parsed, parse_error = _try_parse_ncu_report(report_basename, srcDir, None)
        if parsed is not None:
            print('  Parsed existing report after execution error')
            stdout_capture, ncu_result, kernel_executed = parsed
            status = _classify_status(None, False, kernel_executed, parse_error, None)
            return _build_run_result(ncu_args, srcDir, stdout_capture, ncu_result,
                                     kernel_executed, 0.0, False, None, None, None,
                                     parse_error, status)
        return _build_run_result(ncu_args, srcDir, None, None, None, 0.0, False,
                                 None, None, None, parse_error or str(e), 'error')

    stdout_str = stdout.decode('UTF-8') if stdout else ''
    stderr_str = stderr.decode('UTF-8') if stderr else ''
    combined_str = stdout_str + stderr_str

    parsed, parse_error = _try_parse_ncu_report(report_basename, srcDir, combined_str)
    if parsed is not None:
        stdout_capture, ncu_result, kernel_executed = parsed
    else:
        stdout_capture, ncu_result, kernel_executed = None, None, None

    if '==WARNING== No kernels were profiled.' in combined_str:
        print(f'  No kernels profiled for {targetName}')
        kernel_executed = False

    status = _classify_status(returncode, timed_out, kernel_executed, parse_error, combined_str)

    if timed_out and parsed is not None:
        print('  Parsed existing report after timeout')
    if returncode not in (None, 0) and parsed is not None:
        print('  Parsed existing report despite nonzero exit code')

    return _build_run_result(ncu_args, srcDir, stdout_capture, ncu_result, kernel_executed,
                             elapsed, timed_out, stdout_str, stderr_str, returncode,
                             parse_error, status)

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

def str_to_int(x):
    """Convert string with commas to int"""
    if pd.isna(x) or x == '':
        return np.nan
    return np.int64(str(x).replace(',', ''))

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
    
    # Cycles per second (cycles/sec)
    avgCyclesPerSecond = kdf['smsp__cycles_elapsed.avg.per_second'].apply(str_to_float)
    
    # Double precision ops (operations per cycle)
    # Resulting dpPerf is in OP/s
    sumDPAddOps = kdf['smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumDPMulOps = kdf['smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumDPfmaOps = kdf['derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2'].apply(str_to_float)

    # this is in units of (ops/cycle + ops/cycle + ops/cycle) * (cycle/sec) = (ops/sec)
    kdf['dpPerf'] = (sumDPAddOps + sumDPMulOps + sumDPfmaOps) * avgCyclesPerSecond
    
    
    # Single precision ops (operations per cycle)
    # Resulting spPerf is in OP/s
    sumSPAddOps = kdf['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumSPMulOps = kdf['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumSPfmaOps = kdf['derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2'].apply(str_to_float)

    # this is in units of (ops/cycle + ops/cycle + ops/cycle) * (cycle/sec) = (ops/sec)
    kdf['spPerf'] = (sumSPAddOps + sumSPMulOps + sumSPfmaOps) * avgCyclesPerSecond


    sumHPAddInst = kdf['smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumHPMulInst = kdf['smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    # Already weighted to 4 ops per HFMA2 (2 lanes * 2 FLOPs/lane)
    sumHPFmaOps  = kdf['derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x4'].apply(str_to_float)

    # Convert HADD2/HMUL2 instruction rate -> scalar FLOP rate by *2
    kdf['hpPerf'] = ((2*sumHPAddInst) + (2*sumHPMulInst) + sumHPFmaOps) * avgCyclesPerSecond


    kdf['bytesRead'] = kdf['dram__bytes_read.sum'].apply(str_to_int)
    kdf['bytesWrite'] = kdf['dram__bytes_write.sum'].apply(str_to_int)
    kdf['bytesTotal'] = kdf['bytesRead'] + kdf['bytesWrite']
    
    # DRAM traffic (bytes/sec)
    kdf['traffic'] = kdf['dram__bytes.sum.per_second'].apply(str_to_float)
    
    # Arithmetic intensity (OP/byte)
    kdf['dpAI'] = kdf['dpPerf'] / kdf['traffic']
    kdf['spAI'] = kdf['spPerf'] / kdf['traffic']
    kdf['hpAI'] = kdf['hpPerf'] / kdf['traffic']
    
    # Execution time (ns)
    kdf['xtime'] = kdf['gpu__time_duration.sum'].apply(str_to_float)
    kdf['device'] = kdf['device__attribute_display_name']
    
    # Total floating-point operations (unitless count)
    # spPerf/dpPerf/hpPerf are in OP/s and xtime is in ns
    kdf['SP_FLOP'] = (kdf['spPerf'] * 1e-9 * kdf['xtime']).apply(int)
    kdf['DP_FLOP'] = (kdf['dpPerf'] * 1e-9 * kdf['xtime']).apply(int)
    kdf['HP_FLOP'] = (kdf['hpPerf'] * 1e-9 * kdf['xtime']).apply(int)
    
    # Integer ops (unitless count)
    kdf['INTOP'] = kdf['smsp__sass_thread_inst_executed_op_integer_pred_on.sum'].apply(str_to_float)
    # Integer performance (OP/s), xtime in ns
    kdf['intPerf'] = kdf['INTOP'] / (1e-9 * kdf['xtime'])  # xtime is in nanoseconds
    # Integer arithmetic intensity (OP/byte)
    kdf['intAI'] = kdf['intPerf'] / kdf['traffic']
    
    return kdf

def _expected_kernel_names(target):
    return [k.get('profiler') for k in (target.get('kernels') or []) if k.get('profiler')]


def target_sample_fully_sampled(target, df, sample_idx):
    """Check whether all kernels for a target are already sampled for a given sample."""
    if df is None or df.shape[0] == 0:
        return False

    targetName = target.get('targetName')
    source_name = os.path.basename(target.get('src') or '')
    expected = _expected_kernel_names(target)
    if not expected:
        return False

    if 'sample' not in df.columns:
        return False

    if 'source' in df.columns and source_name:
        subset = df[(df['source'] == source_name) & (df['sample'] == sample_idx)]
    else:
        subset = df[(df['targetName'] == targetName) & (df['sample'] == sample_idx)]

    if subset.empty or 'kernelName' not in subset.columns:
        return False

    if 'kernel_executed' in subset.columns:
        subset = subset[subset['kernel_executed'] == 'normal']

    sampled = set(subset['kernelName'].dropna().tolist())
    return set(expected).issubset(sampled)


def target_fully_sampled(target, df, samples):
    """Check whether all kernels for a target are already sampled for all samples."""
    for sample_idx in range(1, samples + 1):
        if not target_sample_fully_sampled(target, df, sample_idx):
            return False
    return True


def _reorder_output_columns(df):
    """Ensure key columns appear first for readability."""
    if df is None or df.empty:
        return df

    preferred = ['source', 'exePath', 'sample', 'kernel_executed', 'eteProfilerXtime']
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]


def _get_target_model(target):
    targetName = target.get('targetName')
    model = target.get('model')
    if not model:
        if '-cuda' in (targetName or ''):
            model = 'cuda'
        elif '-omp' in (targetName or ''):
            model = 'omp'
    return model


def _get_ncu_report_path(target, sample_idx):
    targetName = target.get('targetName')
    model = _get_target_model(target)
    model_tag = 'cuda' if model == 'cuda' else 'omp' if model == 'omp' else 'UNKNOWN'
    results_dir = os.path.join(THIS_DIR, 'ncu-rep-results')
    report_name = f'{targetName}-{model_tag}-s{sample_idx}-report.ncu-rep'
    if GPU_PREFIX:
        report_name = f'{GPU_PREFIX}_{report_name}'
    return os.path.join(results_dir, report_name)


def _parse_ncu_report(report_path, src_dir):
    if not report_path or not os.path.exists(report_path):
        print(f'  No .ncu-rep file found at {report_path}')
        return None
    if os.path.getsize(report_path) == 0:
        print(f'  .ncu-rep file is empty at {report_path}')
        return None

    try:
        result = subprocess.run(
            [
                'ncu', '--import', report_path,
                '--csv', '--print-units', 'base', '--page', 'raw',
                '--print-kernel-base', 'mangled'
            ],
            cwd=src_dir,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        if result.returncode != 0:
            print(f'  Failed to parse ncu-rep file: {report_path}')
            return None

        return result
    except Exception as e:
        print(f'  Error parsing ncu-rep file {report_path}: {e}')
        return None


def _append_missing_kernel_rows(df, expected_kernels, kernel_map, status, runtime,
                                ete_profiler_xtime, targetName, exeArgs, exe_path, source_name,
                                sample_idx):
    for kernel_mangled in expected_kernels:
        kernel = kernel_map.get(kernel_mangled, {})
        row = {
            'runtime': runtime,
            'eteProfilerXtime': ete_profiler_xtime,
            'sample': sample_idx,
            'CC': np.nan,
            'Kernel Name': np.nan,
            'traffic': np.nan,
            'bytesRead': np.nan,
            'bytesWrite': np.nan,
            'bytesTotal': np.nan,
            'dpAI': np.nan,
            'spAI': np.nan,
            'hpAI': np.nan,
            'dpPerf': np.nan,
            'spPerf': np.nan,
            'hpPerf': np.nan,
            'xtime': np.nan,
            'Block Size': np.nan,
            'Grid Size': np.nan,
            'device': np.nan,
            'SP_FLOP': np.nan,
            'DP_FLOP': np.nan,
            'HP_FLOP': np.nan,
            'INTOP': np.nan,
            'intPerf': np.nan,
            'intAI': np.nan,
            'targetName': targetName,
            'exeArgs': exeArgs,
            'kernelName': kernel.get('profiler'),
            'kernelMangled': kernel.get('mangled'),
            'kernelDemangled': kernel.get('demangled'),
            'kernelProfiler': kernel.get('profiler'),
            'kernel_executed': status,
            'exePath': exe_path,
            'source': source_name,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def _append_ncu_results(df, ncuResult, target, kernel_map, expected_kernels,
                        runtime, exeArgs, exe_path, source_name,
                        ete_profiler_xtime, timed_out, sample_idx):
    targetName = target.get('targetName')
    # Parse results
    try:
        rawDF = roofline_results_to_df(ncuResult)
        roofDF = calc_roofline_data(rawDF)

        if roofDF.empty:
            print(f"  No roofline data parsed for {targetName}")
            return df, 0

        filtered = roofDF[roofDF['Kernel Name'].isin(expected_kernels)].copy()
        if 'CC' not in filtered.columns:
            filtered['CC'] = np.nan
        for col in ['bytesRead', 'bytesWrite', 'bytesTotal']:
            if col not in filtered.columns:
                filtered[col] = np.nan

        found_kernels = set(filtered['Kernel Name'].dropna().tolist())
        missing_kernels = [k for k in expected_kernels if k not in found_kernels]
        sampled_count = len(found_kernels)

        # Extract relevant columns
        subset = filtered[[
            'CC', 'Kernel Name', 'traffic', 'bytesRead', 'bytesWrite', 'bytesTotal',
            'dpAI', 'spAI', 'hpAI', 'dpPerf', 'spPerf', 'hpPerf', 'xtime',
            'Block Size', 'Grid Size', 'device', 'SP_FLOP', 'DP_FLOP', 'HP_FLOP',
            'INTOP', 'intPerf', 'intAI'
        ]].copy()

        subset['targetName'] = targetName
        subset['exeArgs'] = exeArgs
        subset['runtime'] = runtime
        subset['kernelMangled'] = subset['Kernel Name']
        subset['kernelName'] = subset['Kernel Name'].map(
            lambda k: kernel_map.get(k, {}).get('profiler') or k
        )
        subset['sample'] = sample_idx
        subset['eteProfilerXtime'] = ete_profiler_xtime
        subset['kernelDemangled'] = subset['Kernel Name'].map(
            lambda k: kernel_map.get(k, {}).get('demangled')
        )
        subset['kernelProfiler'] = subset['Kernel Name'].map(
            lambda k: kernel_map.get(k, {}).get('profiler') or k
        )
        subset['kernel_executed'] = 'normal'
        subset['exePath'] = exe_path
        subset['source'] = source_name

        # Append sampled kernels
        df = pd.concat([df, subset], ignore_index=True)

        # Add NULL rows for missing kernels
        if missing_kernels:
            missing_labels = [
                kernel_map.get(k, {}).get('profiler') or k for k in missing_kernels
            ]
            print(f"  Missing kernels for {targetName}: {', '.join(missing_labels)}")
            status = 'timeout' if timed_out else 'not profiled'
            df = _append_missing_kernel_rows(
                df,
                missing_kernels,
                kernel_map,
                status,
                runtime,
                ete_profiler_xtime,
                targetName,
                exeArgs,
                exe_path,
                source_name,
                sample_idx,
            )

        # Save incrementally
        df = _reorder_output_columns(df)
        return df, sampled_count

    except Exception as e:
        print(f"  Error processing results: {e}")
        return df, 0

def _init_profiling_log(csvFilename):
    output_dir = os.path.dirname(csvFilename)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_name = f'profiling-log-{timestamp}.json'
    if GPU_PREFIX:
        log_name = f'{GPU_PREFIX}_{log_name}'
    log_path = os.path.join(output_dir, log_name)
    with open(log_path, 'w') as f:
        json.dump([], f, indent=2)
    return log_path


def _append_profiling_log(log_path, entry):
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []

    data.append(entry)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)


def _zip_results(csvFilename, log_path, results_dir):
    output_dir = os.path.dirname(csvFilename)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    zip_name = f'profiling-results-{timestamp}.zip'
    if GPU_PREFIX:
        zip_name = f'{GPU_PREFIX}_{zip_name}'
    zip_path = os.path.join(output_dir, zip_name)

    files_to_zip = [csvFilename]
    if log_path:
        files_to_zip.append(log_path)

    ncu_reports = sorted(glob.glob(os.path.join(results_dir, '*.ncu-rep')))
    files_to_zip.extend(ncu_reports)

    try:
        subprocess.run(
            ['zip', '-j', zip_path] + files_to_zip,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        print(f"Created zip archive: {zip_path}")
        return zip_path
    except Exception as e:
        print(f"WARNING: Failed to create zip archive with zip command: {e}")
        try:
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for fpath in files_to_zip:
                    if os.path.isfile(fpath):
                        zf.write(fpath, arcname=os.path.basename(fpath))
            print(f"Created zip archive with Python: {zip_path}")
            return zip_path
        except Exception as e2:
            print(f"ERROR: Failed to create zip archive: {e2}")
            return None


def execute_targets(targets, csvFilename, skipRuns=False, timeout_sec=120, samples=3, log_path=None, gpu_info=None):
    """Execute all targets and gather profiling data"""
    
    # Load existing data if available
    if skipRuns:
        df = pd.DataFrame()
    elif os.path.isfile(csvFilename):
        df = pd.read_csv(csvFilename)
        if 'kernel_executed' not in df.columns:
            df['kernel_executed'] = 'normal'
        if 'sample' not in df.columns:
            df['sample'] = 1
        print(f"Loaded existing data: {df.shape[0]} rows")
    else:
        df = pd.DataFrame()
    
    for target in tqdm(targets, desc='Profiling benchmarks'):
        targetName = target['targetName']
        kernels = target['kernels']
        exeArgs = target['exeArgs']
        exe_path = target.get('exe')
        source_name = os.path.basename(target.get('src') or '')
        runtime = _get_target_model(target) or 'unknown'

        if not kernels:
            print(f"Skipping {targetName} - no kernels found")
            continue

        if target_fully_sampled(target, df, samples):
            print(f"Skipping {targetName} - already sampled (all kernels, all samples)")
            continue

        for sample_idx in range(1, samples + 1):
            report_path = _get_ncu_report_path(target, sample_idx)

            if target_sample_fully_sampled(target, df, sample_idx):
                if report_path and not os.path.exists(report_path):
                    print(f"Reprofiling {targetName} sample {sample_idx} - missing report file")
                else:
                    print(f"Skipping {targetName} sample {sample_idx} - already sampled")
                    continue

            if skipRuns:
                kernel_map = {k.get('mangled'): k for k in kernels if k.get('mangled')}
                expected_kernels = list(kernel_map.keys())
                ncuResult = _parse_ncu_report(report_path, target.get('src'))
                if ncuResult is None:
                    df = _append_missing_kernel_rows(
                        df,
                        expected_kernels,
                        kernel_map,
                        'not profiled',
                        runtime,
                        np.nan,
                        targetName,
                        exeArgs,
                        exe_path,
                        source_name,
                        sample_idx,
                    )
                    df = _reorder_output_columns(df)
                    df.to_csv(csvFilename, quoting=csv.QUOTE_NONNUMERIC, quotechar='"',
                              index=False, na_rep='NULL')
                    print(f"  Marked {targetName} sample {sample_idx} kernels as not profiled (missing report)")
                    continue

                df, sampled_count = _append_ncu_results(
                    df,
                    ncuResult,
                    target,
                    kernel_map,
                    expected_kernels,
                    runtime,
                    exeArgs,
                    exe_path,
                    source_name,
                    np.nan,
                    False,
                    sample_idx,
                )
                df = _reorder_output_columns(df)
                df.to_csv(csvFilename, quoting=csv.QUOTE_NONNUMERIC, quotechar='"',
                          index=False, na_rep='NULL')
                print(f"  Saved data for {targetName} sample {sample_idx} ({sampled_count} kernels)")
                continue

            if df.shape[0] > 0:
                if 'source' in df.columns and source_name and 'sample' in df.columns:
                    existing = df[(df['source'] == source_name) & (df['sample'] == sample_idx)]
                elif 'sample' in df.columns:
                    existing = df[(df['targetName'] == targetName) & (df['sample'] == sample_idx)]
                else:
                    existing = df[df['targetName'] == targetName]

                if not existing.empty:
                    print(f"Reprofiling {targetName} sample {sample_idx} - incomplete kernel data detected")
                    if 'source' in df.columns and source_name and 'sample' in df.columns:
                        df = df[~((df['source'] == source_name) & (df['sample'] == sample_idx))]
                    elif 'sample' in df.columns:
                        df = df[~((df['targetName'] == targetName) & (df['sample'] == sample_idx))]
                    else:
                        df = df[df['targetName'] != targetName]

            kernel_map = {k.get('mangled'): k for k in kernels if k.get('mangled')}
            expected_kernels = list(kernel_map.keys())

            # Execute with NCU once per target sample
            start_ts = datetime.now().isoformat()
            run_result = execute_target_with_ncu(
                target,
                sample_idx,
                timeout_sec=timeout_sec
            )
            end_ts = datetime.now().isoformat()

            if log_path:
                log_entry = {
                    'targetName': targetName,
                    'sample': sample_idx,
                    'timestamp_start': start_ts,
                    'timestamp_end': end_ts,
                    'stdout': run_result.stdout if run_result.stdout is not None else '',
                    'stderr': run_result.stderr if run_result.stderr is not None else '',
                    'stdout_capture': run_result.stdout_capture if run_result.stdout_capture is not None else '',
                    'returncode': run_result.returncode,
                    'command': ' '.join(run_result.cmd) if run_result.cmd else None,
                    'cwd': run_result.cwd,
                    'timed_out': run_result.timed_out,
                    'status': run_result.status,
                    'parse_error': run_result.parse_error,
                    'gpu_info': gpu_info,
                    'report_path': report_path,
                }
                _append_profiling_log(log_path, log_entry)

            if run_result.ncu_result is None:
                if run_result.kernel_executed is False:
                    status = 'timeout' if run_result.timed_out else 'not profiled'
                    df = _append_missing_kernel_rows(
                        df,
                        expected_kernels,
                        kernel_map,
                        status,
                        runtime,
                        run_result.elapsed_sec,
                        targetName,
                        exeArgs,
                        exe_path,
                        source_name,
                        sample_idx,
                    )
                    df = _reorder_output_columns(df)
                    df.to_csv(csvFilename, quoting=csv.QUOTE_NONNUMERIC, quotechar='"',
                              index=False, na_rep='NULL')
                    print(f"  Marked {targetName} sample {sample_idx} kernels as not executed")
                continue

            df, sampled_count = _append_ncu_results(
                df,
                run_result.ncu_result,
                target,
                kernel_map,
                expected_kernels,
                runtime,
                exeArgs,
                exe_path,
                source_name,
                run_result.elapsed_sec,
                run_result.timed_out,
                sample_idx,
            )
            df = _reorder_output_columns(df)
            df.to_csv(csvFilename, quoting=csv.QUOTE_NONNUMERIC, quotechar='"',
                      index=False, na_rep='NULL')

            print(f"  Saved data for {targetName} sample {sample_idx} ({sampled_count} kernels)")
    
    print(f"Profiling complete! Data saved to {csvFilename}")
    print(f"Total samples: {df.shape[0]}")
    
    return df

def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    default_build_dir = os.path.abspath(os.path.join(script_dir, '../build'))
    default_src_dir = os.path.abspath(os.path.join(script_dir, '../HeCBench/src'))
    default_outfile = os.path.abspath(os.path.join(script_dir, 'gpuData.csv'))

    global GPU_INFO, GPU_PREFIX
    GPU_INFO = get_gpu_info()
    GPU_PREFIX = _sanitize_gpu_name(GPU_INFO.get('gpu_name') if GPU_INFO else None)

    parser = argparse.ArgumentParser(
        description='Profile HeCBench benchmarks with NVIDIA Nsight Compute'
    )
    
    parser.add_argument('--buildDir', type=str, default=default_build_dir,
                       help='Directory containing built executables')
    parser.add_argument('--srcDir', type=str, default=default_src_dir,
                       help='Directory containing source files')
    parser.add_argument('--outfile', type=str, default=default_outfile,
                       help='Output CSV file for profiling data')
    parser.add_argument('--skipRuns', action='store_true',
                       help='Skip execution, only parse existing ncu-rep files')
    parser.add_argument('--timeout', type=int, default=120,
                       help='NCU profiling timeout in seconds (default: 120)')
    parser.add_argument('--cudaOnly', action='store_true',
                       help='Profile CUDA targets only')
    parser.add_argument('--ompOnly', action='store_true',
                       help='Profile OpenMP targets only')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of repeat samples per target (default: 3)')
    
    args = parser.parse_args()

    if GPU_PREFIX:
        args.outfile = _apply_gpu_prefix_to_path(args.outfile, GPU_PREFIX)
    
    # Setup directories
    setup_dirs(args.buildDir, args.srcDir)
    
    # Load benchmarks metadata
    benchmarks = load_benchmarks_yaml()
    
    # Get executable targets
    targets = get_runnable_targets()
    targets = sorted(targets, key=lambda t: t.get('targetName', ''))

    if args.cudaOnly and args.ompOnly:
        print("ERROR: --cudaOnly and --ompOnly cannot be used together")
        sys.exit(1)

    if args.cudaOnly:
        targets = [
            t for t in targets
            if t.get('model') == 'cuda' or '-cuda' in (t.get('targetName') or '')
        ]
    elif args.ompOnly:
        targets = [
            t for t in targets
            if t.get('model') == 'omp' or '-omp' in (t.get('targetName') or '')
        ]
    
    if not targets:
        print("ERROR: No executable targets found!")
        sys.exit(1)
    
    # Get execution arguments from YAML
    targets = get_exe_args_from_yaml(targets, benchmarks)
    
    # Extract kernel names
    targets = get_kernel_names(targets)

    summary = summarize_profiliable_kernels(targets)
    total_codes = summary['cuda_profiliable'] + summary['omp_profiliable']
    total_kernels = summary['cuda_kernels'] + summary['omp_kernels']

    print("\n===== Profiling Summary =====")
    print(
        f"Profiliable codes: {total_codes} "
        f"(CUDA: {summary['cuda_profiliable']}, OpenMP: {summary['omp_profiliable']})"
    )
    print(
        f"Profiliable kernels: {total_kernels} "
        f"(CUDA: {summary['cuda_kernels']}, OpenMP: {summary['omp_kernels']})"
    )

    if summary['cuda_missing']:
        print("\nCUDA codes without profiliable kernels:")
        for name in summary['cuda_missing']:
            print(f"  - {name}")

    if summary['omp_missing']:
        print("\nOpenMP codes without profiliable kernels:")
        for name in summary['omp_missing']:
            print(f"  - {name}")

    print("============================\n")
    # Progress summary for already gathered data
    sampled = summarize_existing_sampling_progress(targets, args.outfile, summary)
    sampled_cuda_codes = sampled['cuda_codes']
    sampled_omp_codes = sampled['omp_codes']
    sampled_cuda_kernels = sampled['cuda_kernels']
    sampled_omp_kernels = sampled['omp_kernels']

    def _print_progress(label, sampled_codes, total_codes, sampled_kernels, total_kernels):
        codes_pct = (sampled_codes / total_codes * 100.0) if total_codes else 0.0
        kernels_pct = (sampled_kernels / total_kernels * 100.0) if total_kernels else 0.0
        print(
            f"{label} sampled codes: {sampled_codes}/{total_codes} "
            f"({codes_pct:.1f}%)"
        )
        print(
            f"{label} sampled kernels: {sampled_kernels}/{total_kernels} "
            f"({kernels_pct:.1f}%)"
        )

    print("Progress based on existing data:")
    if args.cudaOnly:
        _print_progress(
            "CUDA",
            sampled_cuda_codes,
            summary['cuda_profiliable'],
            sampled_cuda_kernels,
            summary['cuda_kernels']
        )
    elif args.ompOnly:
        _print_progress(
            "OpenMP",
            sampled_omp_codes,
            summary['omp_profiliable'],
            sampled_omp_kernels,
            summary['omp_kernels']
        )
    else:
        _print_progress(
            "CUDA",
            sampled_cuda_codes,
            summary['cuda_profiliable'],
            sampled_cuda_kernels,
            summary['cuda_kernels']
        )
        _print_progress(
            "OpenMP",
            sampled_omp_codes,
            summary['omp_profiliable'],
            sampled_omp_kernels,
            summary['omp_kernels']
        )

    input("Press Enter to continue profiling...")
    
    # Initialize profiling log
    log_path = _init_profiling_log(args.outfile)

    # Execute and profile
    results = execute_targets(
        targets,
        args.outfile,
        args.skipRuns,
        args.timeout,
        args.samples,
        log_path,
        GPU_INFO,
    )

    # Zip reports, log, and CSV
    results_dir = os.path.join(THIS_DIR, 'ncu-rep-results')
    _zip_results(args.outfile, log_path, results_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
