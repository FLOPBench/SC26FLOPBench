
import pandas as pd
import os
import glob
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
import re
import sys
from tqdm import tqdm

sys.path.append('../')
from utils import *
from gatherData import _parse_ncu_report, roofline_results_to_df, calc_roofline_data

from tqdm.contrib.concurrent import process_map
from functools import partial
from os import path
import csv
from io import StringIO

reports3080 = sorted([ path.abspath(report) for report in glob.glob('./3080/*.ncu-rep')])
reportsA10 = sorted([ path.abspath(report) for report in glob.glob('./A10/*.ncu-rep')])
reportsA100 = sorted([ path.abspath(report) for report in glob.glob('./A100/*.ncu-rep')])
reportsH100 = sorted([ path.abspath(report) for report in glob.glob('./H100/*.ncu-rep')])

rx = re.compile(
    r'^'
    r'(?P<device>.+?(?:A10|3080|40GB|HBM3))_'   # device ends at one of these tokens, then "_"
    r'(?P<source>.+?)-'                        # source name (can include hyphens)
    r'(?P<source_type>omp|cuda)-'              # source type
    r's(?P<sample>\d+)-'                       # sample number digits
    r'report\.ncu-rep'
    r'$'
)

def parse_filename(fname: str):
    m = rx.match(Path(fname).name)
    if not m:
        raise ValueError(f"Unmatched filename: {fname}")
    d = m.groupdict()
    d["sample"] = int(d["sample"])            # "1" -> 1
    return d

# filter out repeat samples -- so we don't have duplicate SASS instructions
# if we already have 1 sample, we don't need another
# compare the filesizes so we grab the largest one
# def drop_duplicate_samples(reportsList):
#     # group the reports by device/source/sample
#     grouped_reports = {}
#     for report in reportsList:
#         filename = path.basename(report)
#         parsed_name = parse_filename(filename)
#         key = (parsed_name['device'], parsed_name['source'], parsed_name['source_type'])
#         if key not in grouped_reports:
#             grouped_reports[key] = []
#         grouped_reports[key].append(report)
# 
#         for group, files in grouped_reports.items():
#             if len(files) > 1:
#                 # we have duplicates, keep the largest file
#                 files.sort(key=lambda x: os.path.getsize(x), reverse=True)
#                 # keep only the largest file
#                 grouped_reports[group] = [files[0]]
# 
#     # flatten the grouped reports back into a single list
#     deduped_reports = []
#     for files in grouped_reports.values():
#         deduped_reports.extend(files)
# 
#     return deduped_reports

# reports3080 = drop_duplicate_samples(reports3080)
# reportsA10 = drop_duplicate_samples(reportsA10)
# reportsA100 = drop_duplicate_samples(reportsA100)
# reportsH100 = drop_duplicate_samples(reportsH100)

# markers we should ignore / drop kernels containing these from the dataset
#library_markers = [ 'cub::', 'thrust::', '__cuda_' ]

# library_markers = [ 'cub::', 'thrust::' ]

def _parse_ncu_sass_report(report_path, src_dir='./'):
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
                '--csv', '--print-units', 'base', '--page', 'source',
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

def sass_results_to_df(ncuResult):
    stringified = ncuResult.stdout.decode('UTF-8')
    split_by_kernel_names = stringified.split('"Kernel Name",')
    # remove any empty strings
    split_by_kernel_names = [s for s in split_by_kernel_names if s.strip() != '']
    #print('number of kernels', len(split_by_kernel_names))

    df = pd.DataFrame()

    dfs = [] 

    # process each "kernel name" group
    for raw_kernel in split_by_kernel_names:
        lines = raw_kernel.split('\n')
        lines = [l for l in lines if l.strip() != '']

        #print('\t\tnum lines', len(lines))
        kernel_name = lines[0].strip('",')
        csv_data = '\n'.join(lines[1:])
        #print(csv_data)
        csv_df = pd.read_csv(StringIO(csv_data), quotechar='"')
        csv_df['Kernel Name'] = kernel_name
        dfs.append(csv_df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_device_type(deviceName):
    if '3080' in deviceName:
        return '3080'

    elif 'A100' in deviceName:
        return 'A100'

    elif 'A10' in deviceName:
        return 'A10'

    elif 'H100' in deviceName:
        return 'H100'

    else:
        raise ValueError('Unknown GPU')
    



def _process_single_report(report):
    ncuResult = _parse_ncu_sass_report(report)

    #print(report, flush=True)
    rawDF = sass_results_to_df(ncuResult)

    # drop the first row which is units
    # rawDF = rawDF.iloc[1:].copy(deep=True)

    if rawDF.empty:
        print(f'  No SASS instructions parsed from {report}')
        return None

    filename = path.basename(report)
    parsed_name = parse_filename(filename)

    rawDF['device'] = get_device_type(parsed_name['device'])

    rawDF['codenmae'] = parsed_name['source']
    rawDF['source'] = parsed_name['source'] + '-' + parsed_name['source_type']
    rawDF['sample'] = parsed_name['sample']

    model_type = parsed_name['source_type']
    rawDF['model_type'] = model_type

    if model_type == 'cuda':
        rawDF['Demangled Name'] = rawDF['Kernel Name'].apply(lambda x: demangle_kernel_name(x))
    elif model_type == 'omp':
        rawDF['Demangled Name'] = rawDF['Kernel Name'].apply(lambda x: demangle_omp_offload_name(x))
    else:
        raise ValueError(f'Incorrect model_type given: [{model_type}]')

    # drop any rows that contain library markers
    # beforeRows = rawDF.shape[0]
    # rawDF = rawDF[~rawDF['Demangled Name'].str.contains('|'.join(library_markers))]
    # afterRows = rawDF.shape[0]
    # dropped = beforeRows - afterRows
    # if dropped > 0:
    #     print(f'\t  Dropped {dropped} library/kernel rows from {report}')

    #     if rawDF.empty:
    #         print(f'  No roofline data remaining after dropping library kernels from {report}')
    #         return None

    # we need to extract the exeArgs from the og dataframes
    # NOTE: this will break if we execute a code more than once with diff exe args
    exeArgs = extract_exe_args_from_ncu_report(report)
    rawDF['exeArgs'] = ' '.join(exeArgs) if exeArgs else None

    # limit the output columns to the cols we care about
    #return roofDF[all_cols]
    return rawDF 

# had to multithread this operation because single-thread is 30 mins for all reports
def load_ncu_reports(reportsList, max_workers=None, chunksize=1):
    fn = partial(_process_single_report)
    dfs = process_map(fn, reportsList,
                      max_workers=max_workers or os.cpu_count(),
                      chunksize=chunksize,
                      desc=f"Processing {len(reportsList)} reports")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

num_threads = 8

df3080 = load_ncu_reports(reports3080, num_threads)
print()
dfA10 = load_ncu_reports(reportsA10, num_threads)
print()
dfA100 = load_ncu_reports(reportsA100, num_threads)
print()
dfH100 = load_ncu_reports(reportsH100, num_threads)

#assert df3080.columns.tolist() == dfA10.columns.tolist() == dfA100.columns.tolist() == dfH100.columns.tolist(), "DataFrames have different columns!"

# automatically fills missing columns with NaN values
df = pd.concat([df3080, dfA10, dfA100, dfH100], ignore_index=True)

# save the output to a CSV file
df.to_csv('all-NCU-SASS-Data.csv', index=False)

print('CSV File Saved')