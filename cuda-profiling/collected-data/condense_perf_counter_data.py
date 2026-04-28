
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

reports3080 = sorted([ path.abspath(report) for report in glob.glob('./3080/*.ncu-rep')])
reportsA10 = sorted([ path.abspath(report) for report in glob.glob('./A10/*.ncu-rep')])
reportsA100 = sorted([ path.abspath(report) for report in glob.glob('./A100/*.ncu-rep')])
reportsH100 = sorted([ path.abspath(report) for report in glob.glob('./H100/*.ncu-rep')])

# kernel key and metrics
key_cols = ['source', 'Kernel Name', 'exeArgs', 'Block Size', 'Grid Size', 'model_type']
device_col = 'device'

groupings = key_cols + [device_col]
metrics = ['SP_FLOP', 'DP_FLOP', 'HP_FLOP', 'INTOP', 'traffic',
           'bytesRead', 'bytesWrite', 'bytesTotal',
           'dpAI', 'spAI', 'hpAI',
           'dpPerf', 'spPerf', 'hpPerf', 'xtime']

log_metrics = ['traffic', 'xtime']

# all cols
all_cols = groupings + metrics + ['sample']

# markers we should ignore / drop kernels containing these from the dataset
# library_markers = [ 'cub::', 'thrust::', '__cuda_' ]


def _process_single_report(report):
    ncuResult = _parse_ncu_report(report)

    rawDF = roofline_results_to_df(ncuResult)

    # drop any kernel names that contain

    # drop the first row which is units
    rawDF = rawDF.iloc[1:].copy(deep=True)

    if rawDF['dram__bytes_read.sum'].isna().any() or rawDF['dram__bytes_write.sum'].isna().any():
        print(f"  WARNING: NCU report contains NaN values for some metrics for {report}")
        # we need to drop these rows to avoid errors

    rawDF = rawDF[(rawDF['dram__bytes_read.sum'].notna()) & (rawDF['dram__bytes_write.sum'].notna())]

    roofDF = calc_roofline_data(rawDF)

    if roofDF.empty:
        print(f'  No roofline data parsed from {report}')
        return None

    # we need to add the source and modelType columns manually
    # the report file name is in the format of <device>_source-modelType-ssampleNumber-report.ncu-rep
    device_name = roofDF['device'].unique()[0].replace(' ', '_')
    filename = path.basename(report)
    match = re.match(rf'{device_name}_(.+)-(.+)-s(\d+)-report\.ncu-rep', filename)

    source = match.group(1) if match else 'unknown'
    model_type = match.group(2) if match else 'unknown'
    sample = match.group(3) if match else 'unknown'

    assert source != 'unknown', f'Could not parse source from filename {filename}'  
    assert sample != 'unknown', f'Could not parse sample from filename {filename}'
    assert model_type != 'unknown', f'Could not parse model_type from filename {filename}'
    assert model_type in ['cuda', 'omp'], f'Parsed model_type {model_type} is not valid from filename {filename}'

    source_name = source+'-'+model_type
    roofDF['codename'] = source
    roofDF['source'] = source_name
    roofDF['model_type'] = model_type
    roofDF['sample'] = sample

    # for some of the CUDA kernels, if we
    if model_type == 'cuda':
        roofDF['Demangled Name'] = roofDF['Kernel Name'].apply(lambda x: demangle_kernel_name(x))
    elif model_type == 'omp':
        roofDF['Demangled Name'] = roofDF['Kernel Name'].apply(lambda x: demangle_omp_offload_name(x))

    # drop any rows that contain library markers
    # beforeRows = roofDF.shape[0]
    # roofDF = roofDF[~roofDF['Demangled Name'].str.contains('|'.join(library_markers))]
    # afterRows = roofDF.shape[0]
    # dropped = beforeRows - afterRows
    # if dropped > 0:
    #     print(f'\t  Dropped {dropped} library/kernel rows from {report}')

    #     if roofDF.empty:
    #         print(f'  No roofline data remaining after dropping library kernels from {report}')
    #         return None

    # we need to extract the exeArgs from the og dataframes
    # NOTE: this will break if we execute a code more than once with diff exe args
    exeArgs = extract_exe_args_from_ncu_report(report)
    roofDF['exeArgs'] = ' '.join(exeArgs) if exeArgs else None

    # limit the output columns to the cols we care about
    #return roofDF[all_cols]
    return roofDF

# had to multithread this operation because single-thread is 30 mins for all reports
def load_ncu_reports(reportsList, max_workers=None, chunksize=1):
    fn = partial(_process_single_report)
    dfs = process_map(fn, reportsList,
                      max_workers=max_workers or os.cpu_count(),
                      chunksize=chunksize,
                      desc=f"Processing {len(reportsList)} reports")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


df3080 = load_ncu_reports(reports3080, 8)
print()
dfA10 = load_ncu_reports(reportsA10, 8)
print()
dfA100 = load_ncu_reports(reportsA100, 8)
print()
dfH100 = load_ncu_reports(reportsH100, 8)

#assert df3080.columns.tolist() == dfA10.columns.tolist() == dfA100.columns.tolist() == dfH100.columns.tolist(), "DataFrames have different columns!"

# automatically fills missing columns with NaN values
df = pd.concat([df3080, dfA10, dfA100, dfH100], ignore_index=True)

# save the output to a CSV file
df.to_csv('all-NCU-GPU-Data.csv', index=False)

print('CSV File Saved')