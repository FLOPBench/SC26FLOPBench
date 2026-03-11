import sys
import os
import glob
import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "cuda-profiling"))
sys.path.append(str(ROOT_DIR / "cuda-profiling" / "collected-data"))
sys.path.append(str(ROOT_DIR / "gpuFLOPBench-agentic" / "langchain-tools" / "code-search-tools"))
sys.path.append(str(ROOT_DIR / "gpuFLOPBench-agentic" / "langchain-tools" / "treesitter-tools"))

from utils import demangle_kernel_name, demangle_omp_offload_name
from condense_sass_data import SASSFileParser
from condense_sass_data import SASSFileParser

def fix_omp_kernel_name(x):
    if pd.isna(x):
        return x
    x_str = str(x)
    if '__omp_offloading' in x_str:
        match = re.search(r'__omp_offloading_.*?_.*?_(.+)$', x_str)
        if match:
            return match.group(1)
    return x_str

def get_demangled_omp_name(k):
    mangled_core = fix_omp_kernel_name(k)
    line_tag = ""
    line_match = re.search(r'^(.*?)_l(\d+)$', mangled_core)
    if line_match:
        mangled_func = line_match.group(1)
        line_tag = f":l{line_match.group(2)}"
    else:
        mangled_func = mangled_core
        
    if mangled_func.startswith("_Z"):
        demangled = demangle_kernel_name(mangled_func)
        if demangled != mangled_func:
            return f"{demangled}{line_tag}"
            
    return f"{mangled_func}{line_tag}"

def rename_devices(x):
    if pd.isna(x):
        return x
    x_str = str(x)
    if '3080' in x_str:
        return '3080'
    elif 'A100' in x_str:
        return 'A100'
    elif 'A10' in x_str:
        return 'A10'
    elif 'H100' in x_str:
        return 'H100'
    return x_str

def get_program_name(row):
    pname = str(row['Process Name'])
    kname = str(row['Kernel Name'])
    if '__omp_offloading' in kname:
        return f"{pname}-omp"
    else:
        return f"{pname}-cuda"

def build_metrics_db(csv_path):
    print("Loading CSV Data...")
    
    # Only load the columns we actually need to save memory
    needed_cols = ['Process Name', 'Program Name', 'Kernel Name', 'Device', 'Block Size', 'Grid Size', 'exeArgs', 'xtime', 'bytesRead', 'bytesWrite', 'HP_FLOP', 'SP_FLOP', 'DP_FLOP']
    df = pd.read_csv(csv_path, low_memory=False, usecols=lambda x: x in needed_cols)
    
    print("Demangling kernels...")
    unique_kernels = df['Kernel Name'].dropna().unique()
    demangled_map = {}
    for k in tqdm(unique_kernels, desc="Demangling kernels", total=len(unique_kernels)):
        if '__omp_offloading' in str(k):
            demangled_map[k] = get_demangled_omp_name(k)
        else:
            demangled_map[k] = demangle_kernel_name(k)
            
    df['Demangled Name'] = df['Kernel Name'].map(demangled_map)
    df['Fixed Kernel Name'] = df['Kernel Name'].apply(fix_omp_kernel_name)
    df['Device'] = df['Device'].apply(rename_devices)
    df['Program Name'] = df.apply(get_program_name, axis=1)

    cols_to_keep = ['Program Name', 'Kernel Name', 'Fixed Kernel Name', 'Demangled Name', 'Device', 'Block Size', 'Grid Size', 'exeArgs', 'xtime', 'bytesRead', 'bytesWrite', 'HP_FLOP', 'SP_FLOP', 'DP_FLOP']
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]

    groupby_cols = ['Program Name', 'Kernel Name', 'Fixed Kernel Name', 'Demangled Name', 'Device', 'Block Size', 'Grid Size', 'exeArgs']
    groupby_cols_existing = [c for c in groupby_cols if c in df.columns]

    agg_dict = {}
    for metric in ['xtime', 'bytesRead', 'bytesWrite', 'HP_FLOP', 'SP_FLOP', 'DP_FLOP']:
        if metric in df.columns:
            agg_dict[metric] = 'mean'
            
    df_agg = df.groupby(groupby_cols_existing, dropna=False, as_index=False).agg(agg_dict)
    
    for metric in agg_dict.keys():
        df_agg[metric] = df_agg[metric].round().astype(int)
        
    return df_agg

def get_sass_and_imix(program_name, sm_version, sass_dir, kernel_mangled):
    sass_file = os.path.join(sass_dir, f"{program_name}_{sm_version}.sass")
    if not os.path.exists(sass_file):
        return None, None
        
    try:
        parser = SASSFileParser(sass_file)
        
        # SASS files contain the full compiler-hashed name but kernel_mangled has been unified/fixed
        matched_key = None
        if kernel_mangled in parser.text_sections:
            matched_key = kernel_mangled
        else:
            for k in parser.text_sections.keys():
                if fix_omp_kernel_name(k) == kernel_mangled:
                    matched_key = k
                    break
                    
        if not matched_key:
            return None, None
            
        imix_data, _ = parser.getIMIXForKernel(matched_key)
        
        raw_sass_text = ""
        visited = set()
        
        def traverse_sass(k_name):
            nonlocal raw_sass_text
            if k_name in visited or k_name not in parser.text_sections:
                return
            visited.add(k_name)
            sec = parser.text_sections[k_name]
            raw_sass_text += f"// {k_name}\n{sec.raw_text}\n\n"
            for ref in sec.references:
                traverse_sass(ref)
                
        traverse_sass(matched_key)
        return dict(imix_data), raw_sass_text.strip()
    except Exception as e:
        print(f"Error parsing SASS {sass_file}: {e}")
        return None, None

def extract_source_mapping(program_name, kernel_mangled, demangled_name, sources_dict):
    mapped_files = []
    
    if '-omp' in program_name:
        match = re.search(r'(.+)_l(\d+)$', fix_omp_kernel_name(kernel_mangled))
        if match:
            func_name, line_no = match.group(1), int(match.group(2))
            for f_path, content in sources_dict.items():
                if func_name in content:
                    mapped_files.append(f_path)
    else:
        base_name = demangled_name.split('(')[0].split('<')[0].replace("void ", "").strip()
        base_name_clean = base_name.split('::')[-1]
        for f_path, content in sources_dict.items():
            if base_name_clean in content:
                mapped_files.append(f_path)
    
    return list(set(mapped_files))

def build_compile_commands(program_name, gpu="3080"):
    base_file = ROOT_DIR / "cuda-profiling" / "collected-data" / gpu / "compile_commands.json"
    if not base_file.exists():
        return []
        
    try:
        with open(base_file, 'r') as f:
            cmds = json.load(f)
            
        filtered = []
        for cmd in cmds:
            if program_name in cmd.get('directory', '') or program_name in cmd.get('file', ''):
                filtered.append({
                    "file": cmd.get("file"),
                    "command": cmd.get("command")
                })
        return filtered
    except Exception:
        return []

def main():
    csv_path = ROOT_DIR / "cuda-profiling" / "collected-data" / "all-NCU-GPU-Data.csv"
    sass_dir = ROOT_DIR / "cuda-profiling" / "collected-data" / "scraped-sass"
    sources_json_path = ROOT_DIR / "dataset-creation" / "scraped_sources.json"
    
    if not csv_path.exists():
        print("Missing CSV path. Exit.")
        sys.exit(1)
        
    df_agg = build_metrics_db(str(csv_path))
    
    scraped_sources = {}
    if sources_json_path.exists():
        with open(sources_json_path, 'r') as f:
            scraped_sources = json.load(f)
        
    dataset = {}
    
    print("Building comprehensive JSON schema...")
    for _, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
        prog = row['Program Name']
        kmangled = row['Kernel Name']
        
        if prog not in dataset:
            dataset[prog] = {
                "exeArgs": row.get('exeArgs', ""),
                "source_to_kernels": defaultdict(list),
                "kernels": {},
                "compile_commands": {},
                "sources": scraped_sources.get(prog, {})
            }
            
            for gpu in ["3080", "A10", "A100", "H100"]:
                cmds = build_compile_commands(prog, gpu)
                if cmds:
                    dataset[prog]["compile_commands"][gpu] = cmds
                    
        if kmangled not in dataset[prog]["kernels"]:
            dataset[prog]["kernels"][kmangled] = {
                "demangledName": row.get('Demangled Name', ""),
                "gridSz": str(row.get('Grid Size', "")),
                "blockSz": str(row.get('Block Size', "")),
                "metrics": {},
                "imix": {},
                "sass_code": {}
            }
            
            mapped_srcs = extract_source_mapping(prog, kmangled, row.get('Demangled Name', kmangled), dataset[prog]["sources"])
            for m in mapped_srcs:
                if kmangled not in dataset[prog]["source_to_kernels"][m]:
                    dataset[prog]["source_to_kernels"][m].append(kmangled)
            
            for sm, tag in [("sm_80", "A100"), ("sm_86", "3080"), ("sm_90", "H100")]:
                imix, sass_text = get_sass_and_imix(prog, sm, str(sass_dir), kmangled)
                if imix and sass_text:
                    dataset[prog]["kernels"][kmangled]["imix"][sm] = imix
                    dataset[prog]["kernels"][kmangled]["sass_code"][sm] = sass_text
        
        gpu_device = row['Device']
        metrics = {
            "xtime_ns": int(row.get('xtime', 0)) if pd.notna(row.get('xtime')) else 0,
            "bytesRead": int(row.get('bytesRead', 0)) if pd.notna(row.get('bytesRead')) else 0,
            "bytesWritten": int(row.get('bytesWrite', 0)) if pd.notna(row.get('bytesWrite')) else 0
        }
        if 'HP_FLOP' in row and pd.notna(row['HP_FLOP']): metrics["HP_FLOP"] = int(row["HP_FLOP"])
        if 'SP_FLOP' in row and pd.notna(row['SP_FLOP']): metrics["SP_FLOP"] = int(row["SP_FLOP"])
        if 'DP_FLOP' in row and pd.notna(row['DP_FLOP']): metrics["DP_FLOP"] = int(row["DP_FLOP"])
        
        dataset[prog]["kernels"][kmangled]["metrics"][gpu_device] = metrics

    print("Formatting and shifting 'sources' layout...")
    ordered_dataset = {}
    for prog, data in dataset.items():
        ordered_dataset[prog] = {
            "exeArgs": data["exeArgs"],
            "source_to_kernels": dict(data["source_to_kernels"]),
            "kernels": data["kernels"],
            "compile_commands": data["compile_commands"],
            "sources": data["sources"]
        }
        
    out_path = ROOT_DIR / "dataset-creation" / "gpuFLOPBench.json"
    with open(out_path, 'w') as f:
        json.dump(ordered_dataset, f, indent=2)
        
    print(f"Dataset successfully created at {out_path}!")

if __name__ == "__main__":
    main()
