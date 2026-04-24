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

from utils import demangle_kernel_name, demangle_omp_offload_name
from sass_objs import SASSFileParser

def fix_omp_kernel_name(x):
    if pd.isna(x):
        return x
    x_str = str(x)
    if '__omp_offloading' in x_str:
        match = re.search(r'__omp_offloading_.*?_.*?_(.+)$', x_str)
        if match:
            return match.group(1)
    return x_str

def fix_cuda_static_kernel_name(x):
    if pd.isna(x):
        return x
    x_str = str(x)
    if '__nv_static_' in x_str:
        mangled_start = x_str.find('_Z')
        if mangled_start != -1:
            return x_str[mangled_start:]
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
    else:
        raise ValueError(f'Unknown device name in {x_str}')

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
    needed_cols = ['Process Name', 'Program Name', 'Kernel Name', 'device', 'Block Size', 'Grid Size', 'exeArgs', 'xtime', 'bytesRead', 'bytesWrite', 'HP_FLOP', 'SP_FLOP', 'DP_FLOP']
    df = pd.read_csv(csv_path, low_memory=False, usecols=lambda x: x in needed_cols)

    # First determine program name (needs raw Kernel Name to check for __omp_offloading)
    df['Program Name'] = df.apply(get_program_name, axis=1)
    
    # Now fix the OpenMP kernel names to drop the hash, grouping them properly
    df['Kernel Name'] = df['Kernel Name'].apply(fix_omp_kernel_name)
    cuda_mask = df['Program Name'].str.endswith('-cuda', na=False)
    df.loc[cuda_mask, 'Kernel Name'] = df.loc[cuda_mask, 'Kernel Name'].apply(fix_cuda_static_kernel_name)
    df['Fixed Kernel Name'] = df['Kernel Name']

    print("Demangling kernels...")
    unique_kernels = df[['Program Name', 'Kernel Name']].dropna().drop_duplicates()
    demangled_map = {}
    for _, row in tqdm(unique_kernels.iterrows(), desc="Demangling kernels", total=len(unique_kernels)):
        p = row['Program Name']
        k = row['Kernel Name']
        if pd.isna(k): continue
        if '-omp' in p:
            demangled_map[(p, k)] = get_demangled_omp_name(k)
        else:
            demangled_map[(p, k)] = demangle_kernel_name(k)
            
    df['Demangled Name'] = df.apply(lambda r: demangled_map.get((r['Program Name'], r['Kernel Name']), r['Kernel Name']), axis=1)
    
    df['device'] = df['device'].apply(rename_devices)

    cols_to_keep = ['Program Name', 'Kernel Name', 'Fixed Kernel Name', 'Demangled Name', 'device', 'Block Size', 'Grid Size', 'exeArgs', 'xtime', 'bytesRead', 'bytesWrite', 'HP_FLOP', 'SP_FLOP', 'DP_FLOP']
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]

    groupby_cols = ['Program Name', 'Kernel Name', 'Fixed Kernel Name', 'Demangled Name', 'device', 'Block Size', 'Grid Size', 'exeArgs']
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
                if program_name.endswith('-cuda') and fix_cuda_static_kernel_name(k) == kernel_mangled:
                    matched_key = k
                    break
                    
        if not matched_key:
            return None, None
            
        imix_data, _ = parser.getIMIXForKernel(matched_key)
        
        sass_sections = {}
        visited = set()
        
        def traverse_sass(k_name):
            if k_name in visited or k_name not in parser.text_sections:
                return
            visited.add(k_name)
            sec = parser.text_sections[k_name]
            
            # Clean up the trailing non-SASS sections
            clean_lines = []
            
            # Since we now include the header, skip the first line during the inner 
            # cleanup loop so we don't accidentally break on its own `//---` tag
            lines = sec.raw_text.split('\n')
            if lines:
                clean_lines.append(lines[0])
                for line in lines[1:]:
                    if line.startswith('//---------------------'):
                        break
                    clean_lines.append(line)
            
            clean_text = '\n'.join(clean_lines).strip()
            sass_sections[k_name] = clean_text
            
            for ref in sec.references:
                traverse_sass(ref)
                
        traverse_sass(matched_key)
        sorted_imix = dict(sorted(imix_data.items(), key=lambda item: item[1], reverse=True))
        return sorted_imix, sass_sections
    except Exception as e:
        print(f"Error parsing SASS {sass_file}: {e}")
        return None, None

def extract_source_mapping(program_name, kernel_mangled, demangled_name, sources_dict):
    mapped_files = []
    
    # Remove OpenMP line tag e.g. ":l25" or ":l57"
    line_no = None
    line_match = re.search(r':l(\d+)$', demangled_name)
    if line_match:
        line_no = int(line_match.group(1))
    elif '-omp' in program_name:
        omp_match = re.search(r'_l(\d+)$', fix_omp_kernel_name(kernel_mangled))
        if omp_match:
            line_no = int(omp_match.group(1))

    clean_demangled = re.sub(r':l\d+$', '', demangled_name)
    
    # Strip return types, templates, and parameters to get just the function base name
    base_name = clean_demangled.split('(')[0].split('<')[0]
    for prefix in ["void ", "virtual ", "static ", "inline "]:
        base_name = base_name.replace(prefix, "")
    base_name = base_name.strip()
    
    base_name_clean = base_name.split('::')[-1]
    
    # Fallback to old heuristic if demangling yielded empty
    if not base_name_clean:
        if '-omp' in program_name:
            match = re.search(r'(.+)_l(\d+)$', fix_omp_kernel_name(kernel_mangled))
            if match:
                base_name_clean = match.group(1)
            else:
                base_name_clean = fix_omp_kernel_name(kernel_mangled)
        else:
            base_name_clean = fix_cuda_static_kernel_name(kernel_mangled)

    for f_path, content in sources_dict.items():
        # Ensure it's a whole word match before doing anything
        matches = list(re.finditer(r'\b' + re.escape(base_name_clean) + r'\b', content))
        if not matches:
            continue

        if '-omp' in program_name:
            # Stricter cross-referencing for OpenMP kernels
            if line_no:
                lines = content.split('\n')
                # Check if the file is even long enough to support this line tag
                if line_no > len(lines):
                    continue
                    
                # Look in a window around the target line for standard OpenMP/Function hints
                start = max(0, line_no - 15)
                end = min(len(lines), line_no + 15)
                window = "\n".join(lines[start:end]).lower()
                
                # If OpenMP directives aren't near the line, this file is a false positive 
                if 'omp' not in window:
                    continue
            else:
                if 'omp' not in content.lower():
                    continue
        else:
            # For CUDA, cleanly distinguish calls/declarations versus actual kernel definitions
            launch_or_decl_count = 0
            for m in matches:
                tail = content[m.start():]
                # A match denotes a launch if followed by <<<, or a declaration if followed by ; 
                # before ever encountering { which starts a function body
                if re.match(r'\b' + re.escape(base_name_clean) + r'\b[^{;]*<<<|\b' + re.escape(base_name_clean) + r'\b[^{]*;', tail):
                    launch_or_decl_count += 1
                    
            if len(matches) == launch_or_decl_count:
                # If every occurrence is merely calling or declaring the kernel, skip the file
                continue

        mapped_files.append(f_path)
    
    return list(set(mapped_files))

def normalize_path(path):
    if "HeCBench/" in path:
        return "HeCBench/" + path.split("HeCBench/", 1)[-1]
    return path

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
                file_path = str(cmd.get("file", ""))
                filtered.append({
                    "file": normalize_path(file_path),
                    "command": cmd.get("command")
                })
        return filtered
    except Exception:
        return []

def filter_kernels_with_complete_gpu_coverage(dataset, required_gpus):
    dropped_counts = {gpu: 0 for gpu in required_gpus}
    kept_counts = {gpu: 0 for gpu in required_gpus}
    filtered_dataset = {}

    for prog, data in dataset.items():
        kept_kernels = {}
        kept_source_to_kernels = defaultdict(list)

        for kernel_name, kernel_data in data["kernels"].items():
            kernel_gpus = set(kernel_data.get("metrics", {}).keys())
            missing_gpus = [gpu for gpu in required_gpus if gpu not in kernel_gpus]

            if missing_gpus:
                for gpu in missing_gpus:
                    dropped_counts[gpu] += 1
                continue

            kept_kernels[kernel_name] = kernel_data
            for gpu in required_gpus:
                kept_counts[gpu] += 1

        if not kept_kernels:
            continue

        for source_path, kernel_names in data["source_to_kernels"].items():
            filtered_kernel_names = [kernel_name for kernel_name in kernel_names if kernel_name in kept_kernels]
            if filtered_kernel_names:
                kept_source_to_kernels[source_path] = filtered_kernel_names

        filtered_dataset[prog] = {
            "exeArgs": data["exeArgs"],
            "source_to_kernels": kept_source_to_kernels,
            "kernels": kept_kernels,
            "compile_commands": data["compile_commands"],
            "sources": data["sources"]
        }

    kept_totals = list(kept_counts.values())
    assert len(set(kept_totals)) == 1, f"Kept kernel counts differ across GPUs: {kept_counts}"

    print("Dropped kernels due to missing GPU samples:")
    for gpu in required_gpus:
        print(f"  {gpu}: {dropped_counts[gpu]}")

    print("Kernels kept in final dataset:")
    for gpu in required_gpus:
        print(f"  {gpu}: {kept_counts[gpu]}")

    return filtered_dataset

def main():
    csv_path = ROOT_DIR / "cuda-profiling" / "collected-data" / "all-NCU-GPU-Data.csv"
    sass_dir = ROOT_DIR / "cuda-profiling" / "collected-data" / "scraped-sass"
    sources_json_path = ROOT_DIR / "dataset-creation" / "scraped_sources.json"
    required_gpus = ["3080", "A10", "A100", "H100"]
    
    if not csv_path.exists():
        print("Missing CSV path. Exit.")
        sys.exit(1)
        
    df_agg = build_metrics_db(str(csv_path))
    
    scraped_sources = {}
    if sources_json_path.exists():
        with open(sources_json_path, 'r') as f:
            raw_sources = json.load(f)
            for prog, srcs in raw_sources.items():
                scraped_sources[prog] = {}
                for path, content in srcs.items():
                    scraped_sources[prog][normalize_path(path)] = content
        
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
        
        gpu_device = row['device']
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
    dataset = filter_kernels_with_complete_gpu_coverage(dataset, required_gpus)

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
