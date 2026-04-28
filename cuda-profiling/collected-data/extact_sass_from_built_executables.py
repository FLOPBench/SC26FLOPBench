# this script simply invokes the cuobjdump and nvdisasm tools to extract SASS from built CUDA executables


# for CUDA codes, the workflow is as follows:
# 1) extract all embedded cubins as standalone files (this creates .cubin files)
# cuobjdump ./bin/cuda/atan2 -xelf all 

# 2) disassemble the cubins to get SASS
# nvdisasm -g ./<something>.sm_86.cubin > <something>.sm_86.sass

# for OpenMP codes, the workflow is as follows:
# 1) extract the sm_86 image (cubin)
# /usr/lib/llvm-21/bin/clang-offload-packager ./bin/omp/atan2 --image=triple=nvptx64-nvidia-cuda,arch=sm_86,file=atan2_sm86.cubin 

# 2) disassemble the cubins to get SASS
# nvdisasm -g atan2_sm86.cubin > atan2_sm86.sass

# we ideally want to make sure no output files have naming conflicts
# we also want to make sure that we don't overwrite any existing files

# the sm_XX code may change to sm_80 or sm_90, so we should build a flexible script that can handle different SM versions
# the `cuobjdump -xelf all` will extract all embedded cubins from the CUDA executable and save them as separate .cubin files
# we will need to detect what it puts out, and pass them to be disassembled accordingly

import os
import shutil
import subprocess
import re
import zipfile
import sys
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust paths to match project structure relative to this script
CUDA_BIN_DIR = os.path.join(THIS_DIR, "..", "..", "build", "bin", "cuda")
OMP_BIN_DIR = os.path.join(THIS_DIR, "..", "..", "build", "bin", "omp")
OUTPUT_SASS_DIR = os.path.join(THIS_DIR, "scraped-sass")

# Tool paths
CLANG_OFFLOAD_PACKAGER = "/usr/lib/llvm-21/bin/clang-offload-packager"
LLVM_OBJDUMP = "/usr/lib/llvm-21/bin/llvm-objdump"

# Architectures allowed
CUDA_SM_VERSIONS = ["sm_80", "sm_86", "sm_90"]

def get_executables_in_dir(dir_path):
    """Recursively find all executable files in a directory."""
    executables = []
    if not os.path.exists(dir_path):
        return []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            full_path = os.path.join(root, file)
            # Check if executable and not a directory
            if os.access(full_path, os.X_OK) and not os.path.isdir(full_path):
                executables.append(os.path.abspath(full_path))
    return executables

def extract_cuda_sass(exe_path, output_dir):
    """
    Extracts SASS from a CUDA executable.
    Uses cuobjdump to extract cubins, then nvdisasm to get SASS.
    Handles multiple cubins/SM versions by creating separate/merged files.
    """
    basename = os.path.basename(exe_path)
    # Create a temp dir for extraction to handle multiple files cleanly
    temp_dir = os.path.join(output_dir, f".temp_{basename}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Extract all elf sections (cubins)
        # cuobjdump -xelf all <executable>
        # Output files are dumped into cwd (temp_dir)
        cmd = ["cuobjdump", "-xelf", "all", exe_path]
        subprocess.run(cmd, cwd=temp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        
        # Find all .cubin files
        cubins = [f for f in os.listdir(temp_dir) if f.endswith(".cubin")]
        
        if not cubins:
            # print(f"No embedded cubins found in {basename}")
            return

        # Group cubins by SM version
        # Naming convention from cuobjdump is typically {exe_name}.sm_{arch}.cubin
        cubins_by_sm = {}
        for cubin in cubins:
            # Look for sm_XX pattern in filename
            match = re.search(r'(sm_\d+)', cubin)
            if match:
                sm = match.group(1)
                if sm not in cubins_by_sm:
                    cubins_by_sm[sm] = []
                cubins_by_sm[sm].append(cubin)
            else:
                # Handle unknown naming if necessary, or log
                pass

        # Process each SM group
        for sm, files in cubins_by_sm.items():
            sass_filename = f"{basename}-cuda_{sm}.sass"
            sass_path = os.path.join(output_dir, sass_filename)
            
            # Merge logic: if multiple cubins for same sm, append their SASS to one file
            with open(sass_path, "w") as sass_file:
                for cubin_file in sorted(files):
                    src_cubin = os.path.join(temp_dir, cubin_file)
                    
                    # Move cubin to output dir with unambiguous name for preservation
                    # {basename}-cuda-{original_cubin_name}
                    dest_cubin_name = f"{basename}-cuda-{cubin_file}"
                    dest_cubin_path = os.path.join(output_dir, dest_cubin_name)
                    
                    shutil.move(src_cubin, dest_cubin_path)
                    
                    # Disassemble
                    sass_file.write(f"// Disassembly for {dest_cubin_name}\n")
                    subprocess.run(["nvdisasm", "-g", dest_cubin_path], stdout=sass_file, stderr=subprocess.DEVNULL, check=False)
                    sass_file.write("\n\n")
                    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def extract_omp_sass(exe_path, output_dir):
    """
    Extracts SASS from an OpenMP executable.
    deduces SM version using llvm-objdump, then uses clang-offload-packager to extract cubins.
    """
    basename = os.path.basename(exe_path)
    
    # Check if tools exist
    if not os.path.exists(CLANG_OFFLOAD_PACKAGER):
        return
    
    # Determine SM versions from executable using llvm-objdump
    # Output format:
    # OFFLOADING IMAGE [0]:
    # kind            elf
    # arch            sm_86
    cmd_inspect = [LLVM_OBJDUMP, "--offloading", exe_path]
    try:
        result = subprocess.run(cmd_inspect, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, text=True)
        
        found_sms = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("arch"):
                parts = line.split()
                if len(parts) >= 2:
                    sm = parts[1] # e.g. sm_86
                    if sm in CUDA_SM_VERSIONS:
                        found_sms.add(sm)
        
        if not found_sms:
            # Fallback or just return if no compatible arch found
            return

        for sm in found_sms:
            cubin_name = f"{basename}-omp_{sm}.cubin"
            cubin_path = os.path.join(output_dir, cubin_name)
            
            # Command to extract specific architecture
            # clang-offload-packager <exe> --image=triple=nvptx64-nvidia-cuda,arch=<arch>,file=<cubin>
            cmd_extract = [
                CLANG_OFFLOAD_PACKAGER,
                exe_path,
                f"--image=triple=nvptx64-nvidia-cuda,arch={sm},file={cubin_path}"
            ]
            
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            
            # Check if file was created and has content
            if os.path.exists(cubin_path):
                if os.path.getsize(cubin_path) > 0:
                    sass_name = f"{basename}-omp_{sm}.sass"
                    sass_path = os.path.join(output_dir, sass_name)
                    
                    with open(sass_path, "w") as sass_file:
                        sass_file.write(f"// Disassembly for {cubin_name}\n")
                        subprocess.run(["nvdisasm", "-g", cubin_path], stdout=sass_file, stderr=subprocess.DEVNULL, check=False)
                else:
                    # remove empty file
                    os.remove(cubin_path)
    except Exception as e:
        # print(f"Error processing {basename}: {e}")
        pass

def zip_results(output_dir):
    """Zips all .sass files in the output directory."""
    zip_path = os.path.join(output_dir, "sass_files.zip")
    sass_files = [f for f in os.listdir(output_dir) if f.endswith(".sass")]
    
    if not sass_files:
        print("No SASS files to zip.")
        return

    print(f"Zipping {len(sass_files)} SASS files into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in sass_files:
            zipf.write(os.path.join(output_dir, file), arcname=file)

def main():
    # Make sure output directory exists
    os.makedirs(OUTPUT_SASS_DIR, exist_ok=True)
    
    if not os.path.exists(CLANG_OFFLOAD_PACKAGER):
        print(f"Warning: {CLANG_OFFLOAD_PACKAGER} not found. OpenMP extraction may fail.")

    print(f"Scanning for executables...")
    cuda_exes = get_executables_in_dir(CUDA_BIN_DIR)
    omp_exes = get_executables_in_dir(OMP_BIN_DIR)
    
    print(f"Found {len(cuda_exes)} CUDA executables")
    print(f"Found {len(omp_exes)} OpenMP executables")
    
    print("Extracting SASS from CUDA executables...")
    for exe in tqdm(cuda_exes, desc="CUDA Executables", unit="exe"):
        extract_cuda_sass(exe, OUTPUT_SASS_DIR)

    print("Extracting SASS from OpenMP executables...")
    for exe in tqdm(omp_exes, desc="OpenMP Executables", unit="exe"):
        extract_omp_sass(exe, OUTPUT_SASS_DIR)
             
    print("Zipping results...")
    zip_results(OUTPUT_SASS_DIR)
    print("Done.")

if __name__ == "__main__":
    main()