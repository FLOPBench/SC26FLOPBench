import json
import os
from pathlib import Path
from tqdm import tqdm

def parse_d_file(d_filepath):
    """
    Parses a single .d dependency file and returns a set of source file paths.
    Only paths inside HeCBench/src and with valid extensions are kept.
    """
    valid_extensions = {".hpp", ".h", ".cpp", ".c", ".cu", ".cc"}
    paths = set()
    
    with open(d_filepath, "r") as f:
        content = f.read()
    
    # Simple split by whitespace is generally safe if paths don't contain spaces.
    # We replace newline and backslash to easily split.
    content = content.replace("\\\n", " ")
    content = content.replace("\n", " ")
    
    tokens = content.split()
    
    for token in tokens:
        # Ignore target definitions like "target.o: "
        if token.endswith(":"):
            continue
            
        path = Path(token).resolve()
        
        # Check if the path exists, is a file, and has a correct extension
        if path.is_file() and path.suffix in valid_extensions:
            # We want only the files within HeCBench/src
            if "/HeCBench/src/" in str(path):
                # Also exclude Makefile or CMakeLists.txt (handled by suffix check implicitly)
                paths.add(str(path))
                
    return paths

def get_benchmark_files(benchmark_name, build_dir):
    """
    Finds all .d files for a given benchmark and aggregates the source paths.
    """
    benchmark_d_dir = Path(build_dir) / "src" / benchmark_name / "CMakeFiles" / f"{benchmark_name}.dir"
    if not benchmark_d_dir.exists():
        return set()
    
    all_paths = set()
    for d_file in benchmark_d_dir.glob("*.d"):
        all_paths.update(parse_d_file(d_file))
        
    return all_paths

def scrape_sources(build_dir, output_json="scraped_sources.json"):
    """
    Scrapes source code for all cuda and omp benchmarks and saves to JSON.
    """
    build_src_dir = Path(build_dir) / "src"
    if not build_src_dir.exists():
        print(f"Build directory {build_src_dir} not found.")
        return

    # Find all benchmark directories that end with -cuda or -omp
    benchmarks = []
    for d in build_src_dir.iterdir():
        if d.is_dir() and (d.name.endswith("-cuda") or d.name.endswith("-omp")):
            benchmarks.append(d.name)
            
    print(f"Found {len(benchmarks)} benchmarks to scrape.")
    
    data = {}
    
    for benchmark in tqdm(benchmarks, desc="Scraping Benchmarks"):
        source_paths = get_benchmark_files(benchmark, build_dir)
        
        if not source_paths:
            continue
            
        data[benchmark] = {}
        
        for sp in source_paths:
            try:
                with open(sp, "r", encoding="utf-8", errors="replace") as f:
                    data[benchmark][sp] = f.read()
            except Exception as e:
                print(f"\nError reading {sp} for {benchmark}: {e}")
                
    # Write to final JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f"\nSuccessfully wrote scraped sources to {output_json} (included {len(data)} benchmarks).")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    
    BUILD_DIR = str(root_dir / "build")
    OUTPUT_JSON = str(script_dir / "scraped_sources.json")
    
    scrape_sources(BUILD_DIR, OUTPUT_JSON)

