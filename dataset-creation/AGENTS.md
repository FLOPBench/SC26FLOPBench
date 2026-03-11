# Dataset Creation Pipelines for LLM Intake

This directory contains the scripts related to generating the structured dataset of source codes from the HeCBench suite, primarily to provide reliable strings to LLMs for analysis. 

## Source Scraping

The script `scrape-sources.py` performs an organized collection of all source files for each benchmark. 

### Process:
1. It iterates over the generated target directories within the compilation output: `build/src/*-[cuda|omp]/`.
2. Inside each benchmark's `CMakeFiles/<benchmark>.dir/` folder, it searches for `.d` dependency files generated natively by the compiler.
3. It parses these dependency files to collect a fully complete map of which sources and implicitly included headers were touched to build the specific benchmark.
4. It filters the identified paths to strictly keep files existing within `HeCBench/src/` with valid extensions: `.c`, `.cc`, `.cpp`, `.hpp`, `.h`, `.cu`. 
5. It outputs a nested JSON dictionary structured exactly as `{ benchmark_name : { filepath : source_code_string } }`.

### Purpose of Nested JSON
By exporting this as a nested dictionary instead of a pre-formatted string (such as wrapping texts with `<file>` tags right in the script), it offers downstream applications or agents complete flexibility over formatting. The person or script handling this JSON can iterate over the file paths and compile them arbitrarily to fit their target LLM.

### Output
The default output path is: `dataset-creation/scraped_sources.json`.

## Testing 

We enforce tests strictly inside `unit-tests/test_source_scraping_functions.py` to ensure dataset reliability:
- Entrypoint verification: at least one file across the aggregated fileset holds a `main(` declaration.
- Keyword verification: CUDA benchmarks contain `__global__` or `__device__`. OpenMP benchmarks contain `#pragma omp target` or `#pragma omp parallel`.

Run the tests simply with `pytest unit-tests/test_source_scraping_functions.py -v`.
