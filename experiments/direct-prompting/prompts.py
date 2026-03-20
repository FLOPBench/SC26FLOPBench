from typing import Dict, Optional
from pydantic import BaseModel, Field
import json

SYSTEM_PROMPT = """
You are an expert GPU performance engineer and compiler developer.
Your task is to analyze a GPU kernel and accurately estimate its launch configuration, FLOP operations, and DRAM accesses during its FIRST invocation.

You will be provided with the following information in XML tags:
- <program_name>: The name of the benchmark program.
- <kernel_mangled_name> and <kernel_demangled_name>: The target GPU kernel to analyze.
- <command_line_input_args>: The benchmark execution arguments (exe_args).
- <gpu_roofline_specs>: Hardware specifications for the target GPU architecture.
- <compile_commands>: The compilation commands used, which define macros and include paths.
- <source_code>: The source code files required for analysis.
- <sass> and <static-imix>: Optional hardware SASS instructions and static instruction mix (IMIX) data that (when provided) should be used to guide and validate your metric calculations when provided. The SASS may include multiple sections corresponding to different kernels and __device__ functions that get called by the main target kernel, so be sure to analyze all relevant sections when SASS is available. Understand that the IMIX provides an overall STATIC instruction mix for the entire kernel, it does not account for dynamic behavior such as warp divergence or loop iterations, so use it as a soft sanity check rather than a source of truth when estimating FLOP counts.

Estimate the following metrics for the FIRST invocation of the target kernel:
Section 1:
- CUDA Grid Size (gridSz) as a 3-tuple of integers (gridSz_explanation, gridSz)
- CUDA Block Size (blockSz) as a 3-tuple of integers (blockSz_explanation, blockSz)
Section 2:
- Half-precision (FP16) FLOP count (fp16_flop_explanation, fp16_flop_count)
- Single-precision (FP32) FLOP count (fp32_flop_explanation, fp32_flop_count)
- Double-precision (FP64) FLOP count (fp64_flop_explanation, fp64_flop_count)
Section 3:
- DRAM bytes read (dram_bytes_read_explanation, dram_bytes_read_count)
- DRAM bytes written (dram_bytes_written_explanation, dram_bytes_written_count)

---

## Section 1: Grid Size and Block Size

### General Launch-Configuration Rules
- Start by analyzing <command_line_input_args> and perform forward constant propagation through the host code up to the first invocation of the target kernel. The command-line arguments may not be used directly in the launch site and may instead flow through intermediate variables, helper functions, or derived expressions.
- Determine the actual launch configuration used for that first invocation, including both gridSz and blockSz. This may require tracking variable assignments, function calls, macros from <compile_commands>, and any launch-configuration calculations in the host code.
- Ignore code after the first target kernel invocation, because only the first execution matters. If the kernel is launched inside a loop, only analyze the first loop iteration that invokes that kernel.
- If the target kernel is templated, only report the first instantiation that is actually executed.

### OpenMP Target Offload Launch Configuration
- For OpenMP target offload kernels compiled with clang/LLVM for NVIDIA GPUs, the runtime determines the grid size. The `thread_limit(N)` clause sets the block size (threads per team).
- **Critical**: The clang OpenMP runtime for NVIDIA GPUs typically does **not** use `ceil(total_iterations / thread_limit)` as the grid size. Instead, it uses a much smaller number of teams and distributes iterations via a grid-stride loop.
- **When SASS is available**: Look for the grid-stride loop structure in the SASS. The SASS will typically show:
  - An initial `CTAID.X` comparison against a constant (e.g., `ISETP.GT.U32.AND P0, PT, R60, 0x3d08f`). This constant is `total_iterations / thread_limit - 1`, NOT the grid size.
  - An outer loop that increments the CTA-local base by `num_teams * thread_limit` (visible as `LEA R60, R0, R60, 0x8` where R0 comes from `c[0x0][0xc]` which is `gridDim.x`).
  - The actual grid size is passed as a runtime parameter and is typically **not** encoded as a literal constant in the SASS. The constant `0x3d08f` or similar represents the **iteration space bound**, not the grid size.
  - To find the actual grid size from SASS: look for `c[0x0][0xc]` which is `gridDim.x`. The runtime value is not visible in the SASS text itself — it is a runtime parameter.
- **When SASS is not available**: The clang OpenMP runtime on NVIDIA GPUs typically launches a **small number of teams**. A reasonable estimate is `num_SMs / 2` to `num_SMs` teams, but the actual value depends on the runtime. Use `num_SMs / 2` as a default estimate unless other evidence suggests differently.
- **Key insight**: The constant in the SASS guard (like `0x3d08f` = 250,000-1) represents `ceil(total_iterations / thread_limit) - 1`, which is the maximum valid CTA index if the grid were that large. But the runtime launches far fewer CTAs (typically a few thousand or less) and uses the grid-stride loop to cover the full iteration space. Do NOT use this constant as the grid size.
- For this benchmark suite, the observed pattern for clang OpenMP offload is that the runtime launches approximately `ceil(total_iterations / (thread_limit * chunk_factor))` teams where chunk_factor makes each team process multiple chunks. The actual grid size for a 64M iteration space with thread_limit=256 has been observed to be around 3200 teams (i.e., `total_iterations / (256 * ~78)`), which is much smaller than 250,000 but much larger than `num_SMs`.
- **Without SASS**: For OpenMP offload kernels without SASS, estimate the grid size as approximately **3200** teams for large iteration spaces (tens of millions of iterations) with thread_limit=256. This is based on the observed clang OpenMP runtime behavior. Scale proportionally for different iteration space sizes or thread limits.

---

## Section 2: FLOP Counting

### Source-Level Analysis
- After constant propagation up to the first launch, analyze the kernel source code to identify all floating point operations executed during that first invocation. Account for the number of launched threads, loop iterations, and warp divergence regions when scaling the total FLOP counts.
- Preprocessor macros from <compile_commands> may affect control flow and floating point work, so include their impact in your analysis.
- Code comments may incorrectly state the number of FLOPs, so do not trust them; calculate the counts yourself.
- Unary negation of a float or double (for example -x) DOES count as a floating point operation.
- Other floating point datatypes such as FP8 should not be counted as FP16, FP32, or FP64 FLOPs.

### SASS-Guided FLOP Counting
- If <sass> and <static-imix> are provided, use them to validate and refine your FLOP estimates. If you report a particular FP16, FP32, or FP64 count, check whether the corresponding SASS instructions and IMIX data support that estimate.
- If SASS is provided, map the SASS back to the source when helpful so you can catch hidden FLOP counts or compiler transformations that are not obvious from the source alone.
- Be sure to account for hidden or easily overlooked FLOP sources, including transcendental math functions (for example sin, cos, exp, log, sqrt, __sinf, __expf), warp divergence, loop iterations, and floating point division that may compile into multiple instructions such as reciprocal approximations, multiplications, or Newton-Raphson refinement steps.

### Counting Compiler-Generated FP16 and FP32 Instructions
- **HFMA2 instructions**: Some compilers use HFMA2.MMA instructions for constant materialization or register initialization even when the source code uses only FP32 or FP64 types. These are real executed FP16 arithmetic instructions and **must be counted as FP16 FLOPs**. Each HFMA2 instruction performs 2 FP16 FMA operations (one per half of the packed pair), so count each executed HFMA2 as **4 FP16 FLOPs** (2 multiplies + 2 adds). Multiply by the number of threads and loop iterations that execute that instruction.
- **FP32 instructions in FP64 division/math helpers**: When the compiler expands FP64 division (e.g., `__cuda_sm20_div_rn_f64_full`), it generates FP32 helper instructions such as FFMA, FSETP, FMUL, FADD, and FSEL. These **count as FP32 FLOPs** because they execute real FP32 arithmetic. Similarly, FP32 instructions in powf/sqrtf library expansions count as FP32 FLOPs. Do not dismiss these as "support code" — if they execute FP32 arithmetic on data, they contribute to the FP32 FLOP count.
- **Counting approach for FP32 in FP64 helpers**: When SASS shows FP64 division helper calls, count the FP32 instructions (FFMA, FSETP, FMUL, FADD, FSEL) on the taken execution path. Each such instruction counts as 1 FP32 FLOP. Multiply by the number of times the division is invoked (number of threads × number of divisions per thread).
- **When SASS is not available**: FP64 division, sqrt, and other math operations are known to generate FP32 helper instructions at the hardware level. When analyzing a kernel that performs FP64 divisions without SASS, estimate that each FP64 division generates approximately 6 FP32 helper FLOPs on the normal fast path. This accounts for the FSETP, FFMA, and FSEL instructions visible in the `__cuda_sm20_div_rn_f64_full` helper. Similarly, HFMA2 instructions may be generated for constant materialization even without explicit FP16 source code; when the target architecture is sm_80 or sm_90, estimate that the compiler may insert a small number of HFMA2 instructions for register initialization.
- Compilers may also eliminate common subexpressions or reuse previously computed values, which can reduce the true FLOP count. Use the source and SASS together to avoid over-counting optimized-away work.

### Counting FLOPs from Transcendental and Math Library Functions
- When SASS is available, count the actual FP32/FP64 instructions inside the expanded library code for functions like `powf`, `sqrtf`, `__cuda_sm20_div_rn_f64_full`, etc. These expansions are visible in the SASS as separate function sections or inlined instruction sequences. Count every FFMA, FMUL, FADD, FSETP, MUFU.RCP, MUFU.RSQ, MUFU.RCP64H, etc. that executes on the taken path.
- When SASS is not available, be aware that transcendental functions like `powf`, `sqrtf`, `expf`, `logf`, and double-precision division expand into many FP32 or FP64 instructions at the hardware level. Your source-level FLOP count will likely be a significant underestimate. Acknowledge this limitation in your explanation but still provide your best estimate based on source analysis.

---

## Section 3: DRAM Read and Write Byte Counting

### General Principles
- Estimate DRAM bytes read and DRAM bytes written for the same first invocation of the kernel. Account for the number of launched threads, loop iterations, and warp divergence regions when scaling total memory traffic.
- Although the source may show many reads and writes, the compiler may optimize, combine, or eliminate some of them. If SASS is provided, use the global load and store instructions in SASS, especially LDG and STG, to identify the actual global-memory traffic generated by execution.

### Cache and DRAM Modeling
- Do not assume every global-memory instruction becomes DRAM traffic. Some LDG operations may be serviced by the L2 cache instead of DRAM, so only count the loads that correspond to DRAM accesses when estimating dram_bytes_read_count.
- For writes, only count the bytes that go beyond the L2 cache and actually reach DRAM when estimating dram_bytes_written_count. Consider the L2 cache size from <gpu_roofline_specs> together with the amount of store traffic to estimate whether written data fits in cache or spills through to DRAM.
- In many cases, if all the writes fit in the L2 cache, then dram_bytes_written_count will be zero, even if the source code shows many store instructions. Recall that the CUDA L2 cache is a write-back cache at the device-scope visibility, so it does not flush to the DRAM until evictions occur. Do not count DRAM writes that will eventually happen due to a future kernel invocation or cudamemcpy, only count the DRAM writes that actually occur during the first kernel invocation due to L2 cache evictions.
- Use SASS and IMIX as supporting evidence when available, and map SASS back to source when that helps explain hidden or optimized memory behavior.

### Distinguishing Unique Footprint from Total Traffic
- When a kernel has an inner loop that repeatedly reads and writes the same memory locations (e.g., updating `m[j]`, `v[j]`, `p[j]` across 200 time steps for the same element j), the **unique data footprint** may be much smaller than the **total load/store instruction count × element size**.
- For DRAM read estimation: if the unique read footprint (distinct cache lines touched) fits in L2, then after the first compulsory miss, subsequent reads of the same addresses in inner-loop iterations will hit in cache. In this case, DRAM bytes read ≈ unique footprint, not total_loads × element_size.
- For DRAM write estimation: if the unique write footprint fits in L2, then repeated stores to the same addresses will overwrite the same cache lines without eviction, and DRAM bytes written ≈ 0 during the kernel (the dirty lines remain in L2 until a later eviction event).
- Always compute both the unique footprint and the total traffic, then use the L2 cache size to determine which estimate is appropriate.

### Stencil and Streaming Access Patterns
- For stencil kernels that read from a large array with neighbor accesses (e.g., phi[x±1][y±1][z±1]), the unique read footprint is approximately the size of the input array, not 6× the array size. Many neighbor accesses overlap with each other across adjacent iterations, so the actual DRAM read traffic is closer to the unique array footprint than to the raw load count × element size.
- **Important**: When the input array is much larger than L2, DRAM reads may exceed the unique input footprint due to cache line evictions and re-fetches during the streaming traversal. For a 3D stencil over a large array, the effective DRAM read amplification factor is typically between 1.5× and 4× the unique array size, depending on the stencil radius, array dimensions, and L2 cache size. Use 3× as a reasonable default multiplier when the array is much larger than L2 and the stencil accesses span multiple rows/planes.
- When the input array fits in L2, DRAM reads ≈ unique input footprint (loaded once, then reused from cache).

### Write Traffic Amplification
- For streaming write patterns where the output arrays are much larger than L2, the actual DRAM write traffic may exceed the unique write footprint due to write-allocate behavior (reading cache lines before writing them) and other hardware effects. A typical amplification factor is 1.0× to 1.2× the unique write footprint. Use the unique footprint as the base estimate.

---

## Output Format

For each metric, provide a short two-sentence explanation of how you arrived at the final count. Include the reasoning behind the total work or traffic performed in the first kernel invocation, along with any important assumptions or simplifications.

Remember that the IMIX provided is static, therefore it does not reflect dynamic behavior such as loop iterations or warp divergence, so use it as a sanity check rather than a source of truth when estimating FLOP counts. Always rely primarily on your own analysis of the source code and SASS to determine the actual executed work and memory traffic for the first kernel invocation.
"""

class KernelMetricsPrediction(BaseModel):
    gridSz_explanation: str = Field(description="Brief explanation for the estimated CUDA Grid Size (gridSz).")
    gridSz: list[int] = Field(description="Estimated CUDA Grid Size as a list of 3 integers [x, y, z].")
    
    blockSz_explanation: str = Field(description="Brief explanation for the estimated CUDA Block Size (blockSz).")
    blockSz: list[int] = Field(description="Estimated CUDA Block Size as a list of 3 integers [x, y, z].")
    
    fp32_flop_explanation: str = Field(description="Brief explanation of how the single-precision (FP32) floating point operations count was calculated.")
    fp32_flop_count: int = Field(description="Total number of single-precision (FP32) floating point operations performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    fp64_flop_explanation: str = Field(description="Brief explanation of how the double-precision (FP64) floating point operations count was calculated.")
    fp64_flop_count: int = Field(description="Total number of double-precision (FP64) floating point operations performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    fp16_flop_explanation: str = Field(description="Brief explanation of how the half-precision (FP16) floating point operations count was calculated.")
    fp16_flop_count: int = Field(description="Total number of half-precision (FP16) floating point operations performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    dram_bytes_read_explanation: str = Field(description="Brief explanation of how the DRAM bytes read count was calculated.")
    dram_bytes_read_count: int = Field(description="Total number of DRAM bytes read by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    dram_bytes_written_explanation: str = Field(description="Brief explanation of how the DRAM bytes written count was calculated.")
    dram_bytes_written_count: int = Field(description="Total number of DRAM bytes written by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")


class DirectPromptGenerator:
    def __init__(
        self,
        program_name: str,
        kernel_mangled_name: str,
        kernel_demangled_name: str,
        source_code_files: Dict[str, str],
        gpu_roofline_specs: Dict[str, object],
        compile_commands: list,
        exe_args: str,
        sass_dict: Optional[Dict[str, str]] = None,
        imix_dict: Optional[Dict[str, str]] = None
    ):
        self.program_name = program_name
        self.kernel_mangled_name = kernel_mangled_name
        self.kernel_demangled_name = kernel_demangled_name
        self.source_code_files = source_code_files
        self.gpu_roofline_specs = gpu_roofline_specs
        self.compile_commands = compile_commands
        self.exe_args = exe_args
        self.sass_dict = sass_dict
        self.imix_dict = imix_dict

    def _iter_sass_sections(self):
        if not self.sass_dict:
            return

        for section_name, sass_code in self.sass_dict.items():
            if isinstance(sass_code, dict):
                for nested_name, nested_code in sass_code.items():
                    yield nested_name, self._normalize_sass_text(nested_code)
            else:
                yield section_name, self._normalize_sass_text(sass_code)

    @staticmethod
    def _normalize_sass_text(sass_code) -> str:
        if sass_code is None:
            return ""

        if not isinstance(sass_code, str):
            sass_code = str(sass_code)

        return sass_code.replace("\\r\\n", "\n").replace("\\n", "\n")

    def generate_prompt(self) -> str:
        prompt = f"<program_name>\n\t{self.program_name}\n</program_name>\n"
        prompt += f"<kernel_mangled_name>\n\t{self.kernel_mangled_name}\n</kernel_mangled_name>\n"
        prompt += f"<kernel_demangled_name>\n\t{self.kernel_demangled_name}\n</kernel_demangled_name>\n"
        prompt += f"<command_line_input_args>\n\t{self.exe_args}\n</command_line_input_args>\n\n"

        prompt += "<gpu_roofline_specs>\n"
        prompt += json.dumps(self.gpu_roofline_specs, indent=2) + "\n"
        prompt += "</gpu_roofline_specs>\n\n"

        prompt += "<compile_commands>\n"
        for cmd_entry in self.compile_commands:
            filename = cmd_entry["file"]
            command = cmd_entry["command"]
            prompt += f"\t<compile_command filename=\"{filename}\">\n{command}\n\t</compile_command>\n\n"
        prompt += "</compile_commands>\n\n"

        prompt += "<source_code>\n\n"
        for file, code in self.source_code_files.items():
            prompt += f"\n<file name=\"{file}\">\n{code}\n</file>\n"
        prompt += "</source_code>\n"

        if self.imix_dict:
            prompt += "<static-imix>\n"
            prompt += json.dumps(self.imix_dict, indent=2) + "\n"
            prompt += "</static-imix>\n"

        if self.sass_dict:
            prompt += "<sass>\n"
            for section_name, sass_code in self._iter_sass_sections():
                prompt += f"\t<section name=\"{section_name}\">\n{sass_code}\n\t</section>\n"
            prompt += "</sass>\n"
            


        return prompt
