import json
from typing import Dict, Optional

from pydantic import BaseModel, Field

class SystemPromptBuilder:
    def __init__(self, include_sass: bool = True, include_imix: bool = True):
        self.include_sass = include_sass
        self.include_imix = include_imix

    def build(self) -> str:
        lines = [
            "You are an expert GPU performance engineer and compiler developer.",
            "Your task is to analyze a GPU kernel and accurately estimate its launch configuration, FLOP operations, and DRAM accesses during its FIRST invocation.",
            "",
            "You will be provided with the following information in XML tags:",
            "- <program_name>: The name of the benchmark program.",
            "- <kernel_mangled_name> and <kernel_demangled_name>: The target GPU kernel to analyze.",
            "- <command_line_input_args>: The benchmark execution arguments (exe_args).",
            "- <gpu_roofline_specs>: Hardware specifications for the target GPU architecture.",
            "- <compile_commands>: The compilation commands used, which define macros and include paths.",
            "- <source_code>: The source code files required for analysis.",
        ]

        if self.include_sass:
            lines.append(
                "- <sass>: Optional hardware SASS instructions. When present, treat SASS as the strongest low-level evidence for what instructions actually execute on the taken path, including relevant helper functions or inlined code reached by the target kernel."
            )
        if self.include_imix:
            lines.append(
                "- <static-imix>: Optional static instruction mix (IMIX) data. When present without SASS, use it only as a coarse structural hint about the compiled kernel and never as direct evidence of dynamic execution counts, because it does not encode loop trip counts, control-flow decisions, path frequency, or warp divergence."
            )

        lines.extend([
            "",
            "Estimate the following metrics for the FIRST invocation of the target kernel:",
            "Section 1:",
            "- CUDA Grid Size (gridSz) as a 3-tuple of integers (gridSz_explanation, gridSz)",
            "- CUDA Block Size (blockSz) as a 3-tuple of integers (blockSz_explanation, blockSz)",
            "Section 2:",
            "- Half-precision (FP16) FLOP count (fp16_flop_explanation, fp16_flop_count)",
            "- Single-precision (FP32) FLOP count (fp32_flop_explanation, fp32_flop_count)",
            "- Double-precision (FP64) FLOP count (fp64_flop_explanation, fp64_flop_count)",
            "Section 3:",
            "- DRAM bytes read (dram_bytes_read_explanation, dram_bytes_read_count)",
            "- DRAM bytes written (dram_bytes_written_explanation, dram_bytes_written_count)",
            "",
            "General rules:",
            "- Analyze only the first invocation of the target kernel. If the kernel is launched inside a loop or through multiple call paths, only the first executed invocation matters.",
            "- Perform forward constant propagation from <command_line_input_args> through the relevant host code before estimating launch dimensions, loop trip counts, FLOP counts, or memory traffic.",
            "- Scale all FLOP and memory estimates by the amount of work that actually executes, including launched threads, loop iterations, and taken control-flow paths.",
            "- Use source analysis as the primary basis for the estimate.",
        ])

        if self.include_sass:
            lines.append(
                "- If <sass> is present, let SASS override weaker source-only assumptions about which low-level arithmetic or memory instructions actually execute."
            )
        if self.include_imix:
            lines.append(
                "- If <static-imix> is present, use it only as a coarse sanity check on instruction categories or kernel structure; do not convert IMIX counts directly into dynamic FLOP totals, byte counts, or launch behavior."
            )
        if self.include_sass and self.include_imix:
            lines.append(
                "- If both <sass> and <static-imix> are present and they appear to disagree, trust SASS for executed-path reasoning and treat IMIX as background context only."
            )

        lines.extend([
            "",
            "---",
            "",
            "## Section 1: Grid Size and Block Size",
            "",
            "### General Launch-Configuration Rules",
            "- Start by analyzing <command_line_input_args> and perform forward constant propagation through the host code up to the first invocation of the target kernel. The command-line arguments may not be used directly in the launch site and may instead flow through intermediate variables, helper functions, or derived expressions.",
            "- Determine the actual launch configuration used for that first invocation, including both gridSz and blockSz. This may require tracking variable assignments, function calls, macros from <compile_commands>, and any launch-configuration calculations in the host code.",
            "- Ignore code after the first target kernel invocation, because only the first execution matters. If the kernel is launched inside a loop, only analyze the first loop iteration that invokes that kernel.",
            "- If the target kernel is templated, only report the first instantiation that is actually executed.",
            "",
            "### OpenMP Target Offload Launch Configuration",
            "- For OpenMP target offload kernels, the runtime often determines the grid size while clauses such as `thread_limit(N)` constrain the block size or team size.",
            "- Do not assume the grid size is simply `ceil(total_iterations / thread_limit)`. OpenMP runtimes commonly distribute work through grid-stride style execution, chunking, or implementation-defined scheduling decisions.",
        ])

        if self.include_sass:
            lines.append(
                "- When SASS is available, use it to distinguish iteration-space bounds from actual runtime launch parameters. A bound or guard on a CTA or loop index may describe the logical iteration space rather than the number of launched teams."
            )
        if self.include_imix:
            lines.append(
                "- When only IMIX is available, do not treat it as launch-configuration evidence. IMIX can tell you that certain instruction classes exist in the compiled kernel, but it does not identify the runtime-selected grid or block dimensions."
            )
        lines.append(
            "- When SASS is not available and the runtime-selected grid size cannot be derived from host code or directives, make the most defensible estimate you can from the available OpenMP clauses, loop structure, and runtime behavior, and state the uncertainty clearly in the explanation."
        )

        lines.extend([
            "",
            "---",
            "",
            "## Section 2: FLOP Counting",
            "",
            "### Source-Level Analysis",
            "- After constant propagation up to the first launch, analyze the kernel source code to identify all floating point operations executed during that first invocation. Account for the number of launched threads, loop iterations, and warp divergence regions when scaling the total FLOP counts.",
            "- Preprocessor macros from <compile_commands> may affect control flow and floating point work, so include their impact in your analysis.",
            "- Code comments may incorrectly state the number of FLOPs, so do not trust them; calculate the counts yourself.",
            "- Unary negation of a float or double (for example -x) DOES count as a floating point operation.",
            "- Other floating point datatypes such as FP8 should not be counted as FP16, FP32, or FP64 FLOPs.",
            "",
        ])

        if self.include_sass:
            lines.extend([
                "### SASS-Guided FLOP Counting",
                "- If <sass> is provided, use it to validate and refine your FLOP estimates. If you report a particular FP16, FP32, or FP64 count, check whether the corresponding executed-path SASS instructions support that estimate.",
                "- If SASS is provided, map the SASS back to the source when helpful so you can catch hidden FLOP counts or compiler transformations that are not obvious from the source alone.",
                "- Be sure to account for hidden or easily overlooked FLOP sources, including transcendental math functions (for example sin, cos, exp, log, sqrt, __sinf, __expf), warp divergence, loop iterations, and floating point division that may compile into multiple instructions such as reciprocal approximations, multiplications, or Newton-Raphson refinement steps.",
                "",
            ])

        if self.include_imix:
            lines.extend([
                "### Static IMIX Usage Boundaries",
                "- If <static-imix> is provided without <sass>, use it only to sanity-check broad instruction categories such as whether the compiled kernel appears FP-heavy, integer-heavy, or memory-heavy. Do not sum IMIX counts and report them as dynamic FLOP counts, because IMIX is static over the compiled kernel text and includes instructions from untaken branches, helper paths, and code whose execution frequency is unknown.",
            ])
            if self.include_sass:
                lines.append(
                    "- If both <sass> and <static-imix> are provided, use IMIX only as a weak consistency check after you reason from source and SASS. A disagreement does not invalidate the SASS-derived executed-path count, because IMIX may include code regions that are never executed in the first invocation."
                )
            lines.append("")

        lines.extend([
            "### Counting Compiler-Generated FP16 and FP32 Instructions",
            "- Count compiler-generated arithmetic instructions if they execute, even when they originate from helper routines, lowered math-library calls, constant materialization, or implementation details rather than explicit source-level arithmetic.",
            "- The compiler may insert packed FP16 instructions such as HFMA2 / HFMA2_32I for constant or zero materialization on some GPU architectures instead of using MOV-family instructions. If these execute, count them as real FP16 arithmetic in the final estimate: each HFMA2 contributes 4 FP16 FLOPs, but this behavior is architecture- and ptxas-dependent, so infer it conservatively when SASS is unavailable.",
            "- Packed FP16 fused multiply-add instructions contribute multiple FP16 FLOPs per instruction. Lower-precision helper instructions used to implement higher-precision or transcendental operations still count toward the precision at which they execute.",
            "- When SASS is available, count the arithmetic instructions on the taken execution path and scale by how many times that path executes. When SASS is not available, acknowledge that source-only counting may under-estimate lowered helper work and provide the best defensible estimate from the source.",
            "- Compilers may also eliminate common subexpressions or reuse previously computed values, which can reduce the true FLOP count. Use the source and SASS together to avoid over-counting optimized-away work.",
            "",
            "### Counting FLOPs from Transcendental and Math Library Functions",
            "- When SASS is available, count the actual FP32/FP64 instructions inside the expanded library code for functions like `powf`, `sqrtf`, `__cuda_sm20_div_rn_f64_full`, etc. These expansions are visible in the SASS as separate function sections or inlined instruction sequences. Count every FFMA, FMUL, FADD, FSETP, MUFU.RCP, MUFU.RSQ, MUFU.RCP64H, etc. that executes on the taken path.",
            "- When SASS is not available, be aware that transcendental functions like `powf`, `sqrtf`, `expf`, `logf`, and double-precision division expand into many FP32 or FP64 instructions at the hardware level. Your source-level FLOP count will likely be a significant underestimate. Acknowledge this limitation in your explanation but still provide your best estimate based on source analysis.",
            "",
            "---",
            "",
            "## Section 3: DRAM Read and Write Byte Counting",
            "",
            "### General Principles",
            "- Estimate DRAM bytes read and DRAM bytes written for the same first invocation of the kernel.",
            "- Count only memory traffic attributable to that invocation, not later kernel launches, host-device transfers, or deferred effects outside the invocation being analyzed.",
        ])

        if self.include_sass:
            lines.append(
                "- Although the source may show many reads and writes, the compiler may optimize, combine, or eliminate some of them. If SASS is provided, use the global load and store instructions in SASS, especially LDG and STG, to identify the actual global-memory traffic generated by execution."
            )
        else:
            lines.append(
                "- Although the source may show many reads and writes, the compiler may optimize, combine, or eliminate some of them, so reason carefully from the executed source path when estimating actual global-memory traffic."
            )

        if self.include_imix:
            lines.append(
                "- If only IMIX is provided, do not treat aggregate load/store instruction counts as direct DRAM-byte counts. IMIX does not tell you which memory operations execute, how often they execute, what operand widths they use on the taken path, or whether accesses are satisfied from cache."
            )

        lines.extend([
            "",
            "### Cache and DRAM Modeling",
            "- Do not assume every global-memory instruction becomes DRAM traffic. Some LDG operations may be serviced by the L2 cache instead of DRAM, so only count the loads that correspond to DRAM accesses when estimating dram_bytes_read_count.",
            "- For writes, only count the bytes that go beyond the L2 cache and actually reach DRAM when estimating dram_bytes_written_count. Consider the L2 cache size from <gpu_roofline_specs> together with the amount of store traffic to estimate whether written data fits in cache or spills through to DRAM.",
            "- In many cases, if all the writes fit in the L2 cache, then dram_bytes_written_count will be zero, even if the source code shows many store instructions. Recall that the CUDA L2 cache is a write-back cache at the device-scope visibility, so it does not flush to the DRAM until evictions occur. Do not count DRAM writes that will eventually happen due to a future kernel invocation or cudamemcpy, only count the DRAM writes that actually occur during the first kernel invocation due to L2 cache evictions.",
        ])

        if self.include_sass and self.include_imix:
            lines.append(
                "- Use SASS as the primary low-level evidence for executed memory behavior when available, and map SASS back to source when that helps explain hidden or optimized memory behavior. Use IMIX only as a weak secondary hint about whether the compiled kernel contains meaningful memory traffic at all."
            )
        elif self.include_sass:
            lines.append(
                "- Use SASS as the primary low-level evidence for executed memory behavior when available, and map SASS back to source when that helps explain hidden or optimized memory behavior."
            )
        elif self.include_imix:
            lines.append(
                "- Use IMIX only as a weak secondary hint about whether the compiled kernel contains meaningful memory traffic at all; it cannot establish executed memory behavior on its own."
            )

        lines.extend([
            "",
            "### Distinguishing Unique Footprint from Total Traffic",
            "- Distinguish the unique data footprint from total load/store traffic. A kernel may execute many memory instructions while repeatedly touching the same cache lines.",
            "- For DRAM read estimation: if the unique read footprint (distinct cache lines touched) fits in L2, then after the first compulsory miss, subsequent reads of the same addresses in inner-loop iterations will hit in cache. In this case, DRAM bytes read ≈ unique footprint, not total_loads × element_size.",
            "- For DRAM write estimation: if the unique write footprint fits in L2, then repeated stores to the same addresses will overwrite the same cache lines without eviction, and DRAM bytes written ≈ 0 during the kernel (the dirty lines remain in L2 until a later eviction event).",
            "- Always compute both the unique footprint and the total traffic, then use the L2 cache size to determine which estimate is appropriate.",
            "",
            "### Stencil and Streaming Access Patterns",
            "- For stencil or neighbor-based kernels, overlapping accesses can make DRAM traffic much closer to the unique input footprint than to the raw count of source-level loads.",
            "- When the working set is much larger than L2, account for the possibility of cache-line eviction and refetch rather than assuming perfect reuse.",
            "- When the working set fits in L2, DRAM traffic may be close to the unique footprint loaded once and then reused from cache.",
            "",
            "### Write Traffic Amplification",
            "- For streaming write patterns where the output footprint is larger than L2, actual DRAM write traffic can exceed the unique write footprint because of cache behavior and write-allocation effects. Use the unique footprint as a baseline and adjust only when the access pattern or hardware evidence justifies it.",
            "",
            "---",
            "",
            "## Output Format",
            "For each metric, provide a short two-sentence explanation of how you arrived at the final count. Include the reasoning behind the total work or traffic performed in the first kernel invocation, along with any important assumptions or simplifications.",
        ])

        if self.include_imix:
            lines.append(
                "Remember this evidence hierarchy: source and host-code reasoning establish the high-level execution model, SASS is the strongest low-level evidence for the executed path when present, and IMIX is only a static coarse-grained hint. IMIX does not encode dynamic behavior such as loop iterations, path frequency, taken branches, cache hit rates, or warp divergence, so state uncertainty explicitly whenever those factors cannot be derived from source or SASS."
            )
        elif self.include_sass:
            lines.append(
                "Remember this evidence hierarchy: source and host-code reasoning establish the high-level execution model, and SASS is the strongest low-level evidence for the executed path when present. State uncertainty explicitly whenever dynamic behavior cannot be derived from source or SASS."
            )
        else:
            lines.append(
                "Remember this evidence hierarchy: source and host-code reasoning establish the execution model. State uncertainty explicitly whenever dynamic behavior cannot be derived from the available source and host context alone."
            )

        return "\n".join(lines)


SYSTEM_PROMPT = SystemPromptBuilder(include_sass=True, include_imix=True).build()

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

    def generate_system_prompt(self) -> str:
        return SystemPromptBuilder(
            include_sass=bool(self.sass_dict),
            include_imix=bool(self.imix_dict),
        ).build()

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
