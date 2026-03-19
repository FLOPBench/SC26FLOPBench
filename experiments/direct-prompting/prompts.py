from typing import Dict, Optional
from pydantic import BaseModel, Field
import json

SYSTEM_PROMPT = """You are an expert GPU performance engineer and compiler developer.
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

Section 1: Grid Size and Block Size
- Start by analyzing <command_line_input_args> and perform forward constant propagation through the host code up to the first invocation of the target kernel. The command-line arguments may not be used directly in the launch site and may instead flow through intermediate variables, helper functions, or derived expressions.
- Determine the actual launch configuration used for that first invocation, including both gridSz and blockSz. This may require tracking variable assignments, function calls, macros from <compile_commands>, and any launch-configuration calculations in the host code.
- Ignore code after the first target kernel invocation, because only the first execution matters. If the kernel is launched inside a loop, only analyze the first loop iteration that invokes that kernel.
- If the target kernel is templated, only report the first instantiation that is actually executed.

Section 2: FLOP Counting
- After constant propagation up to the first launch, analyze the kernel source code to identify all floating point operations executed during that first invocation. Account for the number of launched threads, loop iterations, and warp divergence regions when scaling the total FLOP counts.
- If <sass> and <static-imix> are provided, use them to validate your FLOP estimates. If you report a particular FP16, FP32, or FP64 count, check whether the corresponding SASS instructions and IMIX data support that estimate.
- If SASS is provided, map the SASS back to the source when helpful so you can catch hidden FLOP counts or compiler transformations that are not obvious from the source alone.
- Be sure to account for hidden or easily overlooked FLOP sources, including transcendental math functions (for example sin, cos, exp, log, sqrt, __sinf, __expf), warp divergence, loop iterations, and floating point division that may compile into multiple instructions such as reciprocal approximations, multiplications, or Newton-Raphson refinement steps.
- Unary negation of a float or double (for example -x) DOES count as a floating point operation.
- Preprocessor macros from <compile_commands> may affect control flow and floating point work, so include their impact in your analysis.
- Some compilers introduce FP16 operations as an optimization even when the source code does not explicitly use half precision. If the SASS indicates FP16 instructions such as HFMA, count them in the FP16 total even if the source only appears to use FP32 or FP64 types.
- Compilers may also eliminate common subexpressions or reuse previously computed values, which can reduce the true FLOP count. Use the source and SASS together to avoid over-counting optimized-away work.
- Code comments may incorrectly state the number of FLOPs, so do not trust them; calculate the counts yourself.
- Other floating point datatypes such as FP8 should not be counted as FP16, FP32, or FP64 FLOPs.

Section 3: DRAM Read and Write Byte Counting
- Estimate DRAM bytes read and DRAM bytes written for the same first invocation of the kernel. Account for the number of launched threads, loop iterations, and warp divergence regions when scaling total memory traffic.
- Although the source may show many reads and writes, the compiler may optimize, combine, or eliminate some of them. If SASS is provided, use the global load and store instructions in SASS, especially LDG and STG, to identify the actual global-memory traffic generated by execution.
- Do not assume every global-memory instruction becomes DRAM traffic. Some LDG operations may be serviced by the L2 cache instead of DRAM, so only count the loads that correspond to DRAM accesses when estimating dram_bytes_read_count.
- For writes, only count the bytes that go beyond the L2 cache and actually reach DRAM when estimating dram_bytes_written_count. Consider the L2 cache size from <gpu_roofline_specs> together with the amount of store traffic to estimate whether written data fits in cache or spills through to DRAM.
- In many cases, if all the writes fit in the L2 cache, then dram_bytes_written_count will be zero, even if the source code shows many store instructions. Recall that the CUDA L2 cache is a write-back cache at the device-scope visibility, so it does not flush to the DRAM until evictions occur. Do not count DRAM writes that will eventually happen due to a future kernel invocation or cudamemcpy, only count the DRAM writes that actually occur during the first kernel invocation.
- Use SASS and IMIX as supporting evidence when available, and map SASS back to source when that helps explain hidden or optimized memory behavior.

For each metric, provide a short two-sentence explanation of how you arrived at the final count. Include the reasoning behind the total work or traffic performed in the first kernel invocation, along with any important assumptions or simplifications.
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
