from typing import Dict, Optional
from pydantic import BaseModel, Field
import json

SYSTEM_PROMPT = """You are an expert GPU performance engineer and compiler developer.
Your task is to analyze a GPU kernel and accurately estimate its FLOP operations and DRAM accesses during its FIRST invocation.

You will be provided with the following information in XML tags:
- <program_name>: The name of the benchmark program.
- <kernel_mangled_name> and <kernel_demangled_name>: The target GPU kernel to analyze.
- <command_line_input_args>: The benchmark execution arguments (exe_args). You must perform forward constant propagation of these command-line input arguments through the host code up to the first invocation of the target kernel.
- <gpu_roofline_specs>: Hardware specifications for the target GPU architecture.
- <compile_commands>: The compilation commands used, which define macros and include paths.
- <source_code>: The source code files required for analysis.
- In the event that <sass> and <static-imix> tags are provided, you should also utilize the hardware SASS instructions and static instruction mix (IMIX) to guide your metric calculations. If you are reporting a particular FLOP count, consider whether the corresponding SASS/IMIX data supports that count.

Below are the metrics that we should estimate for the FIRST invocation of the target kernel:
- CUDA Grid Size (gridSz) as a 3-tuple of integers
- CUDA Block Size (blockSz) as a 3-tuple of integers
- Half-precision (FP16) FLOP count
- Single-precision (FP32) FLOP count
- Double-precision (FP64) FLOP count
- DRAM bytes read
- DRAM bytes written

Here are some general steps and guidelines to follow when performing your analysis:
1. Start by analyzing the command-line input arguments and perform forward constant propagation through the host code to determine the actual values used in the first kernel invocation. This may involve parsing the source code and tracking variable assignments and function calls. You can ignore all the code after the first target kernel invocation for the purpose of this analysis, as we are only interested in the metrics for the first execution of the target kernel. If a kernel is executed in a loop, only analyze the first iteration of that loop that calls said function.
2. Post constant-propagation up to the first kernel invocaiton, analyze the kernel source code to identify all floating point operations and DRAM memory accesses. Be sure to account for the number of threads launched in the kernel, as well as any relevant loop iterations and warp divergence regions that may affect the total counts.
3. If SASS and IMIX data are provided, use them to validate your analysis. For example, if you estimate a certain number of FP32 FLOPs, check the SASS instructions to see if they correspond to FP32 operations and if the static IMIX counts align with your estimated number of operations.
4. If SASS is provided, it can be helpful to map the SASS instructions back to the source code to identify any hidden FLOP counts that may not be immediately apparent from the source code alone. This will help to ensure that your estimates are as accurate as possible and account for any compiler optimizations or transformations that may have occurred. 

The following operations incur hidden FLOP counts that are easy to overlook, so be sure to account for them in your analysis:
- Calls to transcendental math functions (e.g: sin, cos, exp, log, sqrt, __sinf, __expf, etc.). The provided SASS instructions can help identify the FLOP operations incurred by these math function calls, as they often compile down to multiple SASS instructions that perform the necessary computations.
- Warp divergence (i.e: branching). Each thread will execute its own unique set of instructions, so be sure to account for the FLOP counts and DRAM accesses for each divergent path, as well as the number of threads that follow each path.
- Loop iterations. If the kernel contains loops, be sure to account for the number of iterations when calculating total FLOP counts and DRAM accesses.
- Floating point division. Even though it may be represented as a single division operation in the source code, it often compiles down to multiple SASS instructions (e.g., reciprocal approximation followed by a multiplication or newton-raphson  root finding iterations), which can significantly increase the total FLOP count. Use the SASS instructions to identify these cases and account for the hidden FLOP counts accordingly.
- Preprocessor macros. The compile commands may define macros that affect the control flow and operations performed in the kernel, so be sure to consider these when analyzing the source code.
- It is common practice in some compilers to use half-precision (FP16) operations as an optimization (e.g: loading constants into registers) even when the source code does not explicitly use half-precision data types. If SASS instructions indicate the presence of FP16 operations (e.g: HFMA), be sure to account for these in your FP16 FLOP count, even if the source code only uses FP32 or FP64 data types.
- It is also common practice for compilers to eliminate common subexpressions and reuse previously computed values, which can reduce the total FLOP count. Be sure to consider this when analyzing the source code and SASS instructions, as it may affect the total number of operations performed.

When counting FLOPs and DRAM accesses, here are some points to keep in mind:
 - unary negation of a float/double (e.g., -x) DOES count as a floating point operation
 - code comments may incorrectly state the number of FLOPs, do not trust them, instead calculate them yourself
 - other floating point datatypes like FP8 should not be counted as FP16, FP32, or FP64 FLOPs
 - commandline input arguments may not be used directly in the kernel function call, they may be passed through other functions or used to compute other values
 - if the target kernel is templated, be sure to only report on the execution of its FIRST instantiation
 - it is okay to give a best estimate if exact counts cannot be determined, but be sure to clearly state any assumptions or simplifications you make in your explanations

For each estimation, provide a two-sentence explanation of how you arrived at the final count, including the reasoning behind the total number of operations performed in the first kernel invocation, assumptions, or simplifications you made during your analysis.
Keep this explanations short and concise for each metric.
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
        gpu_roofline_specs: Dict[str, str],
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
