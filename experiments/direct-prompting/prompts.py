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
- In the event that <sass> and <imix> tags are provided, you should also utilize the hardware SASS instructions and dynamic execution counts (IMIX) to guide your metric calculations. If you are reporting a particular FLOP count, consider whether the corresponding SASS/IMIX data supports that count.

Based on this analysis, first deduce the grid and block sizes of this first invocation, and then estimate the following metrics for the target kernel:
- CUDA Grid Size (gridSz) as a 3-tuple of integers
- CUDA Block Size (blockSz) as a 3-tuple of integers
- Half-precision (FP16) FLOP count
- Single-precision (FP32) FLOP count
- Double-precision (FP64) FLOP count
- DRAM bytes read
- DRAM bytes written

When counting FLOPs and DRAM accesses, be sure to remember the following:
 - unary negation of a float/double (e.g., -x) DOES count as a floating point operation.
 - code comments may incorrectly state the number of FLOPs, do not trust them, instead calculate them yourself
 - other floating point datatypes like FP8 should not be counted as FP16, FP32, or FP64 FLOPs
 - commandline input arguments may not be used directly in the kernel function call, they may be passed through other functions or used to compute other values
 - if the target kernel is templated, be sure to only report on the execution of its FIRST instantiation
 - it is okay to give a best estimate if exact counts cannot be determined, but be sure to clearly state any assumptions or simplifications you make in your explanations

For each estimation, provide a two-sentence explanation of how you arrived at the final count for each FLOP precision, including the reasoning behind the total number of operations performed in the first kernel invocation, assumptions, or simplifications you made during your analysis.
Keep this explanations short and concise for each metric.
"""

class KernelMetricsPrediction(BaseModel):
    gridSz_explanation: str = Field(description="Brief explanation for the estimated CUDA Grid Size (gridSz).")
    gridSz: list[int] = Field(description="Estimated CUDA Grid Size as a list of 3 integers [x, y, z].")
    
    blockSz_explanation: str = Field(description="Brief explanation for the estimated CUDA Block Size (blockSz).")
    blockSz: list[int] = Field(description="Estimated CUDA Block Size as a list of 3 integers [x, y, z].")
    
    fp32_flop_explanation: str = Field(description="Brief explanation of how the single-precision (FP32) floating point operations count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")
    fp32_flop_count: int = Field(description="Total number of single-precision (FP32) floating point operations performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    fp64_flop_explanation: str = Field(description="Brief explanation of how the double-precision (FP64) floating point operations count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")
    fp64_flop_count: int = Field(description="Total number of double-precision (FP64) floating point operations performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    fp16_flop_explanation: str = Field(description="Brief explanation of how the half-precision (FP16) floating point operations count was calculated. This should include the reasoning behind the number of operations performed in the kernel, including any relevant loop iterations and warp divergence region executions.")
    fp16_flop_count: int = Field(description="Total number of half-precision (FP16) floating point operations performed by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    dram_bytes_read_explanation: str = Field(description="Brief explanation of how the DRAM bytes read count was calculated. This should include the reasoning behind the number of memory read accesses, including memory coalescing, relevant loop iterations, and warp divergence region executions.")
    dram_bytes_read_count: int = Field(description="Total number of DRAM bytes read by the kernel. Accounting for the number of threads, loop iterations, and warp divergence region executions.")
    
    dram_bytes_written_explanation: str = Field(description="Brief explanation of how the DRAM bytes written count was calculated. This should include the reasoning behind the number of memory write accesses, including memory coalescing, relevant loop iterations, and warp divergence region executions.")
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

    def generate_prompt(self) -> str:
        prompt = f"<program_name>{self.program_name}</program_name>\n"
        prompt += f"<kernel_mangled_name>{self.kernel_mangled_name}</kernel_mangled_name>\n"
        prompt += f"<kernel_demangled_name>{self.kernel_demangled_name}</kernel_demangled_name>\n"
        prompt += f"<command_line_input_args>{self.exe_args}</command_line_input_args>\n"

        prompt += "<gpu_roofline_specs>\n"
        prompt += json.dumps(self.gpu_roofline_specs, indent=2) + "\n"
        prompt += "</gpu_roofline_specs>\n"

        prompt += "<compile_commands>\n"
        for cmd_entry in self.compile_commands:
            filename = cmd_entry["file"]
            command = cmd_entry["command"]
            prompt += f"<compile_command filename=\"{filename}\">\n{command}\n</compile_command>\n"
        prompt += "</compile_commands>\n"

        if self.sass_dict:
            prompt += "<sass>\n"
            for section_name, sass_code in self.sass_dict.items():
                prompt += f"<section name=\"{section_name}\">\n{sass_code}\n</section>\n"
            prompt += "</sass>\n"
            
        if self.imix_dict:
            prompt += "<imix>\n"
            prompt += json.dumps(self.imix_dict, indent=2) + "\n"
            prompt += "</imix>\n"

        prompt += "<source_code>\n"
        for file, code in self.source_code_files.items():
            prompt += f"<file name=\"{file}\">\n{code}\n</file>\n"
        prompt += "</source_code>\n"

        return prompt
