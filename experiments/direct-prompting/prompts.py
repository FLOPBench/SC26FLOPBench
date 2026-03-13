from typing import Dict, Optional
from pydantic import BaseModel, Field
import json

SYSTEM_PROMPT = """You are an expert GPU performance engineer and compiler developer.
Your task is to analyze a GPU kernel and accurately predict its precision operations and DRAM accesses.
Based on the provided source code, hardware specifications, compile commands, and optionally SASS/IMIX profiles, estimate the following metrics for the target kernel:
- CUDA Grid Size (gridSz) as a 3-tuple of integers
- CUDA Block Size (blockSz) as a 3-tuple of integers
- Half-precision (FP16) FLOP count
- Single-precision (FP32) FLOP count
- Double-precision (FP64) FLOP count
- DRAM bytes read
- DRAM bytes written

For each estimation, provide a step-by-step reasoning explaining how you arrived at the count before providing the final result.
"""

class KernelMetricsPrediction(BaseModel):
    gridSz_explanation: str = Field(description="Explanation for the estimated CUDA Grid Size (gridSz).")
    gridSz: tuple[int, int, int] = Field(description="Estimated CUDA Grid Size as a 3-tuple of integers (x, y, z).")
    blockSz_explanation: str = Field(description="Explanation for the estimated CUDA Block Size (blockSz).")
    blockSz: tuple[int, int, int] = Field(description="Estimated CUDA Block Size as a 3-tuple of integers (x, y, z).")
    fp32_flop_explanation: str = Field(description="Explanation for the single-precision (FP32) FLOP count estimate.")
    fp32_flop_count: int = Field(description="Estimated single-precision (FP32) FLOP count.")
    fp64_flop_explanation: str = Field(description="Explanation for the double-precision (FP64) FLOP count estimate.")
    fp64_flop_count: int = Field(description="Estimated double-precision (FP64) FLOP count.")
    fp16_flop_explanation: str = Field(description="Explanation for the half-precision (FP16) FLOP count estimate.")
    fp16_flop_count: int = Field(description="Estimated half-precision (FP16) FLOP count.")
    dram_bytes_read_explanation: str = Field(description="Explanation for the DRAM bytes read estimate.")
    dram_bytes_read_count: int = Field(description="Estimated DRAM bytes read count.")
    dram_bytes_written_explanation: str = Field(description="Explanation for the DRAM bytes written estimate.")
    dram_bytes_written_count: int = Field(description="Estimated DRAM bytes written count.")


class DirectPromptGenerator:
    def __init__(
        self,
        program_name: str,
        kernel_mangled_name: str,
        kernel_demangled_name: str,
        source_code_files: Dict[str, str],
        gpu_roofline_specs: Dict[str, str],
        compile_commands: str,
        sass_dict: Optional[Dict[str, str]] = None,
        imix_dict: Optional[Dict[str, str]] = None
    ):
        self.program_name = program_name
        self.kernel_mangled_name = kernel_mangled_name
        self.kernel_demangled_name = kernel_demangled_name
        self.source_code_files = source_code_files
        self.gpu_roofline_specs = gpu_roofline_specs
        self.compile_commands = compile_commands
        self.sass_dict = sass_dict
        self.imix_dict = imix_dict

    def generate_prompt(self) -> str:
        prompt = f"<program_name>{self.program_name}</program_name>\n"
        prompt += f"<kernel_mangled_name>{self.kernel_mangled_name}</kernel_mangled_name>\n"
        prompt += f"<kernel_demangled_name>{self.kernel_demangled_name}</kernel_demangled_name>\n"
        
        prompt += "<source_code>\n"
        for file, code in self.source_code_files.items():
            prompt += f"<file name=\"{file}\">\n{code}\n</file>\n"
        prompt += "</source_code>\n"

        prompt += "<gpu_roofline_specs>\n"
        prompt += json.dumps(self.gpu_roofline_specs, indent=2) + "\n"
        prompt += "</gpu_roofline_specs>\n"

        prompt += f"<compile_commands>\n{self.compile_commands}\n</compile_commands>\n"

        if self.sass_dict:
            prompt += "<sass>\n"
            prompt += json.dumps(self.sass_dict, indent=2) + "\n"
            prompt += "</sass>\n"
            
        if self.imix_dict:
            prompt += "<imix>\n"
            prompt += json.dumps(self.imix_dict, indent=2) + "\n"
            prompt += "</imix>\n"

        return prompt
