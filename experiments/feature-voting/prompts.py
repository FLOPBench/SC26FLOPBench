import json
from typing import Dict, Optional

from pydantic import BaseModel, Field


class SystemPromptBuilder:
    def build(self) -> str:
        lines = [
            "Your task is to statically analyze the provided code and context to determine whether specific code features are present in the target kernel and its first-launch execution path.",
            "You are to inspect the provided benchmark host and target kernel code, then return a binary checklist describing whether specific code features are present for the target kernel and its first-launch path.",
            "",
            "You will be provided with the following information in XML tags:",
            "- <program_name>: The benchmark program name.",
            "- <kernel_mangled_name> and <kernel_demangled_name>: The target kernel to classify.",
            "- <command_line_input_args>: The benchmark execution arguments.",
            "- <source_code>: The relevant source code files.",
        ]

        lines.extend([
            "",
            "Return only the following boolean feature flags (True/False) in structured output:",
            "- has_branching",
            "- has_data_dependent_branching",
            "- has_flop_division",
            "- has_preprocessor_defines",
            "- has_common_float_subexpr",
            "- has_special_math_functions",
            "- calls_device_function",
            "- has_rng_input_data",
            "- reads_input_values_from_file",
            "- has_hardcoded_gridsz",
            "- has_hardcoded_blocksz",
            "",
            "Classification guidance:",
            "- Analyze the target kernel together with the host-side path that prepares and launches its first invocation.",
            "- Use source code as the primary evidence. Track constants, macros, helper calls, file-backed inputs, and launch setup through the first invocation of the target kernel.",
            "- Mark has_branching true when the kernel or directly relevant device helper path contains meaningful conditional (if, for, while, do-while) or switch-based control flow.",
            "- Mark has_data_dependent_branching true only when branch behavior depends on runtime data, thread indices, loaded values, or command-line-derived values rather than purely compile-time constants.",
            "- Mark has_flop_division true when the executed kernel code performs floating-point division or reciprocal-style floating division work at FP16, FP32, or FP64 precision.",
            "- Mark has_preprocessor_defines true when preprocessor macros or compile definitions materially affect the kernel code, launch path, or feature classification.",
            "- Mark has_common_float_subexpr true when the source visibly repeats equivalent floating-point subexpressions or derived floating-point values that could plausibly be reused or common-subexpression-eliminated.",
            "- Mark has_special_math_functions true when the kernel path uses transcendental or special math routines such as sin, cos, exp, log, pow, sqrt, rsqrt, erf, or similar library or intrinsic math functions.",
            "- Mark calls_device_function true when the target kernel directly or indirectly calls a nontrivial device function, helper routine, or inlined device-side helper beyond straight-line code in the kernel body.",
            "- Mark has_rng_input_data true when the kernel consumes random-number inputs, curand state, stochastic masks, dropout masks, random seeds, or data that is explicitly generated as random input.",
            "- Mark reads_input_values_from_file true when the host path reads runtime data from a file, dataset, stream, or similar external file-backed source before the first kernel launch and those loaded values can affect the first execution of the target kernel.",
            "- Mark has_hardcoded_gridsz true when the launch grid size for the first invocation is fixed by constants, literals, preprocessor defines, macros, or fixed OpenMP num_teams clauses rather than being derived from runtime problem size or input-dependent calculations.",
            "- Mark has_hardcoded_blocksz true when the launch block size, OpenMP thread_limit, or equivalent first-invocation thread-group size is fixed by constants, literals, preprocessor defines, or macros rather than being derived from runtime problem size or input-dependent calculations.",
        ])

        lines.extend([
            "",
            "General rules:",
            "- Focus on the first invocation only.",
            "- If evidence is mixed, choose the most defensible boolean value from the source and host path.",
            "- Do not return explanations or extra fields. Return only the structured boolean checklist.",
            "- Answer as accurately as possible based on the provided information. If uncertain, make your best judgment but do not leave any fields blank.",
            "- Always return True or False for each field, NEVER 'unknown' or null.",
            "- Be sure to focus only on the code that will actually be executed. Some programs have preprocessor defines, macros, or dead code that is not relevant to the first kernel execution path; do not let this irrelevant code mislead your classification of the executed kernel behavior.",
        ])

        return "\n".join(lines)


SYSTEM_PROMPT = SystemPromptBuilder().build()


class CodeFeatureFlags(BaseModel):
    has_branching: bool = Field(
        description="True when the CUDA kernel, OpenMP target region, or directly relevant device/helper code contains meaningful control flow such as if/else, switch, for, while, or do-while logic on the first target kernel execution path."
    )
    has_data_dependent_branching: bool = Field(
        description="True when branching in the CUDA or OpenMP first-launch path depends on runtime values such as loaded data, thread or iteration indices, reduction state, command-line-derived inputs, or other non-constant execution data."
    )
    has_flop_division: bool = Field(
        description="True when the first executed CUDA or OpenMP kernel path performs floating-point division, reciprocal-based floating division, or equivalent FP16, FP32, or FP64 division work."
    )
    has_preprocessor_defines: bool = Field(
        description="True when C, C++, CUDA, or OpenMP preprocessor macros or compile-time defines materially affect the kernel body, device/helper code, launch configuration, loop structure, or the feature classification itself."
    )
    has_common_float_subexpr: bool = Field(
        description="True when the source for the relevant CUDA or OpenMP execution path visibly repeats equivalent floating-point subexpressions or derived floating-point values that could plausibly be shared, reused, or common-subexpression-eliminated."
    )
    has_special_math_functions: bool = Field(
        description="True when the first executed CUDA or OpenMP kernel path uses transcendental or special math functions such as sin, cos, exp, log, pow, sqrt, rsqrt, erf, or comparable math-library or intrinsic routines."
    )
    calls_device_function: bool = Field(
        description="True when the target CUDA kernel or OpenMP offload path directly or indirectly calls a nontrivial device-side, target-side, or helper function beyond straight-line code in the main kernel or target region body."
    )
    has_rng_input_data: bool = Field(
        description="True when the first execution of the CUDA or OpenMP kernel consumes random-number inputs, stochastic masks, random seeds, curand or RNG state, or data that is explicitly generated to be random."
    )
    reads_input_values_from_file: bool = Field(
        description="True when the host-side setup for the first kernel launch reads values from a file, dataset, stream, or other file-backed input source at runtime and those loaded values can influence the first execution of the target CUDA or OpenMP kernel."
    )
    has_hardcoded_gridsz: bool = Field(
        description="True when the first-launch CUDA grid size or OpenMP team/grid equivalent is fixed by constants, literals, preprocessor defines, macros, fixed num_teams clauses, or other compile-time configuration instead of being derived from runtime problem size or input-dependent calculations."
    )
    has_hardcoded_blocksz: bool = Field(
        description="True when the first-launch CUDA block size or OpenMP thread_limit or team-size equivalent is fixed by constants, literals, preprocessor defines, macros, or other compile-time configuration instead of being derived from runtime problem size or input-dependent calculations."
    )


class DirectPromptGenerator:
    def __init__(
        self,
        program_name: str,
        kernel_mangled_name: str,
        kernel_demangled_name: str,
        source_code_files: Dict[str, str],
        exe_args: str,
    ):
        self.program_name = program_name
        self.kernel_mangled_name = kernel_mangled_name
        self.kernel_demangled_name = kernel_demangled_name
        self.source_code_files = source_code_files
        self.exe_args = exe_args

    def generate_system_prompt(self) -> str:
        return SystemPromptBuilder().build()

    def generate_prompt(self) -> str:
        prompt = f"<program_name>\n\t{self.program_name}\n</program_name>\n"
        prompt += f"<kernel_mangled_name>\n\t{self.kernel_mangled_name}\n</kernel_mangled_name>\n"
        prompt += f"<kernel_demangled_name>\n\t{self.kernel_demangled_name}\n</kernel_demangled_name>\n"
        prompt += f"<command_line_input_args>\n\t{self.exe_args}\n</command_line_input_args>\n\n"

        prompt += "<source_code>\n\n"
        for file, code in self.source_code_files.items():
            prompt += f"\n<file name=\"{file}\">\n{code}\n</file>\n"
        prompt += "</source_code>\n"

        return prompt
