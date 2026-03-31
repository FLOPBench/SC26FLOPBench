import json
from typing import Dict, Optional

from pydantic import BaseModel, Field


SYSTEM_PROMPT = """Your task is to statically analyze the provided code and context to determine whether specific code features are present in the target kernel definition and its directly relevant device-side execution path.
You are to inspect the provided benchmark code, identify the target kernel definition, and return a binary checklist describing whether specific code features are present in that target kernel itself.

You will be provided with the following information in XML tags:
- <program_name>: The benchmark program name.
- <kernel_mangled_name> and <kernel_demangled_name>: The target kernel to classify.
- <command_line_input_args>: The benchmark execution arguments.
- <source_code>: The relevant source code files.

Return only the following boolean feature flags (True/False) in structured output:
- has_branching
- has_data_dependent_branching
- has_flop_division
- has_preprocessor_defines
- has_common_float_subexpr
- has_special_math_functions
- calls_device_function
- has_rng_input_data
- reads_input_values_from_file
- has_hardcoded_gridsz
- has_hardcoded_blocksz

Classification guidance:
- Use host-side code only to identify the first invocation of the named target kernel, how that target kernel's launch parameters are set, and how runtime inputs flow into that target kernel. Do not count host-only control flow or host-only arithmetic toward kernel feature flags.
- Base kernel-behavior flags on the target kernel definition itself plus any downstream `__device__` functions, inlined device helpers, or OpenMP parallel or target regions that are reached from within that target kernel's execution path. Do not use unrelated host logic as evidence for kernel branching, data-dependent branching, or floating-point division.
- Use source code as the primary evidence. Track constants, macros, helper calls, file-backed inputs, and launch setup only insofar as they affect the target kernel definition or its first invocation. Do not classify based on kernels other than the named target kernel.
- Treat command-line arguments provided in `<command_line_input_args>` as known concrete values for this analysis, not as unknown runtime-dependent values. If source variables are derived entirely from literals, compile-time macros, and these known command-line values, then those derived variables should also be treated as known constants for classification purposes.
- Treat preprocessor macros, compile-time defines, and any values derived entirely from them as known constants. Conditional statements that depend only on such compile-time-known values are not data-dependent.
- Mark has_branching true only when the target kernel body or downstream `__device__` functions or OpenMP regions reached from that target kernel contain meaningful conditional (if, for, while, do-while) or switch-based control flow. Ignore host-side branching for this flag.
- Mark has_data_dependent_branching true only when branch behavior inside the target kernel or downstream `__device__` functions or OpenMP regions reached from that target kernel depends on true runtime-varying data such as loaded memory values, thread indices, reduction state, RNG-driven values, or quantities derived from unknown runtime inputs. Do not mark it true for branches that depend only on literals, preprocessor macros, compile-time constants, provided command-line argument values, or variables derived entirely from those known values. Ignore host-side branching for this flag.
- Mark has_flop_division true only when the target kernel or downstream `__device__` functions or OpenMP regions reached from that target kernel perform floating-point division or reciprocal-style floating division work at FP16, FP32, or FP64 precision. Ignore host-side floating-point division for this flag.
- Mark has_preprocessor_defines true when preprocessor macros or compile definitions materially affect the kernel code, launch path, or feature classification.
- Mark has_common_float_subexpr true when the source for the target kernel body or downstream `__device__` functions or OpenMP regions reached from that target kernel visibly repeats equivalent floating-point subexpressions or derived floating-point values that could plausibly be reused or common-subexpression-eliminated.
- Mark has_special_math_functions true when the target kernel or downstream `__device__` functions or OpenMP regions reached from that target kernel use transcendental or special math routines such as sin, cos, exp, log, pow, sqrt, rsqrt, erf, or similar library or intrinsic math functions.
- Mark calls_device_function true when the target kernel directly or indirectly calls a nontrivial `__device__` function, helper routine, or nested target-side helper beyond straight-line code in the kernel body.
- Mark has_rng_input_data true when the target kernel or downstream `__device__` functions or OpenMP regions reached from that target kernel consume random-number inputs, curand state, stochastic masks, dropout masks, random seeds, or data that is explicitly generated as random input.
- Mark reads_input_values_from_file true when the host path reads runtime data from a file, dataset, stream, or similar external file-backed source before the first invocation of the named target kernel and those loaded values can affect that target kernel's first execution.
- Mark has_hardcoded_gridsz true when the launch grid size for the first invocation of the named target kernel is fixed by constants, literals, preprocessor defines, macros, or fixed OpenMP num_teams clauses rather than being derived from runtime problem size or input-dependent calculations.
- Mark has_hardcoded_blocksz true when the launch block size, OpenMP thread_limit, or equivalent first-invocation thread-group size for the named target kernel is fixed by constants, literals, preprocessor defines, or macros rather than being derived from runtime problem size or input-dependent calculations.

General rules:
- Focus only on the first invocation of the named target kernel, not the first kernel launch that appears anywhere in the program.
- If evidence is mixed, choose the most defensible boolean value while keeping kernel-behavior flags scoped to the target kernel definition and any downstream `__device__` functions or OpenMP regions that execute as part of that target kernel's device-side path.
- Do not return explanations or extra fields. Return only the structured boolean checklist.
- Answer as accurately as possible based on the provided information. If uncertain, make your best judgment but do not leave any fields blank.
- Always return True or False for each field, NEVER 'unknown' or null.
- Be sure to focus only on the code that will actually be executed for the first invocation of the named target kernel. Some programs have preprocessor defines, macros, dead code, or other kernels that are not relevant to that target-kernel execution path; do not let this irrelevant code mislead your classification.
- Keep in mind that a kernel may have data-dependent branching that is not immediately obvious from the kernel code alone. For example, the kernel may read values from global memory, consume random inputs, or receive truly runtime-varying data that influences device-side control flow. Track how data and inputs flow into the target kernel, but distinguish unknown runtime-varying inputs from known compile-time constants and the concrete command-line arguments provided in `<command_line_input_args>`. Do not treat host-side control flow itself as kernel branching.
"""


class CodeFeatureFlags(BaseModel):
    has_branching: bool = Field(
        description="True when the target CUDA kernel, OpenMP target region, or downstream `__device__` functions or OpenMP regions reached from that target contain meaningful control flow such as if/else, switch, for, while, or do-while logic. Host-side control flow does not count for this flag."
    )
    has_data_dependent_branching: bool = Field(
        description="True when branching inside the target CUDA kernel, OpenMP target region, or downstream `__device__` functions or OpenMP regions reached from that target depends on runtime values such as loaded data, thread or iteration indices, reduction state, command-line-derived inputs, or other non-constant execution data. Host-side branching does not count for this flag."
    )
    has_flop_division: bool = Field(
        description="True when the target CUDA kernel, OpenMP target region, or downstream `__device__` functions or OpenMP regions reached from that target perform floating-point division, reciprocal-based floating division, or equivalent FP16, FP32, or FP64 division work. Host-side floating-point division does not count for this flag."
    )
    has_preprocessor_defines: bool = Field(
        description="True when C, C++, CUDA, or OpenMP preprocessor macros or compile-time defines materially affect the kernel body, device/helper code, launch configuration, loop structure, or the feature classification itself."
    )
    has_common_float_subexpr: bool = Field(
        description="True when the source for the target kernel execution path, including downstream `__device__` functions or OpenMP regions reached from that target, visibly repeats equivalent floating-point subexpressions or derived floating-point values that could plausibly be shared, reused, or common-subexpression-eliminated."
    )
    has_special_math_functions: bool = Field(
        description="True when the first execution path of the named target CUDA or OpenMP kernel, including downstream `__device__` functions or OpenMP regions reached from that target, uses transcendental or special math functions such as sin, cos, exp, log, pow, sqrt, rsqrt, erf, or comparable math-library or intrinsic routines."
    )
    calls_device_function: bool = Field(
        description="True when the target CUDA kernel or OpenMP offload path directly or indirectly calls a nontrivial device-side, target-side, or helper function beyond straight-line code in the main kernel or target region body. This includes downstream `__device__` functions and nested target-side helpers reached from the named target kernel."
    )
    has_rng_input_data: bool = Field(
        description="True when the first execution of the named target CUDA or OpenMP kernel, including downstream `__device__` functions or OpenMP regions reached from that target, consumes random-number inputs, stochastic masks, random seeds, curand or RNG state, or data that is explicitly generated to be random."
    )
    reads_input_values_from_file: bool = Field(
        description="True when the host-side setup for the first invocation of the named target CUDA or OpenMP kernel reads values from a file, dataset, stream, or other file-backed input source at runtime and those loaded values can influence that target kernel's first execution."
    )
    has_hardcoded_gridsz: bool = Field(
        description="True when the CUDA grid size or OpenMP team/grid equivalent for the first invocation of the named target kernel is fixed by constants, literals, preprocessor defines, macros, fixed num_teams clauses, or other compile-time configuration instead of being derived from runtime problem size or input-dependent calculations."
    )
    has_hardcoded_blocksz: bool = Field(
        description="True when the CUDA block size or OpenMP thread_limit or team-size equivalent for the first invocation of the named target kernel is fixed by constants, literals, preprocessor defines, macros, or other compile-time configuration instead of being derived from runtime problem size or input-dependent calculations."
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
        return SYSTEM_PROMPT

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
