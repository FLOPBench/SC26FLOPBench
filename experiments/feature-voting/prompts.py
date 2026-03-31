from typing import Dict

from pydantic import BaseModel, Field


SYSTEM_PROMPT = """Your task is to statically analyze the provided code and context to determine whether specific code features are present in the first invocation of the named target kernel definition and its directly relevant device-side execution path.
You are to inspect the provided benchmark code, identify the target kernel definition, its first invocation, and return a binary checklist describing whether specific code features are present in that target kernel itself.

You will be provided with the following information in XML tags:
- <program_name>: The benchmark program name.
- <kernel_mangled_name> and <kernel_demangled_name>: The target kernel to classify.
- <command_line_input_args>: The benchmark execution arguments.
- <source_code>: The relevant source code files.

Return only the following boolean feature flags (True/False) in the structured output tool call of CodeFeatureFlags, without any additional explanation or fields. These flags are listed below:
Kernel-execution-path flags:
- has_branching
- has_data_dependent_branching
- has_flop_division
- uses_preprocessor_defines
- has_common_float_subexpr
- has_loop_invariant_flops
- has_special_math_functions
- calls_device_function

Host-side setup and launch-analysis flags:
- has_rng_input_data
- reads_input_values_from_file
- has_constant_propagatable_gridsz
- has_constant_propagatable_blocksz

General rules:
- Focus only on the first invocation of the named target kernel, NOT the first kernel launch that appears anywhere in the program.
- If evidence is mixed, choose the most defensible boolean value while keeping kernel-behavior flags scoped to the target kernel definition and any downstream `__device__` functions or OpenMP regions that execute as part of that target kernel's device-side path.
- Do not return any explanations or extra fields. 
- Only return the structured boolean checklist.
- Answer as accurately as possible based on the provided information. If uncertain, make your best judgment, do not leave any fields blank.
- Always return True or False for each field, NEVER 'unknown' or null.
- Be sure to focus only on the code that will actually be executed for the first invocation of the named target kernel. Some programs have preprocessor defines, macros, dead code, or other kernels that are not relevant to that target-kernel execution path; do not let this irrelevant code mislead your classification.
"""
# - Keep in mind that a kernel may have data-dependent branching that is not immediately obvious from the kernel code alone. For example, the kernel may read values from global memory, consume random inputs, or receive truly runtime-varying data that influences device-side control flow. Track how data and inputs flow into the target kernel, but distinguish unknown runtime-varying inputs from known compile-time constants and the concrete command-line arguments provided in `<command_line_input_args>` that can be propagated throughout the code. Do not treat host-side control flow itself as kernel branching.


class CodeFeatureFlags(BaseModel):
    has_branching: bool = Field(
        description="Mark has_branching True only when the target kernel body or downstream `__device__` functions or OpenMP regions reached from that target kernel contain meaningful conditional (if, for, while, do-while) or switch-based control flow. Ignore host-side branching for this flag, only focus on the target kernel and its device-side execution path."
    )
    has_data_dependent_branching: bool = Field(
        description="Mark has_data_dependent_branching True only when branch behavior inside the target kernel or downstream `__device__` functions or OpenMP regions reached from that target kernel depends on true runtime-varying data such as loaded memory values, thread indices, reduction state, RNG-driven values, or quantities derived from unknown runtime inputs. Do not mark it True for branches that depend only on literals, preprocessor macros, compile-time constants, provided command-line argument values, or variables derived entirely from those known values. Ignore host-side branching for this flag, only focus on the target kernel and its device-side execution path."
    )
    has_flop_division: bool = Field(
        description="Mark has_flop_division True only when the target kernel or downstream `__device__` functions or OpenMP regions reached from that target kernel perform floating-point division or reciprocal-style floating division work at FP16, FP32, or FP64 precision. Ignore host-side floating-point division for this flag."
    )
    uses_preprocessor_defines: bool = Field(
        description="Mark uses_preprocessor_defines True when the target kernel execution path uses preprocessor macros or compile definitions that materially affect how that kernel executes, such as changing code paths, loop structure, helper behavior, launch-relevant constants, or other execution-relevant logic."
    )
    has_common_float_subexpr: bool = Field(
        description="Mark has_common_float_subexpr True when the source for the target kernel body or downstream `__device__` functions or OpenMP regions reached from that target kernel visibly repeats equivalent floating-point expressions. Only mark this True when noticing floating-point subexpressions that would result in the same numerical value across multiple uses and could plausibly be shared, reused, and thus eliminated by the compiler to avoid redundant subexpression computations."
    )
    has_loop_invariant_flops: bool = Field(
        description="Mark has_loop_invariant_flops True when a loop in the target kernel body or downstream `__device__` functions or OpenMP regions reached from that target kernel contains floating-point computations, conversions, or derived floating-point values that are invariant with respect to the loop iteration and could plausibly be hoisted out of the loop to eliminate redundant FLOPs. This involves identifying computations that do not depend on the loop index or any data that changes across iterations, and that could be computed once before the loop rather than redundantly within each iteration."
    )
    has_special_math_functions: bool = Field(
        description="Mark has_special_math_functions True when the target kernel or downstream `__device__` functions or OpenMP regions reached from that target kernel use transcendental or special math routines such as sin, cos, exp, log, pow, sqrt, rsqrt, erf, or similar library or intrinsic math functions. Calls to these routines do not by themselves count as calls_device_function unless such functions are explicitly defined as device/helper functions in the provided source code."
    )
    calls_device_function: bool = Field(
        description="Mark calls_device_function True when the target kernel directly or indirectly calls a nontrivial source-defined `__device__` function, helper routine, or nested target-side helper beyond straight-line code in the kernel body. Do not set this flag based only on calls to standard library math functions or intrinsics such as sin, cos, exp, log, pow, sqrt, rsqrt, or erf unless the provided source code explicitly defines them as device/helper functions."
    )
    has_rng_input_data: bool = Field(
        description="Mark has_rng_input_data True when any input reaching the first invocation of the named target kernel comes from random-number generation, curand state, stochastic masks, random seeds, dropout masks, or data that is explicitly generated to be random, even if that random data is first produced on the host before being passed to the kernel."
    )
    reads_input_values_from_file: bool = Field(
        description="Mark reads_input_values_from_file True when the host path reads runtime data from a file, dataset, stream, or similar external file-backed source before the first invocation of the named target kernel and those loaded values can affect that target kernel's first execution."
    )
    has_constant_propagatable_gridsz: bool = Field(
        description="Mark has_constant_propagatable_gridsz True when static analysis can determine the launch grid size for the first invocation of the named target kernel after constant propagation through literals, preprocessor defines, macros, compile-time constants, and the concrete `<command_line_input_args>` values. This includes values derived from the command-line input arguments vector argv, as well as fixed OpenMP num_teams clauses or equivalent launch/team expressions that become known after propagation."
    )
    has_constant_propagatable_blocksz: bool = Field(
        description="Mark has_constant_propagatable_blocksz True when static analysis can determine the launch block size, OpenMP thread_limit, or equivalent first-invocation thread-group size for the named target kernel after constant propagation through literals, preprocessor defines, macros, compile-time constants, and the concrete `<command_line_input_args>` values. This includes values derived from the command-line input arguments vector argv."
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
