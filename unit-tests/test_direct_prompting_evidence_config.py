import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


prompts = _load_module("direct_prompting_prompts", "experiments/direct-prompting/prompts.py")
run_queries = _load_module("direct_prompting_run_queries", "experiments/direct-prompting/run_queries.py")
visualize_results = _load_module("direct_prompting_visualize_results", "experiments/direct-prompting/visualize_results.py")


@pytest.mark.parametrize(
    ("use_sass", "use_imix", "expected_config"),
    [
        (False, False, "nosass_noimix"),
        (True, False, "sass_noimix"),
        (False, True, "nosass_imix"),
        (True, True, "sass_imix"),
    ],
)
def test_evidence_thread_part_encodes_all_configurations(use_sass, use_imix, expected_config):
    assert run_queries._evidence_thread_part(use_sass, use_imix) == expected_config


@pytest.mark.parametrize(
    ("thread_id", "expected_use_sass", "expected_use_imix", "expected_label"),
    [
        (
            "prog_kernel_A100_openai_gpt-5.4_nosass_noimix_trial0",
            False,
            False,
            "No SASS / No IMIX",
        ),
        (
            "prog_kernel_A100_openai_gpt-5.4_sass_noimix_trial0",
            True,
            False,
            "SASS Only",
        ),
        (
            "prog_kernel_A100_openai_gpt-5.4_nosass_imix_trial0",
            False,
            True,
            "IMIX Only",
        ),
        (
            "prog_kernel_A100_openai_gpt-5.4_sass_imix_trial0",
            True,
            True,
            "SASS + IMIX",
        ),
        (
            "prog_kernel_A100_openai_gpt-5.4_withsass_trial0",
            True,
            True,
            "SASS + IMIX",
        ),
        (
            "prog_kernel_A100_openai_gpt-5.4_nosass_trial0",
            False,
            False,
            "No SASS / No IMIX",
        ),
    ],
)
def test_thread_metadata_parses_new_and_legacy_configurations(
    thread_id,
    expected_use_sass,
    expected_use_imix,
    expected_label,
):
    metadata = visualize_results._thread_metadata(thread_id)

    assert metadata["use_sass"] is expected_use_sass
    assert metadata["use_imix"] is expected_use_imix
    assert metadata["evidence_configuration"] == expected_label
    assert metadata["gpu"] == "A100"
    assert metadata["trial"] == 0


@pytest.mark.parametrize(
    ("sass_dict", "imix_dict", "expect_sass", "expect_imix"),
    [
        (None, None, False, False),
        ({"kernel": "SASS BODY"}, None, True, False),
        (None, {"FADD": 4}, False, True),
        ({"kernel": "SASS BODY"}, {"FADD": 4}, True, True),
    ],
)
def test_direct_prompt_generator_includes_optional_sections_independently(
    sass_dict,
    imix_dict,
    expect_sass,
    expect_imix,
):
    generator = prompts.DirectPromptGenerator(
        program_name="demo-program",
        kernel_mangled_name="_Z6kernelv",
        kernel_demangled_name="kernel()",
        source_code_files={"kernel.cu": "__global__ void kernel() {}"},
        gpu_roofline_specs={"gpu_target": "H100", "arch": "sm_90"},
        compile_commands=[{"file": "kernel.cu", "command": "clang++ kernel.cu"}],
        exe_args="--demo",
        sass_dict=sass_dict,
        imix_dict=imix_dict,
    )

    prompt = generator.generate_prompt()

    assert ("<sass>" in prompt) is expect_sass
    assert ("<static-imix>" in prompt) is expect_imix


def test_system_prompt_distinguishes_sass_from_imix_roles():
    system_prompt = prompts.SYSTEM_PROMPT

    assert "If <sass> is present, let SASS override weaker source-only assumptions" in system_prompt
    assert "If <static-imix> is present, use it only as a coarse sanity check" in system_prompt
    assert "If both <sass> and <static-imix> are present and they appear to disagree, trust SASS" in system_prompt
    assert "When only IMIX is available, do not treat it as launch-configuration evidence" in system_prompt
    assert "If only IMIX is provided, do not treat aggregate load/store instruction counts as direct DRAM-byte counts" in system_prompt


def test_generate_system_prompt_omits_imix_guidance_when_imix_is_absent():
    generator = prompts.DirectPromptGenerator(
        program_name="demo-program",
        kernel_mangled_name="_Z6kernelv",
        kernel_demangled_name="kernel()",
        source_code_files={"kernel.cu": "__global__ void kernel() {}"},
        gpu_roofline_specs={"gpu_target": "H100", "arch": "sm_90"},
        compile_commands=[{"file": "kernel.cu", "command": "clang++ kernel.cu"}],
        exe_args="--demo",
        sass_dict={"kernel": "SASS BODY"},
        imix_dict=None,
    )

    system_prompt = generator.generate_system_prompt()

    assert "<static-imix>" not in system_prompt
    assert "Static IMIX Usage Boundaries" not in system_prompt
    assert "When only IMIX is available, do not treat it as launch-configuration evidence" not in system_prompt
    assert "If only IMIX is provided, do not treat aggregate load/store instruction counts as direct DRAM-byte counts" not in system_prompt
    assert "If <sass> is present, let SASS override weaker source-only assumptions" in system_prompt


def test_generate_system_prompt_includes_imix_guidance_when_imix_is_present():
    generator = prompts.DirectPromptGenerator(
        program_name="demo-program",
        kernel_mangled_name="_Z6kernelv",
        kernel_demangled_name="kernel()",
        source_code_files={"kernel.cu": "__global__ void kernel() {}"},
        gpu_roofline_specs={"gpu_target": "H100", "arch": "sm_90"},
        compile_commands=[{"file": "kernel.cu", "command": "clang++ kernel.cu"}],
        exe_args="--demo",
        sass_dict=None,
        imix_dict={"FADD": 4},
    )

    system_prompt = generator.generate_system_prompt()

    assert "<static-imix>" in system_prompt
    assert "Static IMIX Usage Boundaries" in system_prompt
    assert "When only IMIX is available, do not treat it as launch-configuration evidence" in system_prompt
    assert "If only IMIX is provided, do not treat aggregate load/store instruction counts as direct DRAM-byte counts" in system_prompt


def test_run_queries_parser_defaults_prompt_printing_to_false():
    parser = run_queries.build_arg_parser()

    args = parser.parse_args([])

    assert args.verbose is False
    assert args.printPrompts is False


def test_run_queries_parser_accepts_print_prompts_flag_independently():
    parser = run_queries.build_arg_parser()

    args = parser.parse_args(["--printPrompts"])

    assert args.printPrompts is True
    assert args.verbose is False
