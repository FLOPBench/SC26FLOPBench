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
db_manager = _load_module("direct_prompting_db_manager", "experiments/direct-prompting/db_manager.py")


def _parser_without_db():
    parser = db_manager.CheckpointDBParser.__new__(db_manager.CheckpointDBParser)
    parser.db_uri = "postgresql://unused"
    parser.conn = None
    parser.serde = None
    return parser


def _checkpoint(thread_id, checkpoint_id, parent_checkpoint_id, channel_values):
    return {
        "thread_id": thread_id,
        "checkpoint_ns": "",
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": parent_checkpoint_id,
        "checkpoint": {
            "channel_values": channel_values,
            "channel_versions": {},
        },
    }


def _completed_channel_values(**overrides):
    channel_values = {
        "program_name": "demo-cuda",
        "kernel_mangled_name": "_Z4demov",
        "kernel_demangled_name": "demo()",
        "llm_model_name": "openai/gpt-5.4",
        "query_time": 1.25,
        "cost_usd": 0.015,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "expected_fp16": 0,
        "expected_fp32": 128,
        "expected_fp64": 0,
        "expected_read_bytes": 64,
        "expected_write_bytes": 32,
        "expected_block_size": [8, 4, 1],
        "expected_grid_size": [2, 2, 1],
        "prediction": {
            "blockSz": [8, 4, 1],
            "gridSz": [2, 2, 1],
        },
        "metrics_diff": {
            "fp16": 0,
            "fp32": 16,
            "fp64": 0,
            "read_bytes": 8,
            "write_bytes": -4,
        },
        "metrics_pct_diff": {
            "fp16": 0,
            "fp32": 12.5,
            "fp64": 0,
            "read_bytes": 12.5,
            "write_bytes": 12.5,
        },
    }
    channel_values.update(overrides)
    return channel_values


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


def test_fetch_tail_checkpoint_for_thread_remains_strict_for_disconnected_history():
    parser = _parser_without_db()
    thread_id = "demo_kernel_H100_openai_gpt-5.4_sass_imix_trial0"
    checkpoints = [
        _checkpoint(thread_id, "root", None, {"program_name": "demo-cuda"}),
        _checkpoint(thread_id, "tail", "root", _completed_channel_values()),
        _checkpoint(thread_id, "orphan", None, {"program_name": "demo-cuda"}),
    ]

    with pytest.raises(ValueError, match="expected exactly one root checkpoint"):
        parser._tail_checkpoint_from_records(checkpoints, thread_id)


def test_fetch_tail_checkpoints_by_thread_skips_invalid_threads_in_tolerant_mode():
    parser = _parser_without_db()
    valid_thread_id = "valid_kernel_H100_openai_gpt-5.4_sass_imix_trial0"
    invalid_thread_id = "invalid_kernel_H100_openai_gpt-5.4_sass_imix_trial0"
    checkpoints = [
        _checkpoint(valid_thread_id, "root-valid", None, {"program_name": "demo-cuda"}),
        _checkpoint(valid_thread_id, "tail-valid", "root-valid", _completed_channel_values()),
        _checkpoint(invalid_thread_id, "root-invalid", None, {"program_name": "demo-cuda"}),
        _checkpoint(invalid_thread_id, "tail-invalid", "root-invalid", {"program_name": "demo-cuda"}),
        _checkpoint(invalid_thread_id, "orphan-invalid", None, {"program_name": "demo-cuda"}),
    ]

    result = parser.fetch_tail_checkpoints_by_thread(checkpoints=checkpoints, tolerate_errors=True)

    assert set(result["tails"]) == {valid_thread_id}
    assert len(result["invalid_threads"]) == 1
    assert result["invalid_threads"][0]["thread_id"] == invalid_thread_id
    assert result["invalid_threads"][0]["kind"] == "invalid_root_count"


def test_database_dataframe_uses_valid_tails_and_ignores_skipped_invalid_threads():
    parser = _parser_without_db()
    valid_thread_id = "valid_kernel_H100_openai_gpt-5.4_sass_imix_trial0"
    invalid_thread_id = "invalid_kernel_H100_openai_gpt-5.4_sass_imix_trial0"
    checkpoints = [
        _checkpoint(valid_thread_id, "root-valid", None, {"program_name": "demo-cuda"}),
        _checkpoint(valid_thread_id, "tail-valid", "root-valid", _completed_channel_values()),
        _checkpoint(invalid_thread_id, "root-invalid", None, {"program_name": "demo-cuda"}),
        _checkpoint(invalid_thread_id, "tail-invalid", "root-invalid", {"program_name": "demo-cuda"}),
        _checkpoint(invalid_thread_id, "orphan-invalid", None, {"program_name": "demo-cuda"}),
    ]

    tail_result = parser.fetch_tail_checkpoints_by_thread(checkpoints=checkpoints, tolerate_errors=True)
    attempts = {
        valid_thread_id: {
            "failed_attempts": 0,
            "last_status": "completed",
            "last_error": None,
        },
        invalid_thread_id: {
            "failed_attempts": 2,
            "last_status": "failed",
            "last_error": "disconnected lineage",
        },
    }

    samples_df = visualize_results._database_dataframe(tail_result["tails"], attempts, include_dry_run=False)

    assert samples_df["thread_id"].tolist() == [valid_thread_id]
    assert samples_df.iloc[0]["status"] == "completed"
