import importlib.util
import math
from pathlib import Path

import pandas as pd
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
make_plots_for_paper = _load_module("direct_prompting_make_plots_for_paper", "experiments/direct-prompting/make_plots_for_paper.py")
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


def _shared_sample_row(**overrides):
    row = {
        "thread_id": "thread-0",
        "status": "completed",
        "program_name": "demo-cuda",
        "runtime": "cuda",
        "kernel_mangled_name": "_Z4demov",
        "kernel_demangled_name": "demo()",
        "model_name": "openai/gpt-5.4",
        "safe_model_name": "openai_gpt-5.4",
        "use_sass": False,
        "use_imix": False,
        "evidence_configuration": "No SASS / No IMIX",
        "gpu": "A100",
        "trial": 0,
    }
    row.update(overrides)
    return row


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
    ("raw_model_name", "expected_display_name"),
    [
        ("anthropic/claude-4.6-opus", "Opus 4.6"),
        ("openai/gpt-5.4", "GPT 5.4"),
        ("openai/gpt-oss-120B", "GPT OSS"),
        ("some/other-model", "some/other-model"),
    ],
)
def test_display_plot_model_name_shortens_supported_models(raw_model_name, expected_display_name):
    assert visualize_results._display_plot_model_name(raw_model_name) == expected_display_name


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


def test_filter_only_shared_samples_keeps_only_identities_present_for_all_models():
    dataframe = pd.DataFrame(
        [
            _shared_sample_row(thread_id="model-a-shared", model_name="model-a"),
            _shared_sample_row(thread_id="model-b-shared", model_name="model-b"),
            _shared_sample_row(thread_id="model-a-shared-sass", model_name="model-a", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="model-b-shared-sass", model_name="model-b", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="model-a-missing", model_name="model-a", kernel_mangled_name="_Z7missedv"),
            _shared_sample_row(thread_id="model-b-missing", model_name="model-b", kernel_mangled_name="_Z7missedv"),
        ]
    )

    filtered = visualize_results._filter_only_shared_samples(dataframe)

    assert set(filtered["thread_id"].tolist()) == {
        "model-a-shared",
        "model-b-shared",
        "model-a-shared-sass",
        "model-b-shared-sass",
    }
    assert set(filtered["model_name"].tolist()) == {"model-a", "model-b"}
    assert set(filtered["kernel_mangled_name"].tolist()) == {"_Z4demov"}


def test_filter_only_shared_samples_uses_failed_rows_for_eligibility():
    dataframe = pd.DataFrame(
        [
            _shared_sample_row(thread_id="model-a-completed", model_name="model-a"),
            _shared_sample_row(thread_id="model-b-failed", model_name="model-b", status="failed"),
            _shared_sample_row(thread_id="model-a-other", model_name="model-a", kernel_mangled_name="_Z5otherv"),
            _shared_sample_row(thread_id="model-b-other", model_name="model-b", kernel_mangled_name="_Z5otherv"),
            _shared_sample_row(thread_id="model-a-other-sass", model_name="model-a", kernel_mangled_name="_Z5otherv", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="model-b-other-sass", model_name="model-b", kernel_mangled_name="_Z5otherv", use_sass=True, evidence_configuration="SASS Only", status="failed"),
        ]
    )

    filtered = visualize_results._filter_only_shared_samples(dataframe)

    assert set(filtered["thread_id"].tolist()) == {
        "model-a-other",
        "model-b-other",
        "model-a-other-sass",
        "model-b-other-sass",
    }
    assert int((filtered["status"] == "failed").sum()) == 1
    assert "model-b-failed" not in filtered["thread_id"].tolist()


def test_filter_only_shared_samples_requires_cross_prompt_matches_for_all_models():
    dataframe = pd.DataFrame(
        [
            _shared_sample_row(thread_id="shared-a100-no-model-a", model_name="model-a", gpu="A100"),
            _shared_sample_row(thread_id="shared-a100-no-model-b", model_name="model-b", gpu="A100"),
            _shared_sample_row(thread_id="shared-a100-sass-model-a", model_name="model-a", gpu="A100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="shared-a100-sass-model-b", model_name="model-b", gpu="A100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="shared-h100-no-model-a", model_name="model-a", gpu="H100"),
            _shared_sample_row(thread_id="shared-h100-no-model-b", model_name="model-b", gpu="H100"),
            _shared_sample_row(thread_id="shared-h100-sass-model-a", model_name="model-a", gpu="H100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="shared-h100-sass-model-b", model_name="model-b", gpu="H100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="source-only-only-model-a", model_name="model-a", kernel_mangled_name="_Z11sourceonlyv"),
            _shared_sample_row(thread_id="source-only-only-model-b", model_name="model-b", kernel_mangled_name="_Z11sourceonlyv"),
            _shared_sample_row(thread_id="sass-only-model-a", model_name="model-a", kernel_mangled_name="_Z8sassonlyv", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="sass-only-model-b", model_name="model-b", kernel_mangled_name="_Z8sassonlyv", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="h100-no-model-a", model_name="model-a", gpu="H100", kernel_mangled_name="_Z11h100onlyrowv"),
            _shared_sample_row(thread_id="imix-model-a", model_name="model-a", use_imix=True, evidence_configuration="IMIX Only"),
        ]
    )

    filtered = visualize_results._filter_only_shared_samples(dataframe)

    assert set(filtered["thread_id"].tolist()) == {
        "shared-a100-no-model-a",
        "shared-a100-no-model-b",
        "shared-a100-sass-model-a",
        "shared-a100-sass-model-b",
        "shared-h100-no-model-a",
        "shared-h100-no-model-b",
        "shared-h100-sass-model-a",
        "shared-h100-sass-model-b",
    }
    assert "h100-no-model-a" not in filtered["thread_id"].tolist()
    assert "imix-model-a" not in filtered["thread_id"].tolist()


def test_filter_only_shared_samples_requires_matches_for_all_gpus():
    dataframe = pd.DataFrame(
        [
            _shared_sample_row(thread_id="full-a100-no-model-a", model_name="model-a", gpu="A100"),
            _shared_sample_row(thread_id="full-a100-no-model-b", model_name="model-b", gpu="A100"),
            _shared_sample_row(thread_id="full-a100-sass-model-a", model_name="model-a", gpu="A100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="full-a100-sass-model-b", model_name="model-b", gpu="A100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="full-h100-no-model-a", model_name="model-a", gpu="H100"),
            _shared_sample_row(thread_id="full-h100-no-model-b", model_name="model-b", gpu="H100"),
            _shared_sample_row(thread_id="full-h100-sass-model-a", model_name="model-a", gpu="H100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="full-h100-sass-model-b", model_name="model-b", gpu="H100", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="missing-gpu-a100-no-model-a", model_name="model-a", gpu="A100", kernel_mangled_name="_Z10missinggpuv"),
            _shared_sample_row(thread_id="missing-gpu-a100-no-model-b", model_name="model-b", gpu="A100", kernel_mangled_name="_Z10missinggpuv"),
            _shared_sample_row(thread_id="missing-gpu-a100-sass-model-a", model_name="model-a", gpu="A100", kernel_mangled_name="_Z10missinggpuv", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="missing-gpu-a100-sass-model-b", model_name="model-b", gpu="A100", kernel_mangled_name="_Z10missinggpuv", use_sass=True, evidence_configuration="SASS Only"),
        ]
    )

    filtered = visualize_results._filter_only_shared_samples(dataframe)

    assert set(filtered["thread_id"].tolist()) == {
        "full-a100-no-model-a",
        "full-a100-no-model-b",
        "full-a100-sass-model-a",
        "full-a100-sass-model-b",
        "full-h100-no-model-a",
        "full-h100-no-model-b",
        "full-h100-sass-model-a",
        "full-h100-sass-model-b",
    }
    assert set(filtered["gpu"].tolist()) == {"A100", "H100"}
    assert "_Z10missinggpuv" not in filtered["kernel_mangled_name"].tolist()


def test_filter_only_shared_samples_with_include_imix_requires_all_prompt_types():
    dataframe = pd.DataFrame(
        [
            _shared_sample_row(thread_id="full-no-model-a", model_name="model-a"),
            _shared_sample_row(thread_id="full-no-model-b", model_name="model-b"),
            _shared_sample_row(thread_id="full-sass-model-a", model_name="model-a", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="full-sass-model-b", model_name="model-b", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="full-imix-model-a", model_name="model-a", use_imix=True, evidence_configuration="IMIX Only"),
            _shared_sample_row(thread_id="full-imix-model-b", model_name="model-b", use_imix=True, evidence_configuration="IMIX Only"),
            _shared_sample_row(thread_id="full-sass-imix-model-a", model_name="model-a", use_sass=True, use_imix=True, evidence_configuration="SASS + IMIX"),
            _shared_sample_row(thread_id="full-sass-imix-model-b", model_name="model-b", use_sass=True, use_imix=True, evidence_configuration="SASS + IMIX"),
            _shared_sample_row(thread_id="missing-imix-no-model-a", model_name="model-a", kernel_mangled_name="_Z11missingimixv"),
            _shared_sample_row(thread_id="missing-imix-no-model-b", model_name="model-b", kernel_mangled_name="_Z11missingimixv"),
            _shared_sample_row(thread_id="missing-imix-sass-model-a", model_name="model-a", kernel_mangled_name="_Z11missingimixv", use_sass=True, evidence_configuration="SASS Only"),
            _shared_sample_row(thread_id="missing-imix-sass-model-b", model_name="model-b", kernel_mangled_name="_Z11missingimixv", use_sass=True, evidence_configuration="SASS Only"),
        ]
    )

    filtered = visualize_results._filter_only_shared_samples(dataframe, include_imix=True)

    assert set(filtered["thread_id"].tolist()) == {
        "full-no-model-a",
        "full-no-model-b",
        "full-sass-model-a",
        "full-sass-model-b",
        "full-imix-model-a",
        "full-imix-model-b",
        "full-sass-imix-model-a",
        "full-sass-imix-model-b",
    }


def test_visualize_results_parser_accepts_only_shared_samples_flag():
    parser = visualize_results.build_arg_parser()

    default_args = parser.parse_args([])
    flagged_args = parser.parse_args(["--onlySharedSamples", "--includeIMIX"])

    assert default_args.onlySharedSamples is False
    assert flagged_args.onlySharedSamples is True
    assert flagged_args.includeIMIX is True


def test_make_plots_for_paper_parser_accepts_only_shared_samples_flag():
    parser = make_plots_for_paper.build_arg_parser()

    default_args = parser.parse_args([])
    flagged_args = parser.parse_args(["--onlySharedSamples"])

    assert default_args.onlySharedSamples is False
    assert flagged_args.onlySharedSamples is True


def test_summarize_expected_rai_distribution_counts_unique_kernels_per_gpu_precision():
    def _paper_plot_row(
        *,
        thread_id: str,
        kernel_mangled_name: str,
        model_name: str,
        use_sass: bool,
        expected_fp16: float,
        expected_fp32: float,
        expected_fp64: float,
        expected_read_bytes: float,
        expected_write_bytes: float,
    ):
        return _shared_sample_row(
            thread_id=thread_id,
            kernel_mangled_name=kernel_mangled_name,
            model_name=model_name,
            safe_model_name=model_name.replace("/", "_"),
            use_sass=use_sass,
            evidence_configuration="SASS Only" if use_sass else "No SASS / No IMIX",
            expected_fp16=expected_fp16,
            expected_fp32=expected_fp32,
            expected_fp64=expected_fp64,
            expected_read_bytes=expected_read_bytes,
            expected_write_bytes=expected_write_bytes,
            metrics_diff_fp16=0,
            metrics_diff_fp32=0,
            metrics_diff_fp64=0,
            metrics_diff_read_bytes=0,
            metrics_diff_write_bytes=0,
        )

    samples_df = pd.DataFrame(
        [
            _paper_plot_row(
                thread_id="a100-zero-model-a",
                kernel_mangled_name="_Z9zeroKernelv",
                model_name="model-a",
                use_sass=False,
                expected_fp16=0,
                expected_fp32=0,
                expected_fp64=0,
                expected_read_bytes=128,
                expected_write_bytes=0,
            ),
            _paper_plot_row(
                thread_id="a100-zero-model-b",
                kernel_mangled_name="_Z9zeroKernelv",
                model_name="model-b",
                use_sass=True,
                expected_fp16=0,
                expected_fp32=0,
                expected_fp64=0,
                expected_read_bytes=128,
                expected_write_bytes=0,
            ),
            _paper_plot_row(
                thread_id="a100-bb-model-a",
                kernel_mangled_name="_Z7bbKernelv",
                model_name="model-a",
                use_sass=False,
                expected_fp16=10,
                expected_fp32=10,
                expected_fp64=10,
                expected_read_bytes=100,
                expected_write_bytes=0,
            ),
            _paper_plot_row(
                thread_id="a100-cb-model-a",
                kernel_mangled_name="_Z7cbKernelv",
                model_name="model-a",
                use_sass=False,
                expected_fp16=6000,
                expected_fp32=2000,
                expected_fp64=1000,
                expected_read_bytes=100,
                expected_write_bytes=0,
            ),
        ]
    )

    completed_df = make_plots_for_paper._enrich_completed_dataframe(samples_df)
    plot_df = make_plots_for_paper._paper_subset(completed_df)
    distribution_df = make_plots_for_paper._summarize_expected_rai_distribution(plot_df)

    assert distribution_df["precision"].tolist() == ["FP16", "FP32", "FP64"]
    assert distribution_df["zero_rai_n"].tolist() == [1, 1, 1]
    assert distribution_df["nonzero_bandwidth_bound_n"].tolist() == [1, 1, 1]
    assert distribution_df["nonzero_compute_bound_n"].tolist() == [1, 1, 1]
    assert distribution_df["nonzero_rai_n"].tolist() == [2, 2, 2]
    assert distribution_df["total_kernels"].tolist() == [3, 3, 3]
    assert distribution_df["count_string"].tolist() == ["(1|1|1)", "(1|1|1)", "(1|1|1)"]


def test_classify_ai_with_zero_distinguishes_zero_bandwidth_compute_and_nan():
    assert make_plots_for_paper._classify_ai_with_zero(0.0, 2.0) == make_plots_for_paper.ZERO_CLASS
    assert make_plots_for_paper._classify_ai_with_zero(1.0, 2.0) == make_plots_for_paper.NEGATIVE_CLASS
    assert make_plots_for_paper._classify_ai_with_zero(4.0, 2.0) == make_plots_for_paper.POSITIVE_CLASS
    assert make_plots_for_paper._classify_ai_with_zero(float("nan"), 2.0) is None
    assert make_plots_for_paper._classify_ai_with_zero(1.0, float("nan")) is None


def test_figure2_5_confusion_heatmap_payload_includes_zero_expected_and_predicted_classes():
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "use_sass": False,
                "expected_ai_fp16": 0.0,
                "predicted_ai_fp16": 0.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": 0.0,
                "predicted_ai_fp32": 0.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": 0.0,
                "predicted_ai_fp64": 0.0,
                "balance_point_fp64": 2.0,
            },
            {
                "model_name": "model-a",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 0.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": 1.0,
                "predicted_ai_fp32": 0.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": 1.0,
                "predicted_ai_fp64": 0.0,
                "balance_point_fp64": 2.0,
            },
            {
                "model_name": "model-a",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": 1.0,
                "predicted_ai_fp32": 1.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": 1.0,
                "predicted_ai_fp64": 1.0,
                "balance_point_fp64": 2.0,
            },
            {
                "model_name": "model-a",
                "use_sass": False,
                "expected_ai_fp16": 4.0,
                "predicted_ai_fp16": 4.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 4.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 4.0,
                "balance_point_fp64": 2.0,
            },
            {
                "model_name": "model-a",
                "use_sass": False,
                "expected_ai_fp16": float("nan"),
                "predicted_ai_fp16": 0.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": float("nan"),
                "predicted_ai_fp32": 0.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": float("nan"),
                "predicted_ai_fp64": 0.0,
                "balance_point_fp64": 2.0,
            },
        ]
    )

    matrix_df, annotation = make_plots_for_paper._figure2_5_confusion_heatmap_payload(
        plot_df,
        "model-a",
        False,
    )

    assert matrix_df.index.tolist() == ["Zero", "BB", "CB"]
    assert matrix_df.columns.tolist() == ["Zero", "BB", "CB"]
    assert matrix_df.loc["Zero", "Zero"] == pytest.approx(100.0)
    assert matrix_df.loc["BB", "Zero"] == pytest.approx(50.0)
    assert matrix_df.loc["BB", "BB"] == pytest.approx(50.0)
    assert matrix_df.loc["CB", "CB"] == pytest.approx(100.0)
    assert matrix_df.loc["Zero"].sum() == pytest.approx(100.0)
    assert matrix_df.loc["BB"].sum() == pytest.approx(100.0)
    assert matrix_df.loc["CB"].sum() == pytest.approx(100.0)
    assert "FP16: 100.0%" in annotation[0, 0]
    assert "FP32: 50.0%" in annotation[1, 0]


def test_save_figure2_5_bound_heatmaps_with_zero_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "use_sass": False,
                "expected_ai_fp16": 0.0,
                "predicted_ai_fp16": 0.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": 0.0,
                "predicted_ai_fp32": 0.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": 0.0,
                "predicted_ai_fp64": 0.0,
                "balance_point_fp64": 2.0,
            },
            {
                "model_name": "model-a",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": 1.0,
                "predicted_ai_fp32": 1.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": 1.0,
                "predicted_ai_fp64": 1.0,
                "balance_point_fp64": 2.0,
            },
            {
                "model_name": "model-b",
                "use_sass": True,
                "expected_ai_fp16": 4.0,
                "predicted_ai_fp16": 0.0,
                "balance_point_fp16": 2.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 0.0,
                "balance_point_fp32": 2.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 0.0,
                "balance_point_fp64": 2.0,
            },
        ]
    )

    output_path = tmp_path / "figure2_5_ai_bound_confusion_heatmaps_with_zero.png"

    make_plots_for_paper._save_figure2_5_bound_heatmaps_with_zero(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_summarize_runtime_distribution_counts_unique_kernels_per_gpu():
    plot_df = pd.DataFrame(
        [
            {
                "program_name": "prog-a",
                "kernel_mangled_name": "_Z7kernelAv",
                "gpu": "A100",
                "runtime": "cuda",
                "model_name": "model-a",
            },
            {
                "program_name": "prog-a",
                "kernel_mangled_name": "_Z7kernelAv",
                "gpu": "A100",
                "runtime": "cuda",
                "model_name": "model-b",
            },
            {
                "program_name": "prog-a",
                "kernel_mangled_name": "_Z7kernelAv",
                "gpu": "H100",
                "runtime": "cuda",
                "model_name": "model-a",
            },
            {
                "program_name": "prog-b",
                "kernel_mangled_name": "_Z7kernelBv",
                "gpu": "A100",
                "runtime": "cuda",
                "model_name": "model-a",
            },
            {
                "program_name": "prog-c",
                "kernel_mangled_name": "_Z7kernelCv",
                "gpu": "A100",
                "runtime": "omp",
                "model_name": "model-a",
            },
            {
                "program_name": "prog-c",
                "kernel_mangled_name": "_Z7kernelCv",
                "gpu": "H100",
                "runtime": "omp",
                "model_name": "model-a",
            },
        ]
    )

    runtime_df = make_plots_for_paper._summarize_runtime_distribution(plot_df)

    assert runtime_df["gpu"].tolist() == ["All GPUs"]
    assert runtime_df["precision"].tolist() == ["Runtime"]
    assert runtime_df["cuda_n"].tolist() == [2]
    assert runtime_df["omp_n"].tolist() == [1]
    assert runtime_df["total_kernels"].tolist() == [3]
    assert runtime_df["count_string"].tolist() == ["(2|1)"]


def test_prepare_ai_pct_long_df_uses_same_nonzero_expected_rows_as_figure1():
    samples_df = pd.DataFrame(
        [
            _shared_sample_row(
                thread_id="thread-a",
                model_name="model-a",
                expected_fp16=0,
                expected_fp32=50,
                expected_fp64=25,
                expected_read_bytes=80,
                expected_write_bytes=20,
                metrics_diff_fp16=0,
                metrics_diff_fp32=10,
                metrics_diff_fp64=-5,
                metrics_diff_read_bytes=0,
                metrics_diff_write_bytes=0,
            ),
            _shared_sample_row(
                thread_id="thread-b",
                model_name="model-b",
                expected_fp16=0,
                expected_fp32=0,
                expected_fp64=0,
                expected_read_bytes=80,
                expected_write_bytes=20,
                metrics_diff_fp16=0,
                metrics_diff_fp32=0,
                metrics_diff_fp64=0,
                metrics_diff_read_bytes=0,
                metrics_diff_write_bytes=0,
            ),
        ]
    )

    completed_df = make_plots_for_paper._enrich_completed_dataframe(samples_df)
    plot_df = make_plots_for_paper._paper_subset(completed_df)
    ai_long_df = make_plots_for_paper._prepare_ai_long_df(plot_df)
    ai_pct_long_df = make_plots_for_paper._prepare_ai_pct_long_df(plot_df)

    assert ai_long_df["precision"].tolist() == ["FP32 RAI", "FP64 RAI"]
    assert ai_pct_long_df["precision"].tolist() == ["FP32 RAI", "FP64 RAI"]
    assert ai_pct_long_df["ai_pct_diff"].tolist() == pytest.approx([20.0, -20.0])


def test_percent_diff_axis_config_uses_fixed_negative_bound_and_dynamic_positive_bound():
    axis_config = make_plots_for_paper._percent_diff_axis_config(pd.Series([-80.0, -5.0, 12.0, 450.0]))

    assert axis_config["x_limits"] == pytest.approx((-110.0, 463.5))
    assert axis_config["x_ticks"] == [-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0]
    assert axis_config["x_tick_labels"] == ["-100", "-75", "-50", "-25", "0", "25", "50", "75", "100"]
    assert axis_config["linthresh"] == 100.0
    assert axis_config["linscale"] == 3.0


def test_summarize_pct_error_thresholds_reports_counts_and_percentages():
    ai_pct_long_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "precision": "FP32 RAI",
                "ai_pct_diff": 5.0,
            },
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "precision": "FP32 RAI",
                "ai_pct_diff": -15.0,
            },
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "precision": "FP32 RAI",
                "ai_pct_diff": 80.0,
            },
        ]
    )

    summary_df = make_plots_for_paper._summarize_pct_error_thresholds(ai_pct_long_df, ["model_name"])

    assert summary_df["threshold_pct"].tolist() == [10.0, 20.0, 25.0, 50.0, 75.0, 100.0]
    assert summary_df["threshold_label"].tolist() == ["+/-10%", "+/-20%", "+/-25%", "+/-50%", "+/-75%", "+/-100%"]
    assert summary_df["within_threshold_n"].tolist() == [1, 2, 2, 2, 2, 3]
    assert summary_df["total_n"].tolist() == [3, 3, 3, 3, 3, 3]
    assert summary_df["within_threshold_pct"].tolist() == pytest.approx(
        [33.3333333333, 66.6666666667, 66.6666666667, 66.6666666667, 66.6666666667, 100.0]
    )


def test_build_figure12_8_pct_threshold_table_fills_missing_combinations_with_dashes():
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 100.0,
                "predicted_ai_fp16": 105.0,
                "expected_ai_fp32": 100.0,
                "predicted_ai_fp32": 80.0,
                "expected_ai_fp64": 100.0,
                "predicted_ai_fp64": 160.0,
            },
            {
                "model_name": "model-b",
                "gpu": "H100",
                "runtime": "omp",
                "use_sass": True,
                "expected_ai_fp16": 100.0,
                "predicted_ai_fp16": 100.0,
                "expected_ai_fp32": 100.0,
                "predicted_ai_fp32": 100.0,
                "expected_ai_fp64": 100.0,
                "predicted_ai_fp64": 100.0,
            },
        ]
    )

    table_df = make_plots_for_paper._build_figure12_8_pct_threshold_table(plot_df)

    present_row = table_df[
        (table_df["Model"] == "model-a")
        & (table_df["Runtime"] == "CUDA")
        & (table_df["Evidence"] == "Source-Only")
        & (table_df["GPU"] == "A100")
    ].iloc[0]
    missing_row = table_df[
        (table_df["Model"] == "model-a")
        & (table_df["Runtime"] == "OpenMP")
        & (table_df["Evidence"] == "Source+SASS")
        & (table_df["GPU"] == "H100")
    ].iloc[0]

    assert present_row["FP16 +/-10%"] == "100.0"
    assert present_row["FP16 +/-50%"] == "100.0"
    assert present_row["FP32 +/-10%"] == "0.0"
    assert present_row["FP32 +/-50%"] == "100.0"
    assert present_row["FP64 +/-10%"] == "0.0"
    assert present_row["FP64 +/-50%"] == "0.0"
    assert missing_row["FP16 +/-10%"] == "-"
    assert missing_row["FP16 +/-50%"] == "-"
    assert missing_row["FP32 +/-10%"] == "-"
    assert missing_row["FP32 +/-50%"] == "-"
    assert missing_row["FP64 +/-10%"] == "-"
    assert missing_row["FP64 +/-50%"] == "-"


def test_write_figure12_8_booktabs_table_writes_compact_tex_with_multicolumn_and_multirow(tmp_path: Path):
    table_df = pd.DataFrame(
        [
            {
                "Model": "model-a",
                "Runtime": "CUDA",
                "Evidence": "Source-Only",
                "GPU": "A100",
                "FP16 +/-10%": "100.0",
                "FP16 +/-50%": "100.0",
                "FP32 +/-10%": "0.0",
                "FP32 +/-50%": "100.0",
                "FP64 +/-10%": "-",
                "FP64 +/-50%": "-",
            },
            {
                "Model": "model-a",
                "Runtime": "CUDA",
                "Evidence": "Source-Only",
                "GPU": "H100",
                "FP16 +/-10%": "0.0",
                "FP16 +/-50%": "50.0",
                "FP32 +/-10%": "10.0",
                "FP32 +/-50%": "60.0",
                "FP64 +/-10%": "-",
                "FP64 +/-50%": "-",
            },
            {
                "Model": "model-a",
                "Runtime": "CUDA",
                "Evidence": "Source+SASS",
                "GPU": "A100",
                "FP16 +/-10%": "5.0",
                "FP16 +/-50%": "55.0",
                "FP32 +/-10%": "15.0",
                "FP32 +/-50%": "65.0",
                "FP64 +/-10%": "-",
                "FP64 +/-50%": "-",
            },
            {
                "Model": "model-a",
                "Runtime": "OpenMP",
                "Evidence": "Source-Only",
                "GPU": "A100",
                "FP16 +/-10%": "-",
                "FP16 +/-50%": "-",
                "FP32 +/-10%": "20.0",
                "FP32 +/-50%": "70.0",
                "FP64 +/-10%": "10.0",
                "FP64 +/-50%": "80.0",
            },
            {
                "Model": "model-b",
                "Runtime": "CUDA",
                "Evidence": "Source-Only",
                "GPU": "A100",
                "FP16 +/-10%": "25.0",
                "FP16 +/-50%": "75.0",
                "FP32 +/-10%": "35.0",
                "FP32 +/-50%": "85.0",
                "FP64 +/-10%": "45.0",
                "FP64 +/-50%": "95.0",
            },
        ]
    )

    output_path = tmp_path / "table_figure12_8_threshold_coverage.tex"

    make_plots_for_paper._write_figure12_8_booktabs_table(table_df, output_path)

    tex_text = output_path.read_text(encoding="utf-8")

    assert "\\begingroup" in tex_text
    assert "\\scriptsize" in tex_text
    assert "\\setlength{\\tabcolsep}{4pt}" in tex_text
    assert "\\toprule" in tex_text
    assert "\\midrule" in tex_text
    assert "\\bottomrule" in tex_text
    assert "\\multicolumn{2}{c}{FP16}" in tex_text
    assert "\\multicolumn{2}{c}{FP32}" in tex_text
    assert "\\multicolumn{2}{c}{FP64}" in tex_text
    assert "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}" in tex_text
    assert "\\multirow{4}{*}{model-a}" in tex_text
    assert "\\multirow{3}{*}{CUDA}" in tex_text
    assert "\\multirow{2}{*}{Source-Only}" in tex_text
    assert "\\cmidrule(lr){3-10}" in tex_text
    assert "\\cmidrule(lr){2-10}" in tex_text
    assert tex_text.count("\\midrule") == 2
    assert r"\cellcolor[rgb]{1.000,0.000,0.000}100.0" in tex_text
    assert r"\cellcolor[rgb]{1.000,1.000,1.000}0.0" in tex_text
    assert r"\cellcolor[rgb]{1.000,1.000,1.000}-" in tex_text
    assert "A100 & \\cellcolor[rgb]{1.000,0.000,0.000}100.0" in tex_text
    assert "H100 & \\cellcolor[rgb]{1.000,1.000,1.000}0.0 & \\cellcolor[rgb]{1.000,0.670,0.670}50.0" in tex_text


def test_prepare_ai_ape_long_df_uses_same_rows_as_figure11_with_absolute_values():
    samples_df = pd.DataFrame(
        [
            _shared_sample_row(
                thread_id="thread-a",
                model_name="model-a",
                expected_fp16=0,
                expected_fp32=50,
                expected_fp64=25,
                expected_read_bytes=80,
                expected_write_bytes=20,
                metrics_diff_fp16=0,
                metrics_diff_fp32=10,
                metrics_diff_fp64=-5,
                metrics_diff_read_bytes=0,
                metrics_diff_write_bytes=0,
            ),
        ]
    )

    completed_df = make_plots_for_paper._enrich_completed_dataframe(samples_df)
    plot_df = make_plots_for_paper._paper_subset(completed_df)
    ai_pct_long_df = make_plots_for_paper._prepare_ai_pct_long_df(plot_df)
    ai_ape_long_df = make_plots_for_paper._prepare_ai_ape_long_df(plot_df)

    assert ai_ape_long_df["precision"].tolist() == ai_pct_long_df["precision"].tolist()
    assert ai_ape_long_df["ai_ape"].tolist() == pytest.approx([20.0, 20.0])


def test_prepare_ai_log_ratio_long_df_uses_base10_ratio_and_skips_invalid_negative_predictions():
    samples_df = pd.DataFrame(
        [
            _shared_sample_row(
                thread_id="thread-a",
                model_name="model-a",
                expected_fp16=0,
                expected_fp32=50,
                expected_fp64=25,
                expected_read_bytes=80,
                expected_write_bytes=20,
                metrics_diff_fp16=0,
                metrics_diff_fp32=10,
                metrics_diff_fp64=-30,
                metrics_diff_read_bytes=0,
                metrics_diff_write_bytes=0,
            ),
        ]
    )

    completed_df = make_plots_for_paper._enrich_completed_dataframe(samples_df)
    plot_df = make_plots_for_paper._paper_subset(completed_df)
    ai_log_ratio_long_df = make_plots_for_paper._prepare_ai_log_ratio_long_df(plot_df)

    expected_ai_fp32 = 50.0 / 100.0
    predicted_ai_fp32 = 60.0 / 100.0
    expected_log_ratio = math.log10(
        (predicted_ai_fp32 + make_plots_for_paper.LOG_RATIO_EPSILON)
        / (expected_ai_fp32 + make_plots_for_paper.LOG_RATIO_EPSILON)
    )

    assert ai_log_ratio_long_df["precision"].tolist() == ["FP32 RAI"]
    assert ai_log_ratio_long_df["ai_log_ratio"].tolist() == pytest.approx([expected_log_ratio])


def test_ape_axis_config_uses_linear_region_through_100_then_log_scale():
    axis_config = make_plots_for_paper._ape_axis_config(pd.Series([5.0, 80.0, 1200.0]))

    assert axis_config["x_limits"] == pytest.approx((-1.0, 1236.0))
    assert axis_config["x_ticks"] == [0.0, 25.0, 50.0, 75.0, 100.0, 1000.0]
    assert axis_config["x_tick_labels"] == ["0", "25", "50", "75", "100", r"$10^{3}$"]
    assert axis_config["linthresh"] == 100.0
    assert axis_config["linscale"] == 3.0


def test_log_ratio_axis_config_uses_fixed_hybrid_limits_ticks_and_scale_functions():
    axis_config = make_plots_for_paper._log_ratio_axis_config(pd.Series([-1.2, -0.1, 0.35, 2.7]))
    forward, inverse = axis_config["scale_functions"]

    assert axis_config["x_limits"] == pytest.approx((-18.0, 6.0))
    assert axis_config["x_ticks"] == [-18.0, -12.0, -6.0, -3.0, -2.0, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, 2.0, 4.0, 6.0]
    assert axis_config["x_tick_labels"] == [
        "-18",
        "-12",
        "-6",
        "-3",
        "-2",
        "-1",
        r"$-10^{-1}$",
        r"$-10^{-2}$",
        "0",
        r"$10^{-2}$",
        r"$10^{-1}$",
        "1",
        "2",
        "4",
        "6",
    ]
    assert forward(1.0) - forward(-1.0) == pytest.approx(0.6)
    assert [inverse(forward(value)) for value in [-18.0, -1.0, -0.1, 0.0, 0.1, 1.0, 6.0]] == pytest.approx(
        [-18.0, -1.0, -0.1, 0.0, 0.1, 1.0, 6.0]
    )


def test_save_figure6_expected_rai_distribution_writes_png(tmp_path: Path):
    distribution_df = pd.DataFrame(
        [
            {
                "gpu": "A100",
                "precision": "FP16",
                "zero_rai_n": 1,
                "nonzero_bandwidth_bound_n": 2,
                "nonzero_compute_bound_n": 3,
                "nonzero_rai_n": 5,
                "total_kernels": 6,
                "count_string": "(1|2|3)",
            },
            {
                "gpu": "A100",
                "precision": "FP32",
                "zero_rai_n": 0,
                "nonzero_bandwidth_bound_n": 1,
                "nonzero_compute_bound_n": 2,
                "nonzero_rai_n": 3,
                "total_kernels": 3,
                "count_string": "(0|1|2)",
            },
        ]
    )
    runtime_distribution_df = pd.DataFrame(
        [
            {
                "gpu": "A100",
                "precision": "Runtime",
                "cuda_n": 2,
                "omp_n": 1,
                "total_kernels": 3,
                "count_string": "(2|1)",
            }
        ]
    )

    output_path = tmp_path / "figure6_expected_rai_distribution_by_gpu_precision.png"

    make_plots_for_paper._save_figure6_expected_rai_distribution(
        distribution_df,
        output_path,
        runtime_distribution_df,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure11_ai_pct_boxplots_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure11_ai_percent_difference_boxplots.png"

    make_plots_for_paper._save_figure11_ai_pct_boxplots(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure12_ai_pct_boxplots_by_gpu_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "H100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure12_ai_percent_difference_boxplots_by_gpu.png"

    make_plots_for_paper._save_figure12_ai_pct_boxplots_by_gpu(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure12_5_ai_pct_boxplots_by_gpu_and_model_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "H100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
            {
                "model_name": "model-c",
                "gpu": "3080",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 3.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 6.0,
                "predicted_ai_fp32": 3.0,
                "expected_ai_fp64": 9.0,
                "predicted_ai_fp64": 18.0,
            },
        ]
    )

    output_path = tmp_path / "figure12_5_ai_percent_difference_boxplots_by_gpu_and_model.png"

    make_plots_for_paper._save_figure12_5_ai_pct_boxplots_by_gpu_and_model(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure12_8_ai_pct_boxplots_by_gpu_runtime_and_model_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "H100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
            {
                "model_name": "model-c",
                "gpu": "3080",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 3.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 6.0,
                "predicted_ai_fp32": 3.0,
                "expected_ai_fp64": 9.0,
                "predicted_ai_fp64": 18.0,
            },
            {
                "model_name": "model-a",
                "gpu": "A10",
                "runtime": "omp",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 2.0,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 3.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 2.0,
            },
            {
                "model_name": "model-b",
                "gpu": "A100",
                "runtime": "omp",
                "use_sass": True,
                "expected_ai_fp16": 5.0,
                "predicted_ai_fp16": 2.5,
                "expected_ai_fp32": 10.0,
                "predicted_ai_fp32": 20.0,
                "expected_ai_fp64": 20.0,
                "predicted_ai_fp64": 10.0,
            },
            {
                "model_name": "model-c",
                "gpu": "H100",
                "runtime": "omp",
                "use_sass": False,
                "expected_ai_fp16": 4.0,
                "predicted_ai_fp16": 8.0,
                "expected_ai_fp32": 8.0,
                "predicted_ai_fp32": 4.0,
                "expected_ai_fp64": 16.0,
                "predicted_ai_fp64": 32.0,
            },
        ]
    )

    output_path = tmp_path / "figure12_8_ai_percent_difference_boxplots_by_gpu_runtime_and_model.png"

    make_plots_for_paper._save_figure12_8_ai_pct_boxplots_by_gpu_runtime_and_model(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure13_ai_pct_boxplots_by_runtime_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "A100",
                "runtime": "omp",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure13_ai_percent_difference_boxplots_by_runtime.png"

    make_plots_for_paper._save_figure13_ai_pct_boxplots_by_runtime(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure14_ai_log_ratio_boxplots_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure14_ai_log10_ratio_error_boxplots.png"

    make_plots_for_paper._save_figure14_ai_log_ratio_boxplots(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure15_ai_log_ratio_boxplots_by_gpu_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "H100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
            {
                "model_name": "model-c",
                "gpu": "3080",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 3.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 6.0,
                "predicted_ai_fp32": 3.0,
                "expected_ai_fp64": 9.0,
                "predicted_ai_fp64": 18.0,
            },
        ]
    )

    output_path = tmp_path / "figure15_ai_log10_ratio_error_boxplots_by_gpu.png"

    make_plots_for_paper._save_figure15_ai_log_ratio_boxplots_by_gpu(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure16_ai_log_ratio_boxplots_by_runtime_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 8.0,
            },
            {
                "model_name": "model-b",
                "gpu": "A100",
                "runtime": "omp",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure16_ai_log10_ratio_error_boxplots_by_runtime.png"

    make_plots_for_paper._save_figure16_ai_log_ratio_boxplots_by_runtime(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure8_ai_ape_boxplots_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 64.0,
            },
            {
                "model_name": "model-b",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure8_ai_absolute_percent_error_boxplots.png"

    make_plots_for_paper._save_figure8_ai_ape_boxplots(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure9_ai_ape_boxplots_by_gpu_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 64.0,
            },
            {
                "model_name": "model-b",
                "gpu": "H100",
                "runtime": "cuda",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure9_ai_absolute_percent_error_boxplots_by_gpu.png"

    make_plots_for_paper._save_figure9_ai_ape_boxplots_by_gpu(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_figure10_ai_ape_boxplots_by_runtime_writes_png(tmp_path: Path):
    plot_df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "gpu": "A100",
                "runtime": "cuda",
                "use_sass": False,
                "expected_ai_fp16": 1.0,
                "predicted_ai_fp16": 1.5,
                "expected_ai_fp32": 2.0,
                "predicted_ai_fp32": 1.0,
                "expected_ai_fp64": 4.0,
                "predicted_ai_fp64": 64.0,
            },
            {
                "model_name": "model-b",
                "gpu": "A100",
                "runtime": "omp",
                "use_sass": True,
                "expected_ai_fp16": 2.0,
                "predicted_ai_fp16": 1.0,
                "expected_ai_fp32": 4.0,
                "predicted_ai_fp32": 8.0,
                "expected_ai_fp64": 8.0,
                "predicted_ai_fp64": 4.0,
            },
        ]
    )

    output_path = tmp_path / "figure10_ai_absolute_percent_error_boxplots_by_runtime.png"

    make_plots_for_paper._save_figure10_ai_ape_boxplots_by_runtime(plot_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_write_paper_summary_tables_prints_figure11_percent_difference_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    samples_df = pd.DataFrame(
        [
            _shared_sample_row(
                thread_id="thread-a",
                model_name="model-a",
                runtime="cuda",
                use_sass=False,
                gpu="A100",
                expected_fp16=0,
                expected_fp32=50,
                expected_fp64=25,
                expected_read_bytes=80,
                expected_write_bytes=20,
                metrics_diff_fp16=0,
                metrics_diff_fp32=10,
                metrics_diff_fp64=-5,
                metrics_diff_read_bytes=0,
                metrics_diff_write_bytes=0,
            ),
            _shared_sample_row(
                thread_id="thread-b",
                model_name="model-a",
                runtime="cuda",
                use_sass=True,
                gpu="A100",
                expected_fp16=0,
                expected_fp32=40,
                expected_fp64=20,
                expected_read_bytes=80,
                expected_write_bytes=20,
                metrics_diff_fp16=0,
                metrics_diff_fp32=-8,
                metrics_diff_fp64=10,
                metrics_diff_read_bytes=0,
                metrics_diff_write_bytes=0,
            ),
        ]
    )

    completed_df = make_plots_for_paper._enrich_completed_dataframe(samples_df)
    plot_df = make_plots_for_paper._paper_subset(completed_df)

    make_plots_for_paper._write_paper_summary_tables(
        plot_df,
        tmp_path,
        expected_rai_distribution_df=pd.DataFrame(),
    )

    captured = capsys.readouterr()

    assert "figure11_rai_percent_difference_summary" in captured.out
    assert "Prompt Type | Precision | N | Q1" in captured.out
    assert "model-a" in captured.out
