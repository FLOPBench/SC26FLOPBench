import importlib.util
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

    output_path = tmp_path / "figure6_expected_rai_distribution_by_gpu_precision.png"

    make_plots_for_paper._save_figure6_expected_rai_distribution(distribution_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
