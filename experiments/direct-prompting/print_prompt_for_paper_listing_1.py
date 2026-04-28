import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(WORKSPACE_ROOT) / "dataset-creation" / "gpuFLOPBench.json"
DEFAULT_LISTING1_PATH = SCRIPT_DIR / "listing1.txt"
DEFAULT_LISTING2_PATH = SCRIPT_DIR / "listing2.txt"
CANONICAL_PROGRAM_NAME = "adam-cuda"
CANONICAL_GPU_NAME = "H100"
DEFAULT_MODEL_PREFERENCES = (
	"anthropic/claude-4.6-opus",
	"anthropic/claude-opus-4.6",
)


def _load_module(module_name: str, file_path: Path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Unable to load module {module_name} from {file_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


prompts = _load_module(
	"direct_prompting_prompts",
	Path(WORKSPACE_ROOT) / "experiments" / "direct-prompting" / "prompts.py",
)
db_manager = _load_module(
	"direct_prompting_db_manager",
	Path(WORKSPACE_ROOT) / "experiments" / "direct-prompting" / "db_manager.py",
)
run_queries = _load_module(
	"direct_prompting_run_queries",
	Path(WORKSPACE_ROOT) / "experiments" / "direct-prompting" / "run_queries.py",
)

DirectPromptGenerator = prompts.DirectPromptGenerator
CheckpointDBParser = db_manager.CheckpointDBParser
ensure_postgres_running = db_manager.ensure_postgres_running
setup_default_database = db_manager.setup_default_database
print_run_result = run_queries.print_run_result


def _load_dataset(dataset_path: Path) -> Dict[str, Any]:
	with dataset_path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _canonical_single_query_target(dataset: Dict[str, Any]) -> Dict[str, str]:
	for program_name, program_data in dataset.items():
		if program_name != CANONICAL_PROGRAM_NAME:
			continue

		for kernel_mangled_name, kernel_data in program_data["kernels"].items():
			for gpu_name in kernel_data["metrics"].keys():
				if gpu_name != CANONICAL_GPU_NAME:
					continue

				return {
					"program_name": program_name,
					"kernel_mangled_name": kernel_mangled_name,
					"kernel_demangled_name": kernel_data["demangledName"],
				}

	raise RuntimeError(
		"Could not derive the canonical single-query example from the dataset."
	)


def _channel_values(checkpoint_record: Dict[str, Any]) -> Dict[str, Any]:
	return checkpoint_record.get("checkpoint", {}).get("channel_values", {})


def _is_completed_checkpoint(checkpoint_record: Dict[str, Any]) -> bool:
	return "total_tokens" in _channel_values(checkpoint_record)


def _matches_model(model_filter: Optional[str], state: Dict[str, Any], thread_id: str) -> bool:
	if not model_filter:
		return True

	normalized_filter = model_filter.strip()
	llm_model_name = state.get("llm_model_name") or ""
	return llm_model_name.startswith(normalized_filter) or normalized_filter in thread_id


def _hydrate_channels(parser: CheckpointDBParser, checkpoint_record: Dict[str, Any]) -> None:
	parser.hydrate_checkpoint_channels(
		checkpoint_record,
		[
			"compile_commands",
			"gpu_roofline_specs",
			"imix_dict",
			"metrics_diff",
			"metrics_explanations",
			"metrics_pct_diff",
			"prediction",
			"raw_response",
			"sass_dict",
			"source_code_files",
		],
	)


def _candidate_priority(state: Dict[str, Any], thread_id: str, model_filter: Optional[str]) -> tuple:
	gpu_key = (state.get("gpu_roofline_specs") or {}).get("dataset_gpu_key")
	llm_model_name = state.get("llm_model_name") or ""

	if model_filter:
		model_rank = 0 if llm_model_name.startswith(model_filter) else 1
	else:
		model_rank = 1
		for preferred_model in DEFAULT_MODEL_PREFERENCES:
			if llm_model_name.startswith(preferred_model) or preferred_model in thread_id:
				model_rank = 0
				break

	gpu_rank = 0 if gpu_key == CANONICAL_GPU_NAME or f"_{CANONICAL_GPU_NAME}_" in thread_id else 1
	return (gpu_rank, model_rank, thread_id)


def _collect_matching_candidates(
	parser: CheckpointDBParser,
	target: Dict[str, str],
	model_filter: Optional[str],
) -> List[Dict[str, Any]]:
	checkpoints = parser.fetch_all_checkpoints()
	tail_result = parser.fetch_tail_checkpoints_by_thread(
		checkpoints=checkpoints,
		tolerate_errors=True,
	)

	candidates: List[Dict[str, Any]] = []
	for thread_id, checkpoint_record in tail_result["tails"].items():
		state = _channel_values(checkpoint_record)
		if state.get("program_name") != target["program_name"]:
			continue
		if state.get("kernel_mangled_name") != target["kernel_mangled_name"]:
			continue
		if not _is_completed_checkpoint(checkpoint_record):
			continue

		_hydrate_channels(parser, checkpoint_record)
		state = _channel_values(checkpoint_record)
		if not state.get("sass_dict"):
			continue
		if state.get("imix_dict"):
			continue
		if not _matches_model(model_filter, state, thread_id):
			continue

		candidates.append(checkpoint_record)

	return candidates


def _select_checkpoint(
	parser: CheckpointDBParser,
	target: Dict[str, str],
	model_filter: Optional[str],
) -> Dict[str, Any]:
	candidates = _collect_matching_candidates(parser, target, model_filter)
	if not candidates:
		model_text = f" and model '{model_filter}'" if model_filter else ""
		raise RuntimeError(
			"No completed SASS-only checkpoint matched the canonical program/kernel"
			f"{model_text}."
		)

	sorted_candidates = sorted(
		candidates,
		key=lambda checkpoint_record: _candidate_priority(
			_channel_values(checkpoint_record),
			checkpoint_record["thread_id"],
			model_filter,
		),
	)

	if len(sorted_candidates) > 1:
		best_priority = _candidate_priority(
			_channel_values(sorted_candidates[0]),
			sorted_candidates[0]["thread_id"],
			model_filter,
		)
		second_priority = _candidate_priority(
			_channel_values(sorted_candidates[1]),
			sorted_candidates[1]["thread_id"],
			model_filter,
		)
		if best_priority == second_priority:
			candidate_lines = []
			for checkpoint_record in sorted_candidates:
				state = _channel_values(checkpoint_record)
				candidate_lines.append(
					f"- {checkpoint_record['thread_id']} | model={state.get('llm_model_name')}"
				)
			raise RuntimeError(
				"Multiple equally good SASS-only checkpoints matched the canonical program/kernel. "
				"Provide --modelName to disambiguate.\n" + "\n".join(candidate_lines)
			)

	return sorted_candidates[0]


def _build_prompt_listing(state: Dict[str, Any]) -> str:
	generator = DirectPromptGenerator(
		program_name=state["program_name"],
		kernel_mangled_name=state["kernel_mangled_name"],
		kernel_demangled_name=state["kernel_demangled_name"],
		source_code_files=state["source_code_files"],
		gpu_roofline_specs=state["gpu_roofline_specs"],
		compile_commands=state["compile_commands"],
		exe_args=state["exe_args"],
		sass_dict=state.get("sass_dict"),
		imix_dict=state.get("imix_dict"),
	)

	system_prompt = generator.generate_system_prompt()
	human_prompt = generator.generate_prompt()
	return "\n\n".join(
		[
			"--- SYSTEM MESSAGE ---",
			system_prompt,
			"--- HUMAN MESSAGE ---",
			human_prompt,
		]
	) + "\n"


def _response_payload(raw_response: Dict[str, Any], prediction: Dict[str, Any]) -> Any:
	tool_calls = raw_response.get("tool_calls") or []
	if tool_calls:
		first_tool_call = tool_calls[0]
		tool_args = first_tool_call.get("args")
		if tool_args:
			return tool_args

	content = raw_response.get("content")
	if isinstance(content, str) and content.strip():
		return content

	return prediction


def _serialize_response_listing(state: Dict[str, Any]) -> str:
	raw_response = state.get("raw_response") or {}
	prediction = state.get("prediction") or {}
	payload = _response_payload(raw_response, prediction)
	if isinstance(payload, str):
		return payload if payload.endswith("\n") else payload + "\n"
	return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _write_text_file(output_path: Path, content: str) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Export the canonical SASS-only prompt/response listing from PostgreSQL."
	)
	parser.add_argument(
		"--modelName",
		type=str,
		default=None,
		help="Optional model prefix used to disambiguate matching checkpoints (for example: openai/gpt-5.4).",
	)
	parser.add_argument(
		"--listing1Path",
		type=Path,
		default=DEFAULT_LISTING1_PATH,
		help="Output path for the prompt listing text file.",
	)
	parser.add_argument(
		"--listing2Path",
		type=Path,
		default=DEFAULT_LISTING2_PATH,
		help="Output path for the response listing text file.",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	dataset = _load_dataset(DATASET_PATH)
	target = _canonical_single_query_target(dataset)

	ensure_postgres_running()
	db_uri = setup_default_database()
	parser = CheckpointDBParser(db_uri)
	try:
		checkpoint_record = _select_checkpoint(parser, target, args.modelName)
		state = _channel_values(checkpoint_record)

		prompt_listing = _build_prompt_listing(state)
		response_listing = _serialize_response_listing(state)

		_write_text_file(args.listing1Path, prompt_listing)
		_write_text_file(args.listing2Path, response_listing)
		print_run_result(checkpoint_record)
	finally:
		parser.close()

	print(f"Selected thread: {checkpoint_record['thread_id']}")
	print(f"Listing 1 written to: {args.listing1Path}")
	print(f"Listing 2 written to: {args.listing2Path}")


if __name__ == "__main__":
	main()
