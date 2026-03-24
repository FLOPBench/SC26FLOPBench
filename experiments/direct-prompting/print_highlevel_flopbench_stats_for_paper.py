import argparse
import json
from pathlib import Path
from typing import Any, Dict


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = WORKSPACE_ROOT / "dataset-creation" / "gpuFLOPBench.json"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Print high-level CUDA/OpenMP code and kernel counts from gpuFLOPBench.json."
	)
	parser.add_argument(
		"--dataset",
		type=Path,
		default=DEFAULT_DATASET_PATH,
		help=f"Path to the gpuFLOPBench JSON dataset (default: {DEFAULT_DATASET_PATH})",
	)
	return parser.parse_args()


def load_dataset(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
	with dataset_path.open("r", encoding="utf-8") as infile:
		data = json.load(infile)

	if not isinstance(data, dict):
		raise ValueError("Expected the dataset JSON to be a top-level object keyed by program name.")

	return data


def runtime_from_program_name(program_name: str) -> str:
	if program_name.endswith("-cuda"):
		return "cuda"
	if program_name.endswith("-omp"):
		return "omp"
	return "other"


def percentage(count: int, total: int) -> float:
	if total == 0:
		return 0.0
	return (count / total) * 100.0


def format_count(label: str, count: int, total: int) -> str:
	return f"{label}: {count} ({percentage(count, total):.2f}% of total)"


def collect_stats(dataset: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
	stats = {
		"cuda_codes": 0,
		"omp_codes": 0,
		"cuda_kernels": 0,
		"omp_kernels": 0,
		"total_codes": len(dataset),
		"total_kernels": 0,
	}

	for program_name, program_data in dataset.items():
		runtime = runtime_from_program_name(program_name)
		kernels = program_data.get("kernels", {}) if isinstance(program_data, dict) else {}
		kernel_count = len(kernels) if isinstance(kernels, dict) else 0

		stats["total_kernels"] += kernel_count

		if runtime == "cuda":
			stats["cuda_codes"] += 1
			stats["cuda_kernels"] += kernel_count
		elif runtime == "omp":
			stats["omp_codes"] += 1
			stats["omp_kernels"] += kernel_count

	return stats


def main() -> None:
	args = parse_args()
	dataset = load_dataset(args.dataset)
	stats = collect_stats(dataset)

	print(format_count("Number of CUDA codes", stats["cuda_codes"], stats["total_codes"]))
	print(format_count("Number of OpenMP codes", stats["omp_codes"], stats["total_codes"]))
	print(format_count("Number of CUDA kernels", stats["cuda_kernels"], stats["total_kernels"]))
	print(format_count("Number of OpenMP kernels", stats["omp_kernels"], stats["total_kernels"]))
	print(f"Total number of codes: {stats['total_codes']}")
	print(f"Total number of kernels: {stats['total_kernels']}")


if __name__ == "__main__":
	main()