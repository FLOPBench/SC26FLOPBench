from __future__ import annotations

from pydantic import BaseModel, Field


class CudaTreeArgs(BaseModel):
    """Shared arguments for synthesizing a directory tree view."""

    dir_path: str = Field(
        ...,
        description=(
            "Absolute directory path or virtual FilesystemBackend path (e.g., `/lulesh-cuda`) "
            "to render a CUDA file tree for."
        ),
    )


CUDA_FILE_TREE_DESCRIPTION = (
    "Generate an indented file tree for the provided directory. "
    "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
)


class FunctionDefinitionListerArgs(BaseModel):
    """Arguments for listing declarations/definitions inside a specific source file."""

    file_path: str = Field(
        ...,
        description=(
            "Absolute path to the CUDA/C++ file on disk, or the virtual path that "
            "the FilesystemBackend exposes (e.g., `/lulesh-cuda/lulesh.cu`)."
        ),
    )


FUNCTION_DEFINITION_LISTER_DESCRIPTION = (
    "Return every function declaration or definition found in the provided CUDA/C++/header file. "
    "Pass either an actual disk path or the virtual FilesystemBackend path (e.g., `/lulesh-cuda/lulesh.cu`)."
)


class DirectoryArgs(BaseModel):
    """Shared arguments for Tools rooted at a CUDA benchmark directory."""

    dir_path: str = Field(
        ...,
        description=(
            "Absolute disk path or virtual FilesystemBackend path "
            "(e.g., `/lulesh-cuda`) to a directory that lives under gpuFLOPBench/src."
        ),
    )


CUDA_GLOBAL_FUNCTIONS_DESCRIPTION = (
    "List __global__ CUDA kernel definitions (name, file, line) inside the provided directory. "
    "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
)


CUDA_MAIN_FILES_DESCRIPTION = (
    "List source files under the provided directory that define a free-function main(). "
    "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
)


CUDA_COMPILE_COMMANDS_DESCRIPTION = (
    "Return the compiler arguments listed in gpuFLOPBench/cuda-profiling/compile_commands.json for the provided CUDA directory. "
    "Example: cuda_compile_commands(dir_path=\"/lulesh-cuda\")."
)


class KernelSourceArgs(BaseModel):
    """Arguments for fetching the source code of a specific CUDA kernel."""

    file_path: str = Field(
        ...,
        description=(
            "Absolute disk path or virtual FilesystemBackend path (e.g., `/lulesh-cuda/lulesh.cu` or `/lulesh-cuda`). "
            "That identifies the file or directory to analyze."
        ),
    )
    kernel_name: str = Field(
        ...,
        description="Name of the __global__ CUDA kernel to extract.",
        min_length=1,
    )


EXTRACT_KERNEL_SOURCE_DESCRIPTION = (
    "Return the source code for a specific __global__ kernel within the provided file or directory. "
    "Pass an absolute disk path or FilesystemBackend path (e.g., `/lulesh-cuda/lulesh.cu`)."
)


class IncludeTreeArgs(BaseModel):
    """Arguments for walking the include tree of a specific source file."""

    file_path: str = Field(
        ...,
        description=(
            "Absolute disk path or virtual FilesystemBackend path (e.g., `/lulesh-cuda/lulesh.cu`) "
            "to the CUDA/C++ file to analyze."
        ),
        min_length=1,
    )


INCLUDE_TREE_DESCRIPTION = (
    "Walk the #include hierarchy for a specific CUDA/C++ file inside a *-cuda benchmark, "
    "annotating missing files (DNE) and stopping recursion when loops are detected. "
    "Pass an absolute disk path or a FilesystemBackend path."
)
