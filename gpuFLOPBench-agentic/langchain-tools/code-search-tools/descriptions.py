from __future__ import annotations

from pydantic import BaseModel, Field


class CudaTreeArgs(BaseModel):
    """Shared arguments for synthesizing a directory tree view."""

    dir_path: str = Field(
        ...,
        description=(
            "Directory path to render a file tree for, expressed within the filesystem portion that the tool can access (root `/`)."
        ),
    )


CUDA_FILE_TREE_DESCRIPTION = (
    "Generate an indented file tree for the provided directory. "
    "Specify the directory using the path that exists under the accessible root (e.g:`/`)."
    " Use this tool when you need a quick overview of the directory layout before diving into individual files."
)


class FunctionDefinitionListerArgs(BaseModel):
    """Arguments for listing declarations/definitions inside a specific source file."""

    file_path: str = Field(
        ...,
        description=(
            "Path to the CUDA/C++/OpenMP source file within the accessible directory tree (the entry point is `/`)."
        ),
    )


FUNCTION_DEFINITION_LISTER_DESCRIPTION = (
    "Return every function declaration or definition found in the provided source file. "
    "Refer to the file using the path that exists under the accessible root (`/`)."
    " Use this tool when you need a detailed listing of the functions a particular source file contains."
)


class DirectoryArgs(BaseModel):
    """Shared arguments for tools rooted at a benchmark directory."""

    dir_path: str = Field(
        ...,
        description=(
            "Directory path within the accessible filesystem area (entry point `/`)."
        ),
    )


CUDA_GLOBAL_FUNCTIONS_DESCRIPTION = (
    "List __global__ CUDA kernel definitions (name, file, line) inside the provided directory. "
    "Refer to the directory path reachable under the accessible root (`/`)."
    " Use this tool when you want to inventory the CUDA kernel entry points defined in that directory."
)


CUDA_MAIN_FILES_DESCRIPTION = (
    "List source files under the provided directory that define a free-function main(). "
    "Refer to directories reachable under the accessible root (`/`)."
    " Use this tool when you need to know which files act as program entry points."
)


CUDA_COMPILE_COMMANDS_DESCRIPTION = (
    "Return the compiler arguments listed in gpuFLOPBench/cuda-profiling/compile_commands.json for the provided directory. "
    "Example: cuda_compile_commands(dir_path=\"/\") where `/` is the accessible root."
    " Use this tool when you need to understand how the sources in that directory are built."
)


class KernelSourceArgs(BaseModel):
    """Arguments for fetching the source code of a specific kernel."""

    file_path: str = Field(
        ...,
        description=(
            "Path to the file or directory to analyze within the accessible filesystem area (entry point `/`)."
        ),
    )
    kernel_name: str = Field(
        ...,
        description="Name of the __global__ CUDA kernel to extract.",
        min_length=1,
    )


EXTRACT_KERNEL_SOURCE_DESCRIPTION = (
    "Return the source code for a specific __global__ CUDA kernel within the provided file or directory. "
    "Use the path that exists under the accessible root (`/`)."
    " Use this tool when you want the canonical source of a kernel for comparison or review."
)


class IncludeTreeArgs(BaseModel):
    """Arguments for walking the include tree of a specific source file."""

    file_path: str = Field(
        ...,
        description=(
            "Path to the CUDA/C++/OpenMP source file to analyze inside the accessible filesystem area (entry point `/`)."
        ),
        min_length=1,
    )


INCLUDE_TREE_DESCRIPTION = (
    "Walk the #include hierarchy for a specific source file inside a benchmark, "
    "annotating missing files (DNE) and stopping recursion when loops are detected. "
    "Refer to the file using the path that exists under the accessible root (`/`)."
    " Use this tool when you need to trace the include dependencies for that source file."
)
