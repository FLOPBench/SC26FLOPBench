# The following was generated from the NVIDIA SASS documentation:
# https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvidia-ampere-gpu-and-ada-instruction-set
# https://docs.nvidia.com/cuda/cuda-binary-utilities/#hopper-instruction-set


import re
from typing import Optional, List

# SASS opcode metadata for:
# - Ampere + Ada (SM 80, 86, 89)
# - Hopper (SM 90)

AMPERE_ADA = [80, 86, 89]
HOPPER = [90]
AMPERE_ADA_HOPPER = [80, 86, 89, 90]

def _mk(
    description,
    *,
    data_type=None,          # e.g., FP32 / FP16 / FP64 / INT / BIT / PRED / FP8 / None
    op_type=None,            # e.g., movement / integer / floating point / uniform datapath / control / load/store / predicate / synchronization / texture / surface / tensor memory access / warpgroup / conversion / miscellaneous
    datapath=None,           # vector / uniform / predicate / uniform_predicate / texture / surface / tensor / control / None
    address_space=None,      # global / shared / local / constant / texture / surface / generic / distributed_shared / None
    dst_address_space=None,  # for copy-ish ops (e.g., global->shared), else None
    access_operation=None,   # load / store / atomic / None  (reductions/barriers go into memory_effect)
    memory_effect=None,      # reduction / cache_control / barrier / dependency_barrier / fence / sync / query / copy / prefetch / flush / None
    is_async=None,           # True/False/None
    sm_arch=None,            # list like [80,86,89] or [90] or [80,86,89,90]
    notes=None,
):
    return {
        "description": description,
        "data_type": data_type,
        "op_type": op_type,
        "datapath": datapath,
        "address_space": address_space,
        "dst_address_space": dst_address_space,
        "access_operation": access_operation,
        "memory_effect": memory_effect,
        "is_async": is_async,
        "sm_arch": sm_arch,
        "notes": notes,
    }

SASS_INSTR_METADATA = {}

def add_group(op_to_desc, **meta):
    for op, desc in op_to_desc.items():
        if op in SASS_INSTR_METADATA:
            raise KeyError(f"Duplicate opcode: {op}")
        SASS_INSTR_METADATA[op] = _mk(desc, **meta)

# ---------------------------
# Floating point (common)
# ---------------------------
add_group(
    {
        "FADD": "FP32 Add",
        "FADD32I": "FP32 Add",
        "FCHK": "Floating-point Range Check",
        "FFMA32I": "FP32 Fused Multiply and Add",
        "FFMA": "FP32 Fused Multiply and Add",
        "FMNMX": "FP32 Minimum/Maximum",
        "FMUL": "FP32 Multiply",
        "FMUL32I": "FP32 Multiply",
        "FSEL": "Floating Point Select",
        "FSET": "FP32 Compare And Set",
        "FSETP": "FP32 Compare And Set Predicate",
        "FSWZADD": "FP32 Swizzle Add",
        "MUFU": "FP32 Multi Function Operation",
    },
    data_type="FP32",
    op_type="floating point",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
)

add_group(
    {
        "HADD2": "FP16 Add",
        "HADD2_32I": "FP16 Add",
        "HFMA2": "FP16 Fused Mutiply Add",
        "HFMA2_32I": "FP16 Fused Mutiply Add",
        "HMMA": "Matrix Multiply and Accumulate",
        "HMNMX2": "FP16 Minimum / Maximum",
        "HMUL2": "FP16 Multiply",
        "HMUL2_32I": "FP16 Multiply",
        "HSET2": "FP16 Compare And Set",
        "HSETP2": "FP16 Compare And Set Predicate",
    },
    data_type="FP16",
    op_type="floating point",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
    notes="HMMA is TensorCore MMA; exact operand/accumulator types depend on encoding.",
)

add_group(
    {
        "DADD": "FP64 Add",
        "DFMA": "FP64 Fused Mutiply Add",
        "DMMA": "Matrix Multiply and Accumulate",
        "DMUL": "FP64 Multiply",
        "DSETP": "FP64 Compare And Set Predicate",
    },
    data_type="FP64",
    op_type="floating point",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
    notes="DMMA is listed under FP64 section; exact MMA operand types depend on encoding.",
)

# ---------------------------
# Integer (common) + Hopper SIMD integer adds
# ---------------------------
add_group(
    {
        "BMMA": "Bit Matrix Multiply and Accumulate",
        "BMSK": "Bitfield Mask",
        "BREV": "Bit Reverse",
        "FLO": "Find Leading One",
        "IABS": "Integer Absolute Value",
        "IADD": "Integer Addition",
        "IADD3": "3-input Integer Addition",
        "IADD32I": "Integer Addition",
        "IDP": "Integer Dot Product and Accumulate",
        "IDP4A": "Integer Dot Product and Accumulate",
        "IMAD": "Integer Multiply And Add",
        "IMMA": "Integer Matrix Multiply and Accumulate",
        "IMNMX": "Integer Minimum/Maximum",
        "IMUL": "Integer Multiply",
        "IMUL32I": "Integer Multiply",
        "ISCADD": "Scaled Integer Addition",
        "ISCADD32I": "Scaled Integer Addition",
        "ISETP": "Integer Compare And Set Predicate",
        "LEA": "LOAD Effective Address",
        "LOP": "Logic Operation",
        "LOP3": "Logic Operation",
        "LOP32I": "Logic Operation",
        "POPC": "Population count",
        "SHF": "Funnel Shift",
        "SHL": "Shift Left",
        "SHR": "Shift Right",
        "VABSDIFF": "Absolute Difference",
        "VABSDIFF4": "Absolute Difference",
    },
    data_type="INT",
    op_type="integer",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
    notes="Some ops may support multiple widths/types depending on encoding.",
)

# Hopper-only SIMD (vector) integer/FP16 minmax family
add_group(
    {
        "VHMNMX": "SIMD FP16 3-Input Minimum / Maximum",
        "VIADD": "SIMD Integer Addition",
        "VIADDMNMX": "SIMD Integer Addition and Fused Min/Max Comparison",
        "VIMNMX": "SIMD Integer Minimum / Maximum",
        "VIMNMX3": "SIMD Integer 3-Input Minimum / Maximum",
    },
    data_type=None,  # mixed/encoding-dependent; description indicates FP16 for VHMNMX
    op_type="integer",
    datapath="vector",
    sm_arch=HOPPER,
    notes="VHMNMX is FP16-oriented per description; others are SIMD integer-family ops.",
)

# ---------------------------
# Conversion (common)
# ---------------------------
add_group(
    {
        "F2F": "Floating Point To Floating Point Conversion",
        "F2I": "Floating Point To Integer Conversion",
        "F2FP": "FP32 Convert and Pack to FP16/BF16",
        "I2F": "Integer To Floating Point Conversion",
        "I2I": "Integer To Integer Conversion",
        "I2IP": "Integer To Integer Conversion and Packing",
        "I2FP": "Integer to FP32 Convert and Pack",
        "F2IP": "FP32 Down-Convert to Integer and Pack",
        "FRND": "Round To Integer",
    },
    data_type=None,
    op_type="conversion",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
)

# ---------------------------
# Movement (common)
# ---------------------------
add_group(
    {
        "MOV": "Move",
        "MOV32I": "Move",
        "MOVM": "Move Matrix with Transposition or Expansion",
        "PRMT": "Permute Register Pair",
        "SEL": "Select Source with Predicate",
        "SGXT": "Sign Extend",
        "SHFL": "Warp Wide Register Shuffle",
    },
    data_type=None,
    op_type="movement",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
)

# ---------------------------
# Predicate (common)
# ---------------------------
add_group(
    {
        "PLOP3": "Predicate Logic Operation",
        "PSETP": "Combine Predicates and Set Predicate",
        "P2R": "Move Predicate Register To Register",
        "R2P": "Move Register To Predicate Register",
    },
    data_type="PRED",
    op_type="predicate",
    datapath="predicate",
    sm_arch=AMPERE_ADA_HOPPER,
)

# ---------------------------
# Load/Store / Memory system (Ampere/Ada + Hopper)
# ---------------------------
# Ampere/Ada + Hopper common loads/stores
add_group(
    {"LD": "Load from generic Memory"},
    op_type="load/store",
    datapath="vector",
    address_space="generic",
    access_operation="load",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"LDC": "Load Constant"},
    op_type="load/store",
    datapath="vector",
    address_space="constant",
    access_operation="load",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"LDG": "Load from Global Memory"},
    op_type="load/store",
    datapath="vector",
    address_space="global",
    access_operation="load",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"LDL": "Load within Local Memory Window"},
    op_type="load/store",
    datapath="vector",
    address_space="local",
    access_operation="load",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"LDS": "Load within Shared Memory Window"},
    op_type="load/store",
    datapath="vector",
    address_space="shared",
    access_operation="load",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"LDSM": "Load Matrix from Shared Memory with Element Size Expansion"},
    op_type="load/store",
    datapath="vector",
    address_space="shared",
    access_operation="load",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"ST": "Store to Generic Memory"},
    op_type="load/store",
    datapath="vector",
    address_space="generic",
    access_operation="store",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"STG": "Store to Global Memory"},
    op_type="load/store",
    datapath="vector",
    address_space="global",
    access_operation="store",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"STL": "Store to Local Memory"},
    op_type="load/store",
    datapath="vector",
    address_space="local",
    access_operation="store",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"STS": "Store to Shared Memory"},
    op_type="load/store",
    datapath="vector",
    address_space="shared",
    access_operation="store",
    sm_arch=AMPERE_ADA_HOPPER,
)

# Atomics / reductions
add_group(
    {"ATOM": "Atomic Operation on Generic Memory"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="generic",
    access_operation="atomic",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"ATOMS": "Atomic Operation on Shared Memory"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="shared",
    access_operation="atomic",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"ATOMG": "Atomic Operation on Global Memory"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="global",
    access_operation="atomic",
    sm_arch=AMPERE_ADA_HOPPER,
)

# Ampere/Ada only reduction opcode name (Hopper uses REDG instead)
add_group(
    {"RED": "Reduction Operation on Generic Memory"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="generic",
    access_operation=None,
    memory_effect="reduction",
    sm_arch=AMPERE_ADA,
)

# Common cross-thread/group-ish utilities listed under load/store
add_group(
    {"MATCH": "Match Register Values Across Thread Group"},
    data_type=None,
    op_type="synchronization",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"QSPC": "Query Space"},
    data_type=None,
    op_type="control",
    datapath="vector",
    memory_effect="query",
    sm_arch=AMPERE_ADA_HOPPER,
)

# Cache / barrier / memory ordering
add_group(
    {
        "CCTL": "Cache Control",
        "CCTLL": "Cache Control",
        "CCTLT": "Texture Cache Control",
    },
    data_type=None,
    op_type="load/store",
    datapath="vector",
    access_operation=None,
    memory_effect="cache_control",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"LDGDEPBAR": "Global Load Dependency Barrier"},
    data_type=None,
    op_type="synchronization",
    datapath="vector",
    address_space="global",
    memory_effect="dependency_barrier",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"ERRBAR": "Error Barrier"},
    data_type=None,
    op_type="synchronization",
    datapath="vector",
    memory_effect="barrier",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"MEMBAR": "Memory Barrier"},
    data_type=None,
    op_type="synchronization",
    datapath="vector",
    memory_effect="barrier",
    sm_arch=AMPERE_ADA_HOPPER,
)

# Async global->shared copy (both)
add_group(
    {"LDGSTS": "Asynchronous Global to Shared Memcopy"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="global",
    dst_address_space="shared",
    access_operation=None,
    memory_effect="copy",
    is_async=True,
    sm_arch=AMPERE_ADA_HOPPER,
)

# Hopper-only load/store additions
add_group(
    {"FENCE": "Memory Visibility Guarantee for Shared or Global Memory"},
    data_type=None,
    op_type="synchronization",
    datapath="vector",
    address_space=None,
    access_operation=None,
    memory_effect="fence",
    sm_arch=HOPPER,
)
add_group(
    {"LDGMC": "Reducing Load"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="global",
    access_operation="load",
    memory_effect="reduction",
    sm_arch=HOPPER,
)
add_group(
    {"STSM": "Store Matrix to Shared Memory"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="shared",
    access_operation="store",
    sm_arch=HOPPER,
)
add_group(
    {"STAS": "Asynchronous Store to Distributed Shared Memory With Explicit Synchronization"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="distributed_shared",
    access_operation="store",
    memory_effect="sync",
    is_async=True,
    sm_arch=HOPPER,
)
add_group(
    {"SYNCS": "Sync Unit"},
    data_type=None,
    op_type="synchronization",
    datapath="vector",
    memory_effect="sync",
    sm_arch=HOPPER,
)
add_group(
    {"REDAS": "Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="distributed_shared",
    access_operation=None,
    memory_effect="reduction",
    is_async=True,
    sm_arch=HOPPER,
)
add_group(
    {"REDG": "Reduction Operation on Generic Memory"},
    data_type=None,
    op_type="load/store",
    datapath="vector",
    address_space="generic",
    access_operation=None,
    memory_effect="reduction",
    sm_arch=HOPPER,
)

# ---------------------------
# Uniform datapath (Ampere/Ada + Hopper)
# ---------------------------
add_group(
    {
        "R2UR": "Move from Vector Register to a Uniform Register",
        "REDUX": "Reduction of a Vector Register into a Uniform Register",
        "S2UR": "Move Special Register to Uniform Register",
        "UBMSK": "Uniform Bitfield Mask",
        "UBREV": "Uniform Bit Reverse",
        "UCLEA": "Load Effective Address for a Constant",
        "UF2FP": "Uniform FP32 Down-convert and Pack",
        "UFLO": "Uniform Find Leading One",
        "UIADD3": "Uniform Integer Addition",
        "UIADD3.64": "Uniform Integer Addition",
        "UIMAD": "Uniform Integer Multiplication",
        "UISETP": "Integer Compare and Set Uniform Predicate",
        "ULDC": "Load from Constant Memory into a Uniform Register",
        "ULEA": "Uniform Load Effective Address",
        "ULOP": "Logic Operation",
        "ULOP3": "Logic Operation",
        "ULOP32I": "Logic Operation",
        "UMOV": "Uniform Move",
        "UP2UR": "Uniform Predicate to Uniform Register",
        "UPLOP3": "Uniform Predicate Logic Operation",
        "UPOPC": "Uniform Population Count",
        "UPRMT": "Uniform Byte Permute",
        "UPSETP": "Uniform Predicate Logic Operation",
        "UR2UP": "Uniform Register to Uniform Predicate",
        "USEL": "Uniform Select",
        "USGXT": "Uniform Sign Extend",
        "USHF": "Uniform Funnel Shift",
        "USHL": "Uniform Left Shift",
        "USHR": "Uniform Right Shift",
        "VOTEU": "Voting across SIMD Thread Group with Results in Uniform Destination",
    },
    data_type=None,
    op_type="uniform datapath",
    datapath="uniform",
    sm_arch=AMPERE_ADA_HOPPER,
)

# More specific uniform memory semantics
SASS_INSTR_METADATA["ULDC"]["address_space"] = "constant"
SASS_INSTR_METADATA["ULDC"]["access_operation"] = "load"
SASS_INSTR_METADATA["REDUX"]["memory_effect"] = "reduction"

# Hopper-only uniform additions
add_group(
    {
        "UCGABAR_ARV": "CGA Barrier Synchronization",
        "UCGABAR_WAIT": "CGA Barrier Synchronization",
        "ULEPC": "Uniform Load Effective PC",
        "USETMAXREG": "Release, Deallocate and Allocate Registers",
    },
    data_type=None,
    op_type="uniform datapath",
    datapath="uniform",
    sm_arch=HOPPER,
    notes="Uniform CGA barrier and register management are Hopper additions per table.",
)

# ---------------------------
# Texture (common)
# ---------------------------
add_group(
    {
        "TEX": "Texture Fetch",
        "TLD": "Texture Load",
        "TLD4": "Texture Load 4",
        "TMML": "Texture MipMap Level",
        "TXD": "Texture Fetch With Derivatives",
        "TXQ": "Texture Query",
    },
    data_type=None,
    op_type="texture",
    datapath="texture",
    address_space="texture",
    access_operation=None,
    sm_arch=AMPERE_ADA_HOPPER,
)

# ---------------------------
# Surface (common)
# ---------------------------
add_group(
    {"SUATOM": "Atomic Op on Surface Memory"},
    data_type=None,
    op_type="surface",
    datapath="surface",
    address_space="surface",
    access_operation="atomic",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"SULD": "Surface Load"},
    data_type=None,
    op_type="surface",
    datapath="surface",
    address_space="surface",
    access_operation="load",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"SUST": "Surface Store"},
    data_type=None,
    op_type="surface",
    datapath="surface",
    address_space="surface",
    access_operation="store",
    sm_arch=AMPERE_ADA_HOPPER,
)
add_group(
    {"SURED": "Reduction Op on Surface Memory"},
    data_type=None,
    op_type="surface",
    datapath="surface",
    address_space="surface",
    access_operation=None,
    memory_effect="reduction",
    sm_arch=AMPERE_ADA_HOPPER,
)

# ---------------------------
# Control (Ampere/Ada + Hopper) + Hopper-only control additions
# ---------------------------
add_group(
    {
        "BMOV": "Move Convergence Barrier State",
        "BPT": "BreakPoint/Trap",
        "BRA": "Relative Branch",
        "BREAK": "Break out of the Specified Convergence Barrier",
        "BRX": "Relative Branch Indirect",
        "BRXU": "Relative Branch with Uniform Register Based Offset",
        "BSSY": "Barrier Set Convergence Synchronization Point",
        "BSYNC": "Synchronize Threads on a Convergence Barrier",
        "CALL": "Call Function",
        "EXIT": "Exit Program",
        "JMP": "Absolute Jump",
        "JMX": "Absolute Jump Indirect",
        "JMXU": "Absolute Jump with Uniform Register Based Offset",
        "KILL": "Kill Thread",
        "NANOSLEEP": "Suspend Execution",
        "RET": "Return From Subroutine",
        "RPCMOV": "PC Register Move",
        "WARPSYNC": "Synchronize Threads in Warp",
        "YIELD": "Yield Control",
    },
    data_type=None,
    op_type="control",
    datapath="control",
    sm_arch=AMPERE_ADA_HOPPER,
)

# Hopper-only control
add_group(
    {
        "ACQBULK": "Wait for Bulk Release Status Warp State",
        "CGAERRBAR": "CGA Error Barrier",
        "ELECT": "Elect a Leader Thread",
        "ENDCOLLECTIVE": "Reset the MCOLLECTIVE mask",
        "PREEXIT": "Dependent Task Launch Hint",
    },
    data_type=None,
    op_type="control",
    datapath="control",
    sm_arch=HOPPER,
)

# Some control ops are primarily synchronization-like
for _op in ["BMOV", "BSSY", "BSYNC", "WARPSYNC", "CGAERRBAR", "ACQBULK"]:
    if _op in SASS_INSTR_METADATA:
        SASS_INSTR_METADATA[_op]["op_type"] = "synchronization"
        SASS_INSTR_METADATA[_op]["memory_effect"] = SASS_INSTR_METADATA[_op]["memory_effect"] or "sync"

# ---------------------------
# Miscellaneous (common)
# ---------------------------
add_group(
    {
        "B2R": "Move Barrier To Register",
        "BAR": "Barrier Synchronization",
        "CS2R": "Move Special Register to Register",
        "DEPBAR": "Dependency Barrier",
        "GETLMEMBASE": "Get Local Memory Base Address",
        "LEPC": "Load Effective PC",
        "NOP": "No Operation",
        "PMTRIG": "Performance Monitor Trigger",
        "S2R": "Move Special Register to Register",
        "SETCTAID": "Set CTA ID",
        "SETLMEMBASE": "Set Local Memory Base Address",
        "VOTE": "Vote Across SIMT Thread Group",
    },
    data_type=None,
    op_type="miscellaneous",
    datapath="vector",
    sm_arch=AMPERE_ADA_HOPPER,
)

# Refine a few misc types
SASS_INSTR_METADATA["BAR"]["op_type"] = "synchronization"
SASS_INSTR_METADATA["BAR"]["memory_effect"] = "barrier"
SASS_INSTR_METADATA["DEPBAR"]["op_type"] = "synchronization"
SASS_INSTR_METADATA["DEPBAR"]["memory_effect"] = "dependency_barrier"
SASS_INSTR_METADATA["NOP"]["op_type"] = "control"
SASS_INSTR_METADATA["VOTE"]["op_type"] = "predicate"
SASS_INSTR_METADATA["VOTE"]["data_type"] = "PRED"

# ---------------------------
# Warpgroup (Hopper-only)
# ---------------------------
add_group(
    {
        "BGMMA": "Bit Matrix Multiply and Accumulate Across Warps",
        "HGMMA": "Matrix Multiply and Accumulate Across a Warpgroup",
        "IGMMA": "Integer Matrix Multiply and Accumulate Across a Warpgroup",
        "QGMMA": "FP8 Matrix Multiply and Accumulate Across a Warpgroup",
        "WARPGROUP": "Warpgroup Synchronization",
        "WARPGROUPSET": "Set Warpgroup Counters",
    },
    data_type=None,
    op_type="warpgroup",
    datapath="warpgroup",
    sm_arch=HOPPER,
)
SASS_INSTR_METADATA["BGMMA"]["data_type"] = "BIT"
SASS_INSTR_METADATA["IGMMA"]["data_type"] = "INT"
SASS_INSTR_METADATA["QGMMA"]["data_type"] = "FP8"
SASS_INSTR_METADATA["WARPGROUP"]["op_type"] = "synchronization"
SASS_INSTR_METADATA["WARPGROUP"]["memory_effect"] = "sync"
SASS_INSTR_METADATA["WARPGROUPSET"]["op_type"] = "control"

# ---------------------------
# Tensor Memory Access (Hopper-only)
# ---------------------------
add_group(
    {
        "UBLKCP": "Bulk Data Copy",
        "UBLKPF": "Bulk Data Prefetch",
        "UBLKRED": "Bulk Data Copy from Shared Memory with Reduction",
        "UTMACCTL": "TMA Cache Control",
        "UTMACMDFLUSH": "TMA Command Flush",
        "UTMALDG": "Tensor Load from Global to Shared Memory",
        "UTMAPF": "Tensor Prefetch",
        "UTMAREDG": "Tensor Store from Shared to Global Memory with Reduction",
        "UTMASTG": "Tensor Store from Shared to Global Memory",
    },
    data_type=None,
    op_type="tensor memory access",
    datapath="tensor",
    sm_arch=HOPPER,
)

# Refine TMA/bulk semantics
SASS_INSTR_METADATA["UBLKCP"]["memory_effect"] = "copy"
SASS_INSTR_METADATA["UBLKCP"]["is_async"] = True
SASS_INSTR_METADATA["UBLKPF"]["memory_effect"] = "prefetch"
SASS_INSTR_METADATA["UBLKPF"]["is_async"] = True
SASS_INSTR_METADATA["UBLKRED"]["memory_effect"] = "copy+reduction"
SASS_INSTR_METADATA["UBLKRED"]["is_async"] = True
SASS_INSTR_METADATA["UTMACCTL"]["memory_effect"] = "cache_control"
SASS_INSTR_METADATA["UTMACMDFLUSH"]["memory_effect"] = "flush"
SASS_INSTR_METADATA["UTMALDG"]["address_space"] = "global"
SASS_INSTR_METADATA["UTMALDG"]["dst_address_space"] = "shared"
SASS_INSTR_METADATA["UTMALDG"]["memory_effect"] = "copy"
SASS_INSTR_METADATA["UTMALDG"]["is_async"] = True
SASS_INSTR_METADATA["UTMAPF"]["memory_effect"] = "prefetch"
SASS_INSTR_METADATA["UTMAPF"]["is_async"] = True
SASS_INSTR_METADATA["UTMAREDG"]["address_space"] = "shared"
SASS_INSTR_METADATA["UTMAREDG"]["dst_address_space"] = "global"
SASS_INSTR_METADATA["UTMAREDG"]["memory_effect"] = "copy+reduction"
SASS_INSTR_METADATA["UTMAREDG"]["is_async"] = True
SASS_INSTR_METADATA["UTMASTG"]["address_space"] = "shared"
SASS_INSTR_METADATA["UTMASTG"]["dst_address_space"] = "global"
SASS_INSTR_METADATA["UTMASTG"]["memory_effect"] = "copy"
SASS_INSTR_METADATA["UTMASTG"]["is_async"] = True

# ---------------------------
# Sanity: SASS_INSTR_METADATA is the requested opcode->metadata dict
# ---------------------------
# Example:
#   SASS_INSTR_METADATA["LDG"]
#   SASS_INSTR_METADATA["QGMMA"]


_GUARD_RE = re.compile(
    r"""
    ^\s*
    (?:/\*.*?\*/\s*)?              # optional leading /*...*/ comment (address/encoding)
    @!?                           # @ or @!
    (?:P\d+|UP\d+|PT|UPT)          # P0.., UP0.., PT, UPT (extend if needed)
    \b
    """,
    re.VERBOSE,
)

# Capture only the base mnemonic (no dot modifiers)
_BASE_OPCODE_RE = re.compile(
    r"""
    ^\s*
    (?:/\*.*?\*/\s*)?                 # optional leading /*...*/ comment
    (?:@!?(?:P\d+|UP\d+|PT|UPT)\s+)?  # optional guard predicate + whitespace
    (?P<opcode>[A-Z][A-Z0-9_]*)       # base mnemonic only
    \b
    """,
    re.VERBOSE,
)


def extract_opcode_from_line(src_str: str) -> Optional[str]:
    """
    Extract base mnemonic from a SASS line (e.g., 'LDG' from 'LDG.E.CONSTANT ...').
    Returns None if no mnemonic is found.
    """
    if src_str is None:
        return None
    m = _BASE_OPCODE_RE.match(str(src_str))
    return m.group("opcode") if m else None


def detect_guard_pred_instruction(src_str: str) -> bool:
    """
    True if the line begins with a guard predicate like @P0 / @!P0 / @PT / @UP0.
    """
    if src_str is None:
        return False
    return bool(_GUARD_RE.match(str(src_str)))



# pandas usage:
# df["opcode"] = df["Source"].apply(extract_opcode_from_line)
# df["is_predicated"] = df["Source"].apply(detect_guard_pred_instruction)

# "0x" followed by at least 8 hex digits (case-insensitive)
_HEX_REF_RE = re.compile(r"\b0x[0-9a-fA-F]{8,}\b")

def extract_hex_references(src_str: str) -> List[str]:
    """
    Return a list of hex addresses referenced in a SASS line.
    Matches tokens like 0x7563ef27bf00 (0x + >= 8 hex digits).
    Returns [] if none are found.
    """
    if src_str is None:
        return []
    refs = _HEX_REF_RE.findall(str(src_str))

    return ':'.join(refs) if refs else ''
