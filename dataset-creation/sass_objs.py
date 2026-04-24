
import os

import re
import sys
from collections import defaultdict, deque

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "cuda-profiling"))

# Import from user provided utils and sass_helper
from utils import *
from sass_helper import SASS_INSTR_METADATA, extract_opcode_from_line, detect_guard_pred_instruction


class SASSTextSection:
    def __init__(self, filename, line_number, mangled_name, raw_text):
        self.filename = filename
        self.line_number = line_number
        self.mangled_name = mangled_name
        self.raw_text = raw_text
        self.imix = defaultdict(int)
        
        # Metadata fields
        self.num_labels = 0
        self.labels = []
        self.num_references = 0
        self.num_self_references = 0
        self.references = []
        self.num_predicated_guards = 0
        self.num_lines = 0
        self.num_math_ops_with_constant = 0
        self.num_fp16 = 0
        self.num_fp32 = 0
        self.num_fp64 = 0
        self.num_global_loads = 0
        self.num_global_stores = 0
        
        # High-level categorical counts
        self.op_type_counts = defaultdict(int)
        self.access_op_counts = defaultdict(int)
        self.address_space_counts = defaultdict(int)
        self.data_type_counts = defaultdict(int)

        self._parse_section()

    def _parse_section(self):
        lines = self.raw_text.splitlines()
        
        # Regex for identifying lines with instructions or labels
        # Assuming lines starting with '/*' are instructions
        # Labels start with '.' and end with ':'
        label_pattern = re.compile(r"^\s*(\.L_x_\d+):")
        
        # Regex for capturing references in `(name) format. 
        # Example: BRA `(.L_x_14); or CALL `(_Z3...);
        ref_pattern = re.compile(r"`\(([^)]+)\)")
        
        # Regex for capturing immediate constants (hex, integer, float)
        # Exclude register names like R1, UR2, SR_...
        # Heuristic: looks for numbers that are not part of identifiers.
        # But instructions like FADD R1, R2, R3 have no constants.
        # IMAD R1, R2, 0x4, R3 has constant.
        # Constant memory: c[0x0][0x28]
        const_mem_pattern = re.compile(r"c\[")
        immediate_pattern = re.compile(r"(?<![a-zA-Z_])(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![a-zA-Z_])")
        # Note: the immediate pattern might match '0' in 'R0' if not careful. 
        # R0 -> R is [a-zA-Z_], so (?<!...) handles it. 
        # But 'R0' -> '0' is preceded by 'R'. Correct.
        # 'P0' -> '0' is preceded by 'P'. Correct.
        # 'c[0x0]' -> matches 0x0. Correct. 
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 1. Check for Label
            label_match = label_pattern.search(line)
            if label_match:
                self.num_labels += 1
                self.labels.append(label_match.group(1))
                continue # Labels are not instructions

            # 2. Check for Instruction (lines starting with /* offset */)
            if line.startswith("/*"):
                self.num_lines += 1
                
                # Extract Opcode
                opcode = extract_opcode_from_line(line)
                
                # Check for validity and skip if not found? 
                # User requirement: "An assert should be made to make sure that all the instructions encountered are valid instructions."
                # Some typical directive lines might slip in? But we are filtering by starting with /*
                
                if opcode:
                    assert opcode in SASS_INSTR_METADATA, f"Unknown instruction '{opcode}' found in {self.filename}:{self.line_number} in kernel {self.mangled_name}. Line: {line}"
                    self.imix[opcode] += 1
                    
                    # Metadata Extraction
                    meta = SASS_INSTR_METADATA[opcode]
                    
                    # Predicated Guards
                    if detect_guard_pred_instruction(line):
                        self.num_predicated_guards += 1
                        
                    # Op Type / Access / Address Space
                    if meta.get("op_type"):
                        self.op_type_counts[meta["op_type"]] += 1
                    if meta.get("access_operation"):
                        self.access_op_counts[meta["access_operation"]] += 1
                    if meta.get("address_space"):
                        self.address_space_counts[meta["address_space"]] += 1
                    if meta.get("data_type"):
                        self.data_type_counts[meta["data_type"]] += 1

                    # Specific Counters
                    if meta.get("data_type") == "FP16":
                        self.num_fp16 += 1
                    elif meta.get("data_type") == "FP32":
                        self.num_fp32 += 1
                    elif meta.get("data_type") == "FP64":
                        self.num_fp64 += 1
                        
                    is_global = meta.get("address_space") == "global"
                    is_load = meta.get("access_operation") == "load"
                    is_store = meta.get("access_operation") == "store"
                    
                    if is_global and is_load:
                        self.num_global_loads += 1
                    if is_global and is_store:
                        self.num_global_stores += 1
                        
                    # Math ops with constants
                    if meta.get("op_type") in ["floating point", "integer"]:
                        # Check for constant in line (excluding opcode part ideally, but line check is okay)
                        # We need to look at the arguments part. 
                        # The line format is: /*offset*/ (@P0)? OPCODE Args... ;
                        # Let's simple check if immediate pattern matches
                        has_const = False
                        if const_mem_pattern.search(line):
                            has_const = True
                        else:
                            # Check immediates
                            # We need to avoid counting offset /*0020*/ as constant.
                            # Remove comments first
                            line_no_comment = re.sub(r"/\*.*?\*/", "", line)
                            if immediate_pattern.search(line_no_comment):
                                has_const = True
                        
                        if has_const:
                            self.num_math_ops_with_constant += 1

                    # References to other sections
                    # Look for `(name)
                    # Exclude references to labels which we collected? 
                    # "Number of references to other .text sections"
                    # Labels are local. Functions are external (other sections).
                    # A reference to a label usually starts with .L
                    # A reference to a function usually does NOT start with .L (except maybe some mangled names?)
                    # Let's capture all references first
                    refs = ref_pattern.findall(line)
                    for r in refs:
                        # Filter out data references (start with $) or expressions (start with ()
                        if r.startswith('$') or r.startswith('('):
                            continue

                        if not r.startswith(".L"):
                            self.num_references += 1
                            self.references.append(r)
                            if r == self.mangled_name:
                                self.num_self_references += 1
                            
                else:
                    # Should we warn? Some lines might be comment lines inside text block?
                    # The block is composed of assembly lines.
                    pass


class SASSFileParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        
        # Parse filename parts: {programName}-{cuda|omp}_sm_{80|86|90}.sass
        # Example: accuracy-cuda_sm_80.sass
        # Regex: ^(.*)-(cuda|omp)_(sm_\d+)\.sass$
        match = re.match(r"^(.*)-(cuda|omp)_(sm_\d+)\.sass$", self.filename)
        if match:
            self.program_name = match.group(1)
            self.model = match.group(2) # cuda or omp
            self.sm_arch = match.group(3)
        else:
            # Fallback or error?
            self.program_name = "unknown"
            self.model = "unknown"
            self.sm_arch = "unknown"

        self.text_sections = {} # map mangled_name -> SASSTextSection
        self._imix_cache = {} # Cache for resolved IMIX to handle shared dependencies and speed up

        self._parse_file()

    def _parse_file(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Regex to match header line
        header_regex = re.compile(r"^//[-]+\s+\.text\.(\S+)\s+[-]+")
        
        current_mangled_name = None
        current_section_lines = []
        current_start_line = 0

        for i, line in enumerate(lines):
            match = header_regex.search(line)
            if match:
                # Found a new section start
                # If we were collecting a previous section, save it
                if current_mangled_name:
                    # Join lines to form raw text
                    raw_section = "".join(current_section_lines)
                    # Create Section Object
                    section = SASSTextSection(self.filename, current_start_line, current_mangled_name, raw_section)
                    self.text_sections[current_mangled_name] = section
                
                # Start new section
                current_mangled_name = match.group(1)
                current_start_line = i + 1  # 1-based line number points to the header line
                current_section_lines = [line]
                # We add the header line to the section content to keep it clean
            
            else:
                if current_mangled_name:
                    current_section_lines.append(line)
        
        # Save the last section
        if current_mangled_name:
            raw_section = "".join(current_section_lines)
            section = SASSTextSection(self.filename, current_start_line, current_mangled_name, raw_section)
            self.text_sections[current_mangled_name] = section

    def getAllTextSections(self):
        return list(self.text_sections.values())

    def getIMIXForKernel(self, kernel_name):
        """
        Returns combined IMIX for kernel and its dependencies.
        Checks for circular dependencies.
        """
        if kernel_name not in self.text_sections:
            return defaultdict(int), 0

        path = [] # For cycle detection
        circular_deps_count = 0

        def traverse(name):
            nonlocal circular_deps_count
            # Check cache first
            # We assume _imix_cache stores only "clean" (cycle-free subtree) results
            if name in self._imix_cache:
                return self._imix_cache[name].copy(), True # imix, is_clean

            if name in path:
                # Circular dependency found
                circular_deps_count += 1
                cycle = " -> ".join(path + [name])
                file_info = f"{self.filename}:{self.text_sections[name].line_number}"
                print(f"Warning: Circular dependency detected in {file_info}: {cycle} (Ignoring back-edge)")
                return defaultdict(int), False # Return empty, mark as dirty
            
            section = self.text_sections.get(name)
            if not section:
                return defaultdict(int), True # External symbol or unknown

            current_imix = section.imix.copy()
            is_clean_subtree = True
            
            # Recurse on references
            path.append(name)
            for ref in section.references:
                # Skip self-references to avoid infinite recursion
                if ref == name:
                    continue

                # Check if ref is a known text section (kernel)
                if ref in self.text_sections:
                    ref_imix, ref_clean = traverse(ref)
                    if not ref_clean:
                        is_clean_subtree = False
                    # Add ref_imix to current_imix
                    for k, v in ref_imix.items():
                        current_imix[k] += v
            path.pop()
            
            # Cache the result only if the subtree is clean (no cycles pruned)
            if is_clean_subtree:
                self._imix_cache[name] = current_imix
            
            return current_imix, is_clean_subtree

        imix, _ = traverse(kernel_name)
        return imix, circular_deps_count



    


