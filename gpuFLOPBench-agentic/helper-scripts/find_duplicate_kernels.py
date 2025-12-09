#!/usr/bin/env python3
from pathlib import Path
import collections
import json
import re
import sys

ALLOWED_EXT = {'.cu', '.cuh', '.cpp', '.cc', '.c', '.h', '.hpp', '.hh', '.cxx', '.ipp', '.inl', '.cu.h', '.cuhh'}
NORMALIZE_RE = re.compile(r'\s+')

root = Path('gpuFLOPBench/src')
if not root.exists():
    print('root not found', file=sys.stderr)
    sys.exit(1)


def normalize(text: str) -> str:
    return NORMALIZE_RE.sub(' ', text).strip()


def strip_commented_lines(text: str) -> str:
    lines = text.splitlines()
    kept = []
    in_block = False
    for line in lines:
        current = line
        while True:
            if in_block:
                end = current.find('*/')
                if end == -1:
                    current = ''
                    break
                current = current[end + 2:]
                in_block = False
                continue
            stripped = current.lstrip()
            if stripped.startswith('//'):
                current = ''
                break
            if stripped.startswith('/*'):
                end = stripped.find('*/')
                if end == -1:
                    current = ''
                    in_block = True
                    break
                prefix_len = len(current) - len(stripped)
                current = ' ' * prefix_len + stripped[end + 2:]
                continue
            break
        if in_block:
            continue
        if stripped.startswith('//'):
            continue
        kept.append(current)
    return '\n'.join(kept)


def skip_block(text: str, idx: int, open_char: str, close_char: str) -> int:
    depth = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return idx + 1
        elif ch in ('"', "'"):
            quote = ch
            idx += 1
            while idx < length:
                if text[idx] == '\\':
                    idx += 2
                    continue
                if text[idx] == quote:
                    idx += 1
                    break
                idx += 1
            continue
        idx += 1
    return idx


def skip_string(text: str, idx: int, quote: str) -> int:
    i = idx + 1
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == '\\':
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    return length


def find_matching_brace(text: str, start_idx: int) -> int | None:
    depth = 1
    i = start_idx + 1
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        elif ch in ('"', "'"):
            i = skip_string(text, i, ch)
            continue
        elif ch == '/':
            if i + 1 < length and text[i + 1] == '/':
                newline = text.find('\n', i + 2)
                i = newline if newline != -1 else length
                continue
            if i + 1 < length and text[i + 1] == '*':
                close = text.find('*/', i + 2)
                if close == -1:
                    return length
                i = close + 2
                continue
        i += 1
    return None


def find_body_range(text: str, idx: int) -> tuple[int | None, int | None]:
    i = idx
    length = len(text)
    while i < length:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith('__attribute__', i):
            i = skip_block(text, i, '(', ')')
            continue
        if ch == '{':
            end = find_matching_brace(text, i)
            if end:
                return i, end
            return None, None
        if text.startswith('//', i):
            newline = text.find('\n', i + 2)
            if newline == -1:
                return None, None
            i = newline
            continue
        if text.startswith('/*', i):
            close = text.find('*/', i + 2)
            if close == -1:
                return None, None
            i = close + 2
            continue
        i += 1
    return None, None


def parse_kernel(text: str, global_idx: int) -> tuple[str | None, str | None]:
    length = len(text)
    i = global_idx + len('__global__')
    while i < length:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if text.startswith('__launch_bounds__', i):
            i = skip_block(text, i + len('__launch_bounds__'), '(', ')')
            continue
        if c.isalpha() or c == '_':
            j = i + 1
            while j < length and (text[j].isalnum() or text[j] in ('_', ':')):
                j += 1
            ident = text[i:j]
            k = j
            while k < length and text[k].isspace():
                k += 1
            if k < length and text[k] == '<':
                k = skip_block(text, k, '<', '>')
                while k < length and text[k].isspace():
                    k += 1
            if k < length and text[k] == '(':
                if ident == '__launch_bounds__':
                    k = skip_block(text, k, '(', ')')
                    i = k
                    continue
                params_end = skip_block(text, k, '(', ')')
                body_range = find_body_range(text, params_end)
                if body_range[0] is None:
                    return None, None
                snippet = text[global_idx:body_range[1]]
                return ident, snippet
            i = j
            continue
        i += 1
    return None, None

results = {}
for cuda_dir in sorted(d for d in root.iterdir() if d.is_dir() and d.name.endswith('-cuda')):
    counts = collections.defaultdict(lambda: collections.defaultdict(set))
    for path in cuda_dir.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXT:
            continue
        try:
            raw_text = path.read_text()
        except UnicodeDecodeError:
            raw_text = path.read_text(errors='ignore')
        text = strip_commented_lines(raw_text)
        pos = 0
        while True:
            idx = text.find('__global__', pos)
            if idx == -1:
                break
            name, snippet = parse_kernel(text, idx)
            if name and snippet:
                normalized = normalize(snippet)
                rel_path = str(path.relative_to(root))
                counts[name][normalized].add(rel_path)
            pos = idx + len('__global__')
    dupes = {}
    for name, body_map in counts.items():
        if len(body_map) <= 1:
            continue
        dupes[name] = len(body_map)
    if dupes:
        results[cuda_dir.name] = dupes
print(json.dumps(results, indent=2))
