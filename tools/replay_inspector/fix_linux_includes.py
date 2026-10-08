"""Normalize upstream Windows-only include spelling; no parser logic changes."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
spellings = {}
for candidate in root.rglob("*"):
    if candidate.suffix not in {".h", ".hpp", ".inl"}:
        continue
    parts = candidate.relative_to(root).parts
    for start in range(len(parts)):
        spelling = "/".join(parts[start:])
        spellings.setdefault(spelling.lower(), set()).add(spelling)

def normalize(match):
    prefix, include_path = match.groups()
    name = include_path.decode("utf-8")
    if "/" not in name:
        return match.group(0)
    candidates = spellings.get(name.lower(), set())
    if len(candidates) == 1:
        return prefix + next(iter(candidates)).encode("utf-8")
    return match.group(0)

for subtree in ("src", "replay", "python", "3rdParty/Dagor_Engine"):
    for path in (root / subtree).rglob("*"):
        if path.suffix not in {".cpp", ".h", ".hpp", ".inl"}:
            continue
        original = path.read_bytes()
        updated = re.sub(rb'(#\s*include\s*[<"])([^>"\r\n]+)', normalize, original)
        if original != updated:
            path.write_bytes(updated)
            print(path.relative_to(root))
