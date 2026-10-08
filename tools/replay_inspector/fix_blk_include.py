"""Fix std::vector byte allocation when expanding text BLK includes.

The upstream insert_items helper calls insert(pos, n). For std::vector<char>
this inserts one char, not n chars, then blk_parser writes n bytes over the heap.
Use the standard count-and-value overload at the two include-buffer sites.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "3rdParty/Dagor_Engine/ioSys/datablock/blk_parser.cpp"
source = path.read_text()
old = "insert_items(buffer, pos, len + 2);"
new = "buffer.insert(buffer.begin() + pos, len + 2, '\\0');"
if source.count(old) == 2:
    path.write_text(source.replace(old, new))
    print("Fixed byte allocation for BLK include expansion")
elif source.count(new) != 2:
    raise RuntimeError("Unexpected include-buffer code; patch was not applied")
