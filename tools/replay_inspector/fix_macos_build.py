"""macOS (x86_64 under Rosetta) build fixes; no parser logic changes.

- enet: compile unix.c on Apple too (only Linux was listed, leaving enet_host_random_seed undefined);
- PyReplayParser: drop -static-libgcc / -static-libstdc++, which Apple clang rejects.
Use with build_backend_macos.sh (Apple Silicon: the Dagor vector math needs x86 SSE4, so build for x86_64 and load it
from a universal Python, e.g. Xcode's /usr/bin/python3, under `arch -x86_64`).
"""
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "3rdParty/enet/CMakeLists.txt"
source = path.read_text()
old = 'elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")'
new = 'elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)'
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError("Unexpected enet CMake layout")
    path.write_text(source.replace(old, new))
path = root / "python/main/CMakeLists.txt"
source = path.read_text()
if "-static-libgcc" in source:
    source = source.replace("            -static-libgcc\n", "").replace("            -static-libstdc++\n", "")
    path.write_text(source)
print("macOS build fixes applied")
