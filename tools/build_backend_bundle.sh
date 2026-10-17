#!/usr/bin/env bash
set -euo pipefail
# Build only the bundled patched source. It does not clone or change replay/model files.
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
build=${1:-"$root/build/backend"}
mkdir -p "$build"
build=$(cd -- "$build" && pwd)
py=${PYTHON3:-python3}
extra=()
if [ "$(uname -s)" = Darwin ]; then
  py=${PYTHON3:-/Applications/Xcode.app/Contents/Developer/usr/bin/python3}
  extra=(-DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_SYSTEM_PROCESSOR=x86_64
    -DCMAKE_C_FLAGS=-msse4.2\ -mpopcnt
    -DCMAKE_CXX_FLAGS=-msse4.2\ -mpopcnt\ -include\ smmintrin.h)
fi
mkdir -p "$build/vendor"
for pair in 'pybind11 pybind11-d03662f' 'tracy tracy-05cceee' 'libdwarf libdwarf-5dfb2cd' 'zstd zstd-f8745da'; do
  read -r name archive <<< "$pair"
  mkdir -p "$build/vendor/$name"
  tar -xzf "$root/tools/dependencies/$archive.tar.gz" -C "$build/vendor/$name" --strip-components=1
done
(cd "$build/vendor/libdwarf" && cmake -P "$root/tools/WrplReplayParser/cmake/patches/patch_libdwarf.cmake")
version=$($py -c 'import sys;print("%s.%s"%sys.version_info[:2])')
cmake -S "$root/tools/WrplReplayParser" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DENABLE_LTO=OFF -DPARSER_ENABLE_PYTHON_BINDINGS=ON -DPARSER_ENABLE_INSPECTOR=OFF \
  -DMI_BUILD_TESTS=OFF -DMI_BUILD_SHARED=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$build/vendor/pybind11" \
  -DFETCHCONTENT_SOURCE_DIR_TRACY="$build/vendor/tracy" \
  -DFETCHCONTENT_SOURCE_DIR_LIBDWARF="$build/vendor/libdwarf" \
  -DFETCHCONTENT_SOURCE_DIR_ZSTD="$build/vendor/zstd" \
  -DPYTHON_VERSION_TO_USE="$version" -DPython_EXECUTABLE="$py" "${extra[@]}"
cmake --build "$build" --target PyReplayParser -j4
printf 'Module directory: %s/python/main\n' "$build"
