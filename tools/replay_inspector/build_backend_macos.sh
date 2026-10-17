#!/usr/bin/env bash
set -euo pipefail
# macOS build of the backend as an x86_64 module (runs under Rosetta on Apple Silicon; the Dagor vector math uses
# x86 SSE4 intrinsics). Needs Xcode (clang, universal python3 3.9 with headers) and Homebrew cmake.
# Load it with:  PYTHONPATH=<base>/build/python/main arch -x86_64 /usr/bin/python3 inspect_match.py ...
base=${1:-$HOME/.cache/wrpl-replay-parser-221be5c-macos}
revision=221be5cac8e772ab41eefb0e7dac35ce6fb5c39b
py=${PYTHON3:-/Applications/Xcode.app/Contents/Developer/usr/bin/python3}
mkdir -p "$base"
if [ ! -d "$base/WrplReplayParser/.git" ]; then
  git clone --branch develop https://github.com/LivingTheDagor/WrplReplayParser.git "$base/WrplReplayParser"
  git -C "$base/WrplReplayParser" checkout --detach "$revision"
fi
[ "$(git -C "$base/WrplReplayParser" rev-parse HEAD)" = "$revision" ] || { echo "Unexpected source revision" >&2; exit 1; }
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
for patch in fix_linux_includes fix_callback_shutdown fix_local_package fix_blk_include fix_custom_weapon_slots \
             enable_weapon_probe preserve_weapon_raw fix_ecs_scalar_bindings retain_fm_sync retain_unit_mpi \
             expose_kill_fields fix_macos_build; do
  "$py" "$script_dir/$patch.py" "$base/WrplReplayParser"
done
cmake -S "$base/WrplReplayParser" -B "$base/build" -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DENABLE_LTO=OFF \
  -DPARSER_ENABLE_PYTHON_BINDINGS=ON -DPARSER_ENABLE_INSPECTOR=OFF -DMI_BUILD_TESTS=OFF -DMI_BUILD_SHARED=OFF \
  -DPYTHON_VERSION_TO_USE=3.9 -DPython_EXECUTABLE="$py" \
  -DCMAKE_C_FLAGS="-msse4.2 -mpopcnt" -DCMAKE_CXX_FLAGS="-msse4.2 -mpopcnt -include smmintrin.h"
cmake --build "$base/build" --target PyReplayParser signal_tracer -j8
cp "$base/build/3rdParty/signal_tracer/signal_tracer" "$base/build/python/main/"
