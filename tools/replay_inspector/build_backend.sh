#!/usr/bin/env bash
set -euo pipefail
# Run inside Ubuntu WSL. Requires git, cmake, g++, python3-dev,
# libunwind-dev and liblzma-dev. No game process is used.
base=${1:-$HOME/.cache/wrpl-replay-parser-221be5c}
revision=221be5cac8e772ab41eefb0e7dac35ce6fb5c39b
mkdir -p "$base"
if [ ! -d "$base/WrplReplayParser/.git" ]; then
  git clone --branch develop https://github.com/LivingTheDagor/WrplReplayParser.git "$base/WrplReplayParser"
  git -C "$base/WrplReplayParser" checkout --detach "$revision"
fi
actual=$(git -C "$base/WrplReplayParser" rev-parse HEAD)
if [ "$actual" != "$revision" ]; then
  printf 'Unexpected source revision: %s\n' "$actual" >&2
  exit 1
fi
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
python3 "$script_dir/fix_linux_includes.py" "$base/WrplReplayParser"
python3 "$script_dir/fix_callback_shutdown.py" "$base/WrplReplayParser"
python3 "$script_dir/fix_local_package.py" "$base/WrplReplayParser"
python3 "$script_dir/fix_blk_include.py" "$base/WrplReplayParser"
python3 "$script_dir/fix_custom_weapon_slots.py" "$base/WrplReplayParser"
python3 "$script_dir/enable_weapon_probe.py" "$base/WrplReplayParser"
python3 "$script_dir/preserve_weapon_raw.py" "$base/WrplReplayParser"
python3 "$script_dir/fix_ecs_scalar_bindings.py" "$base/WrplReplayParser"
cmake -S "$base/WrplReplayParser" -B "$base/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DENABLE_LTO=OFF -DPARSER_ENABLE_PYTHON_BINDINGS=ON \
  -DPARSER_ENABLE_INSPECTOR=OFF -DMI_BUILD_TESTS=OFF -DMI_BUILD_SHARED=OFF
cmake --build "$base/build" --target PyReplayParser -j4
