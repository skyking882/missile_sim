"""Retain otherwise discarded WeaponSync fields without changing decoding."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "replay/mpi/PositionSync.cpp"
source = path.read_text()
if "enable_weapon_field_probe" not in source:
    preamble = '''#include <map>
#include <vector>
#include <string>
using WeaponProbeRow = std::map<std::string, double>;
static bool weapon_field_probe_enabled = false;
static std::vector<WeaponProbeRow> weapon_field_probe_rows;
void enable_weapon_field_probe() { weapon_field_probe_rows.clear(); weapon_field_probe_enabled = true; }
std::vector<WeaponProbeRow> get_weapon_field_probe() { return weapon_field_probe_rows; }
'''
    start = source.index("bool ParseWeapon(")
    end = source.index("\nint count = 0;", start)
    fn = source[start:end]
    fn = fn.replace("  bool bool1, bool2;", '''  WeaponProbeRow probe;
  auto save = [&](const std::string &key, double value) {
    if (weapon_field_probe_enabled) probe[key] = value;
  };
  save("time_ms", state.curr_time_ms);
  bool bool1, bool2;''', 1)
    fn = fn.replace("  Rocket *entity = cb(state, eid);", '''  Rocket *entity = cb(state, eid);
  save("eid", eid.get_handle());
  save("bool1", bool1); save("bool2", bool2);''', 1)
    fn = fn.replace("  SpaceTimeEuler *back = nullptr;", '''  save("position_uncompressed", is_not_compressed);
  save("x", WeaponPos.x); save("y", WeaponPos.y); save("z", WeaponPos.z);
  SpaceTimeEuler *back = nullptr;''', 1)
    fn = fn.replace("    bs.ReadZigZag(b5);", "    bs.ReadZigZag(b5);\n    save(\"b5\", b5);", 1)
    fn = fn.replace("      RET_FAIL(bs.Read(back->euler));", '''      RET_FAIL(bs.Read(back->euler));
      save("euler_x", back->euler.x); save("euler_y", back->euler.y); save("euler_z", back->euler.z);''', 1)
    fn = fn.replace("      RET_FAIL(bs.Read(b6));", '''      RET_FAIL(bs.Read(b6));
      save("euler_x", b6.x); save("euler_y", b6.y); save("euler_z", b6.z);''', 1)
    for field in ("b7", "b8"):
        limit = "4000.0f" if field == "b7" else "30.0f"
        for axis in "xyz":
            text = f"    {field}.{axis} = netutils::UNPACK(temp_t, {limit});"
            fn = fn.replace(text, text + f'\n    save("{field}_{axis}_raw_u16", temp_t);\n    save("{field}_{axis}_upstream", {field}.{axis});', 1)
    fn = fn.replace("    if (!bool1) {", '''    save("sf_flags", flags);
    save("sf1", sf1); save("sf2", sf2); save("sf3", sf3); save("sf4", sf4);
    if (!bool1) {''', 1)
    fn = fn.replace("      bs.IgnoreBits(sz);", '''      save("vv", vv); save("vv2", vv2); save("skipped_bits", sz);
      bs.IgnoreBits(sz);''', 1)
    fn = fn.replace("  return true;\n}", "  if (weapon_field_probe_enabled) weapon_field_probe_rows.push_back(std::move(probe));\n  return true;\n}", 1)
    source = preamble + source[:start] + fn + source[end:]
    path.write_text(source)

path = root / "python/state/State.cpp"
source = path.read_text()
if 'm.def("enableWeaponFieldProbe"' not in source:
    source = '''#include <pybind11/stl.h>
#include <map>
#include <vector>
#include <string>
void enable_weapon_field_probe();
std::vector<std::map<std::string, double>> get_weapon_field_probe();
''' + source
    marker = '  m.def("g_log_flush", []() {'
    source = source.replace(marker, '''  m.def("enableWeaponFieldProbe", &enable_weapon_field_probe);
  m.def("getWeaponFieldProbe", &get_weapon_field_probe);
''' + marker, 1)
    path.write_text(source)
print("Weapon field probe enabled in source")
