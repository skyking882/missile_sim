"""Preserve creation fields and complete successful WeaponSync bit fragments."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "replay/mpi/PositionSync.cpp"
source = path.read_text()
if "get_weapon_field_probe_hex" not in source:
    source = source.replace("static std::vector<WeaponProbeRow> weapon_field_probe_rows;", '''static std::vector<WeaponProbeRow> weapon_field_probe_rows;
static std::vector<std::string> weapon_field_probe_hex;
std::vector<std::string> get_weapon_field_probe_hex() { return weapon_field_probe_hex; }''')
    source = source.replace("weapon_field_probe_rows.clear();", "weapon_field_probe_rows.clear(); weapon_field_probe_hex.clear();")
    source = source.replace("  WeaponProbeRow probe;", "  const auto probe_start_bit = bs.GetReadOffset();\n  WeaponProbeRow probe;", 1)
    source = source.replace("      save(\"vv\", vv);", '      save("seeker_start_bit", bs.GetReadOffset());\n      save("vv", vv);', 1)
    source = source.replace("  if (weapon_field_probe_enabled) weapon_field_probe_rows.push_back(std::move(probe));", '''  if (weapon_field_probe_enabled) {
    const auto end = bs.GetReadOffset();
    save("start_bit", probe_start_bit); save("end_bit", end);
    std::string raw;
    constexpr char digits[] = "0123456789abcdef";
    for (auto i = probe_start_bit / 8; i < (end + 7) / 8; ++i) {
      auto byte = bs.GetData()[i];
      raw += digits[byte >> 4]; raw += digits[byte & 15];
    }
    weapon_field_probe_rows.push_back(std::move(probe));
    weapon_field_probe_hex.push_back(std::move(raw));
  }''', 1)
    path.write_text(source)

path = root / "python/state/State.cpp"
source = path.read_text()
if 'm.def("getWeaponFieldProbeHex"' not in source:
    source = source.replace("void enable_weapon_field_probe();", "void enable_weapon_field_probe();\nstd::vector<std::string> get_weapon_field_probe_hex();")
    source = source.replace('  m.def("getWeaponFieldProbe", &get_weapon_field_probe);', '  m.def("getWeaponFieldProbe", &get_weapon_field_probe);\n  m.def("getWeaponFieldProbeHex", &get_weapon_field_probe_hex);')
    path.write_text(source)

path = root / "python/mpi/unit.cpp"
source = path.read_text()
if '.def_property_readonly("creation_raw_fields"' not in source:
    scalar = "uleb_1 u1_1 u4_1 weapon_ref u1_2 shell_type creation_time u4_4 u1_4 u4_5 u1_5 u4_6 u1_6".split()
    point3 = "starting_pos u12_2 u12_3 u12_4 u12_5".split()
    binding = '''    .def_property_readonly("creation_raw_fields", [](const Rocket &r) {
      py::dict d;
'''
    for field in scalar:
        binding += f'      d["{field}"] = r.{field};\n'
    for field in point3:
        binding += f'      d["{field}"] = py::make_tuple(r.{field}.x, r.{field}.y, r.{field}.z);\n'
    binding += '''      d["u16_1"] = py::make_tuple(r.u16_1.x, r.u16_1.y, r.u16_1.z, r.u16_1.w);
      d["u8_1"] = py::make_tuple(r.u8_1.x, r.u8_1.y);
      auto raw = [](const BitStream &bs) {
        py::dict value;
        value["bit_count"] = bs.GetWriteOffset();
        std::string hex;
        constexpr char digits[] = "0123456789abcdef";
        for (uint32_t i = 0; i < (bs.GetWriteOffset() + 7) / 8; ++i) {
          auto byte = bs.GetData()[i]; hex += digits[byte >> 4]; hex += digits[byte & 15];
        }
        value["hex"] = hex;
        return value;
      };
      d["some_weap_type_info"] = raw(r.some_weap_type_info);
      d["bomb_info"] = raw(r.bomb_info);
      d["maybe_sensor_info"] = raw(r.maybe_sensor_info);
      return d;
    })
'''
    marker = '  py::class_<Rocket, std::unique_ptr<Rocket, py::nodelete>>(unit, "Rocket")\n'
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected Rocket binding layout")
    source = source.replace(marker, marker + binding)
    path.write_text(source)
print("Added raw packet and creation-field preservation")
