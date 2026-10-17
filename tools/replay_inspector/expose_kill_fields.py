"""Expose the KillMessage fields the parser decodes but does not bind: death type (0 normal, -1 crash, -2 left),
weapon type (1 bullet, 2 bomb, 3 rocket/missile, 4 torpedo; upstream enum, 'IsBullet' presumed), weapon flags and
the unnamed extra fields."""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "python/mpi/battle_messages.cpp"
source = path.read_text()
if '"death_type"' not in source:
    old = '      .def_readonly("is_burav_kill", &mpi::KillMessage::maybe_is_burav_kill)\n'
    if source.count(old) != 1:
        raise RuntimeError("Unexpected KillMessage binding layout")
    source = source.replace(old, old + '''      .def_property_readonly("death_type", [](const mpi::KillMessage &k) { return (int) k.DeathType; })
      .def_property_readonly("weapon_type", [](const mpi::KillMessage &k) { return (int) k.some_enum; })
      .def_property_readonly("weapon_flags", [](const mpi::KillMessage &k) { return (int) k.some_weap_flags; })
      .def_readonly("extra_str_1", &mpi::KillMessage::weird_str_1)
      .def_readonly("extra_str_2", &mpi::KillMessage::weird_str_2)
      .def_readonly("extra_val_3", &mpi::KillMessage::weird_val_3)
      .def_readonly("extra_str_4", &mpi::KillMessage::weird_str_4)
''', 1)
    path.write_text(source)
print("Kill message fields exposed")
