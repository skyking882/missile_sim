"""Infer slotted loadouts from the loaded BLK if custom units lack shop metadata.

Keep explicit hasWeaponSlots values unchanged. Custom CDK units are absent from
the stock wpcost index, but their actual merged BLK defines WeaponSlots.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "replay/ecs/Unit.cpp"
source = path.read_text()
old = 'this->unit_wpcost->getBool("hasWeaponSlots", false)'
new = 'this->unit_wpcost->getBool("hasWeaponSlots", blk.getBlockByName("WeaponSlots") != nullptr)'
if source.count(old) == 1:
    path.write_text(source.replace(old, new))
    print("Enabled slot metadata fallback for custom units")
elif source.count(new) != 1:
    raise RuntimeError("Unexpected slot-selection code; patch was not applied")
