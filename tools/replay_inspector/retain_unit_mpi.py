"""Inventory per-unit MPI messages that the parser does not handle (it skips them).

Unit-bound object types (FMW, GM, INF, FM_FX, UFX, FM_DVM, GM_DVM, CUD, WEAP; see ObjectDispatcher.cpp ExtMpiTypes)
carry messages whose IDs the upstream parser does not know, e.g. damage-model (DVM) and effect (FX) updates. This
patch records each unhandled one without changing what is parsed:

  getUnitMpiProbe() -> [(time_ms, obj_type, obj_index, ext_uid, message_id, payload_bytes, payload_hex_prefix)]

ext_uid is the unit's entity handle (ObjectDispatcher uses ecs::EntityId(extUid)). Only the first 24 payload bytes
are kept, and at most 3,000,000 rows. The payloads are not decoded.
"""
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "replay/mpi/mpi.cpp"
source = path.read_text()
if "get_unit_mpi_probe" not in source:
    old = """      IObject *o = dispatch_object(oid, extUid, state);
      if (o) {
        Message *m = (mid != INVALID_MESSAGE_ID) ? o->dispatchMpiMessage(mid) : nullptr;
"""
    new = """      IObject *o = dispatch_object(oid, extUid, state);
      uint8_t probe_type = (uint8_t) (oid >> 0xb);
      bool probe_unit = probe_type == 0 || probe_type == 1 || probe_type == 2 || probe_type == 12 ||
                        probe_type == 13 || probe_type == 16 || probe_type == 17 || probe_type == 18 || probe_type == 25;
      Message *probe_m = nullptr;
      if (o) {
        Message *m = (mid != INVALID_MESSAGE_ID) ? o->dispatchMpiMessage(mid) : nullptr;
        probe_m = m;
"""
    if source.count(old) != 1:
        raise RuntimeError("Unexpected dispatch layout")
    source = source.replace(old, new, 1)
    old = """      } else {
        // LOG("object doesen't exist");
      }
    }
    return nullptr;
  }
"""
    new = """      } else {
        // LOG("object doesen't exist");
      }
      if (unit_mpi_probe_enabled && probe_unit && !probe_m && unit_mpi_probe_rows.size() < 3000000) {
        uint32_t at = BITS_TO_BYTES(bs.GetReadOffset());
        uint32_t size = bs.GetNumberOfBytesUsed() > at ? bs.GetNumberOfBytesUsed() - at : 0;
        std::string hex;
        constexpr char digits[] = "0123456789abcdef";
        for (uint32_t q = 0; q < size && q < 24; ++q) {
          uint8_t byte = bs.GetData()[at + q];
          hex += digits[byte >> 4];
          hex += digits[byte & 15];
        }
        unit_mpi_probe_rows.emplace_back(state->curr_time_ms, probe_type, (uint16_t) (oid & 0x7ff), extUid, mid, size,
                                         hex);
      }
    }
    return nullptr;
  }
"""
    if source.count(old) != 1:
        raise RuntimeError("Unexpected dispatch tail layout")
    source = source.replace(old, new, 1)
    marker = "namespace mpi {\n#define MAX_CALL_DEPTH (16)"
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected mpi.cpp layout")
    source = source.replace(marker, """#include <string>
#include <tuple>
#include <vector>
using UnitMpiRow = std::tuple<uint32_t, uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, std::string>;
static bool unit_mpi_probe_enabled = false;
static std::vector<UnitMpiRow> unit_mpi_probe_rows;
void enable_unit_mpi_probe() { unit_mpi_probe_rows.clear(); unit_mpi_probe_enabled = true; }
std::vector<UnitMpiRow> get_unit_mpi_probe() { return unit_mpi_probe_rows; }
""" + marker, 1)
    path.write_text(source)

path = root / "python/state/State.cpp"
source = path.read_text()
if 'm.def("enableUnitMpiProbe"' not in source:
    source = '''#include <pybind11/stl.h>
#include <tuple>
void enable_unit_mpi_probe();
std::vector<std::tuple<uint32_t, uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, std::string>> get_unit_mpi_probe();
''' + source
    marker = '  m.def("g_log_flush", []() {'
    source = source.replace(marker, '''  m.def("enableUnitMpiProbe", &enable_unit_mpi_probe);
  m.def("getUnitMpiProbe", &get_unit_mpi_probe);
''' + marker, 1)
    path.write_text(source)
print("Unit MPI inventory added")
