"""Retain the aircraft sync fields that FMSync decodes and then drops, without changing the decoding.

Per aircraft sync (AircraftSmall packet) the parser already reads: a weapon list (count + extra values per weapon),
the packed velocity, the sensor states (SensorsControlStates), the countermeasure states
(CounterMeasuresControlState), the target designations (TargetDesignationControlState), two optional u16 values and
the length of a skipped seeker block. This patch keeps them:

  getFmProbe()    -> [(time_ms, uid, eid, discrete, continuous)], one row when the discrete text changes or at most
                     once a second per aircraft; ``discrete`` is "" when unchanged since the previous row of that
                     aircraft. Both texts are "key=value;" lists (keys documented in inspect_match.py).
  getFmVelocity() -> [(time_ms, uid, eid, vx, vy, vz)] every sync: netutils::unpack_velocity(vals_4, vel, 500),
                     Dagor world axes, m/s.
Field meanings are the upstream reverse-engineered names, not confirmed semantics.
"""
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "replay/mpi/PositionSync.cpp"
source = path.read_text()
if "get_fm_probe" in source:
    print("FM sync probe already present")
    sys.exit(0)

preamble = '''#include <cstdio>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
using FmProbeRow = std::tuple<uint32_t, uint32_t, uint32_t, std::string, std::string>;
using FmVelRow = std::tuple<uint32_t, uint32_t, uint32_t, float, float, float>;
static bool fm_probe_enabled = false;
static std::vector<FmProbeRow> fm_probe_rows;
static std::vector<FmVelRow> fm_vel_rows;
static std::unordered_map<uint32_t, std::string> fm_last_disc;
static std::unordered_map<uint32_t, uint32_t> fm_last_time;
void enable_fm_probe() { fm_probe_rows.clear(); fm_vel_rows.clear(); fm_last_disc.clear(); fm_last_time.clear();
  fm_probe_enabled = true; }
std::vector<FmProbeRow> get_fm_probe() { return fm_probe_rows; }
std::vector<FmVelRow> get_fm_velocity() { return fm_vel_rows; }
static void fm_put(std::string &out, const std::string &key, double value) {
  char buf[96];
  std::snprintf(buf, sizeof buf, "%s=%.10g;", key.c_str(), value);
  out += buf;
}
'''

start = source.index("bool FMSync(")
end = source.index("struct TankRef", start)
fn = source[start:end]


def sub(old, new):
    global fn
    if fn.count(old) != 1:
        raise RuntimeError(f"FMSync anchor not unique or missing: {old!r}")
    fn = fn.replace(old, new, 1)


sub("""          bool v1 = bs.ReadBit();
""", """          bool v1 = bs.ReadBit();
          std::string fm_disc, fm_cont;
          auto D = [&](const std::string &k, double x) { if (fm_probe_enabled) fm_put(fm_disc, k, x); };
          auto C = [&](const std::string &k, double x) { if (fm_probe_enabled) fm_put(fm_cont, k, x); };
          D("f41", fVar41LessThan0p5); D("b7330", _0x7330_val); D("v1", v1);
""")
sub("""              temp_bits.push_back(bs.ReadBit());
            }
""", """              temp_bits.push_back(bs.ReadBit());
            }
            D("v1_v2", v2); D("v1_u1", unk1); D("v1_u2", unk2); D("v1_n", count);
            { std::string tb; for (bool b: temp_bits) tb += b ? '1' : '0'; if (fm_probe_enabled) fm_disc += "v1_bits=" + tb + ";"; }
""")
sub("""          vals.resize(zig_val);
""", """          vals.resize(zig_val);
          D("nw", zig_val);
""")
sub("""                bs.ReadZigZag(temp_zig);
""", """                bs.ReadZigZag(temp_zig);
                D("w" + std::to_string(i) + "x" + std::to_string(ii), temp_zig);
""")
sub("""            vals[i] = temp;
""", """            vals[i] = temp;
            D("w" + std::to_string(i), temp); D("w" + std::to_string(i) + "n", temp1);
""")
sub("""          netutils::unpack_velocity(vals_4, vel, 500);
""", """          netutils::unpack_velocity(vals_4, vel, 500);
          if (fm_probe_enabled)
            fm_vel_rows.emplace_back(state.curr_time_ms, uid, unit->curr_eid.get_handle(), vel.x, vel.y, vel.z);
""")
sub("""          for (auto &s: sensors) {
            RET_FAIL(s.deserialize(bs));
          }
""", """          for (auto &s: sensors) {
            RET_FAIL(s.deserialize(bs));
          }
          D("ns", sensorsCount);
          for (size_t k = 0; k < sensors.size(); ++k) {
            auto &s = sensors[k];
            auto p = "s" + std::to_string(k) + "_";
            D(p + "type", s.sensor_type_maybe); D(p + "on", s.first_bool); D(p + "a", s.field136_0x88);
            D(p + "b", s.field137_0x89); D(p + "c", s.field138_0x8a); D(p + "e", s.field149_0xa4);
            D(p + "f", s.field150_0xa8); D(p + "n", s.field132_0x84); D(p + "m", s.field133_0x85);
            D(p + "6a", (uint8_t) s.some_data_6[0]); D(p + "6b", (uint8_t) s.some_data_6[1]);
            for (size_t j = 0; j < s.field4_0x4.size(); ++j) D(p + "l" + std::to_string(j), s.field4_0x4[j]);
            C(p + "d1", s.some_data_1); C(p + "d2", s.some_data_2); C(p + "d3", s.some_data_3);
            C(p + "d4", s.some_data_4); C(p + "d5", s.some_data_5); C(p + "g", s.field147_0xa8);
          }
""")
sub("""          if (sensorsCount != 0) {
            bs.Read(v);
          }
""", """          if (sensorsCount != 0) {
            bs.Read(v);
            D("s_v", v);
          }
""")
sub("""          for (auto &c: counterMeasures) {
            RET_FAIL(c.deserialize(bs));
          }
""", """          for (auto &c: counterMeasures) {
            RET_FAIL(c.deserialize(bs));
          }
          D("ncm", counterMeasuresCount);
          for (size_t k = 0; k < counterMeasures.size(); ++k) {
            D("cm" + std::to_string(k) + "_1", counterMeasures[k].v1);
            D("cm" + std::to_string(k) + "_2", counterMeasures[k].v2);
          }
""")
sub("""          if (counterMeasuresCount > 0)
            bs.Read(counter_measures_v);
""", """          if (counterMeasuresCount > 0) {
            bs.Read(counter_measures_v);
            D("cm_v", counter_measures_v);
          }
""")
sub("""          for (auto &t: targets) {
            RET_FAIL(t.deserialize(bs));
          }
""", """          for (auto &t: targets) {
            RET_FAIL(t.deserialize(bs));
          }
          D("nt", targetsNum);
          for (size_t k = 0; k < targets.size(); ++k) {
            auto &t = targets[k];
            auto p = "t" + std::to_string(k) + "_";
            D(p + "1", t.v1); D(p + "2", t.v2); D(p + "3", t.v3); D(p + "c", t.write_compressed); D(p + "8", t.v8);
            D(p + "10", t.v10); D(p + "12", t.v12); D(p + "13", t.v13); D(p + "14", t.v14); D(p + "15", t.v15);
            C(p + "5", t.v5); C(p + "11", t.v11);
            if (t.v3) { C(p + "4x", t.v4.x); C(p + "4y", t.v4.y); C(p + "4z", t.v4.z); }
            if (t.write_compressed) { C(p + "6x", t.v6.x); C(p + "6y", t.v6.y); C(p + "6z", t.v6.z); }
            else { C(p + "7x", t.v7.x); C(p + "7y", t.v7.y); C(p + "7z", t.v7.z); }
            C(p + "9x", t.v9.x); C(p + "9y", t.v9.y); C(p + "9z", t.v9.z);
          }
""")
sub("""            bs.Read(vt_1);
            bs.Read(vt_2);
""", """            bs.Read(vt_1);
            bs.Read(vt_2);
            D("vt1", vt_1); D("vt2", vt_2);
""")
sub("""          if (bit_thing_2) {
            DeserializeSeekerData(bs);
          }
""", """          if (bit_thing_2) {
            auto sk_pos = bs.GetReadOffset();
            uint32_t sk_bits = 0;
            bs.Read(sk_bits);
            D("sk_bits", sk_bits);
            if (fm_probe_enabled && sk_bits > 0) {
              uint8_t sk_buf[64] = {};
              uint32_t take = sk_bits < 512 ? sk_bits : 512;
              bs.ReadBits(sk_buf, take);
              std::string hex;
              constexpr char digits[] = "0123456789abcdef";
              for (uint32_t q = 0; q < (take + 7) / 8; ++q) { hex += digits[sk_buf[q] >> 4]; hex += digits[sk_buf[q] & 15]; }
              fm_cont += "sk_hex=" + hex + ";";
            }
            bs.SetReadOffset(sk_pos);
            DeserializeSeekerData(bs);
          }
          if (fm_probe_enabled) {
            uint32_t key = uid;
            auto &last = fm_last_disc[key];
            auto found = fm_last_time.find(key);
            bool changed = last != fm_disc;
            if (changed || found == fm_last_time.end() || state.curr_time_ms >= found->second + 1000) {
              fm_probe_rows.emplace_back(state.curr_time_ms, uid, unit->curr_eid.get_handle(),
                                         changed ? fm_disc : std::string(), fm_cont);
              last = fm_disc;
              fm_last_time[key] = state.curr_time_ms;
            }
          }
""")
source = preamble + source[:start] + fn + source[end:]
path.write_text(source)

path = root / "python/state/State.cpp"
source = path.read_text()
if 'm.def("enableFmProbe"' not in source:
    source = '''#include <pybind11/stl.h>
#include <tuple>
void enable_fm_probe();
std::vector<std::tuple<uint32_t, uint32_t, uint32_t, std::string, std::string>> get_fm_probe();
std::vector<std::tuple<uint32_t, uint32_t, uint32_t, float, float, float>> get_fm_velocity();
''' + source
    marker = '  m.def("g_log_flush", []() {'
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected binding layout")
    source = source.replace(marker, '''  m.def("enableFmProbe", &enable_fm_probe);
  m.def("getFmProbe", &get_fm_probe);
  m.def("getFmVelocity", &get_fm_velocity);
''' + marker, 1)
    path.write_text(source)
print("FM sync probe added")
