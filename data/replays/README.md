# Replay digitizations (game truth)

Per-frame readouts transcribed from server-replay screenshots of the missile
info box. TSV, UTF-8, no comment lines (consumers use `csv.DictReader`);
missing values are `NA`.

Column notes (apply to every file here):

- `missile_flight_time_s` is the info-box "时长" field (missile flight time);
  `replay_time_s` is the replay timeline display. Frame comparisons key on
  `missile_flight_time_s`.
- `overload_g` is the displayed G: **trajectory-normal, thrust-inclusive**
  (UPDATE_V1.0.2.md §5 1c/1d), signed near zero (the model's
  `trajectory_lateral_load_g` is an unsigned magnitude).
- `displayed_distance_km` is the replay-camera-to-missile distance, **not**
  missile-to-target range (UPDATE_V1.0.2.md §5 1d note ③). Do not use it as a
  physical quantity; `displayed_flown_distance_km` does validate a speed
  integration.
- `angle_of_attack_deg` carries a known small-alpha display bias
  (alpha ≲ 1.5°, UPDATE_V1.0.2.md §5 item 5).

## Confirmed replay outcomes

On 2026-09-08, the user confirmed that **all missiles in these five screenshot
groups hit their targets** ("所有导弹都命中了"). This is the observed game outcome
for each recorded scenario, confirmed by the user independently of which
terminal readouts remain visible in the screenshots.

| screenshot folder | missile | screenshots | game outcome |
|---|---|---:|---|
| `30degpl12` | PL-12 | 30 | Hit — user confirmed |
| `30degr77` | R-77 | 41 | Hit — user confirmed |
| `30degpl12slow` | PL-12 | 50 | Hit — user confirmed |
| `pl12level` | PL-12 | 12 | Hit — user confirmed |
| `r77-1` | R-77-1 | 46 | Hit — user confirmed |

The [user-provided screenshot recovery report](/Users/skyking/.codex/attachments/91b06aea-11c9-43b4-af87-9e7401598e4f/pasted-text.txt)
contains the five groups' 179 frame records. The final `30degr77` image has no
missile readout box, so its missing telemetry remains `NA`; the scenario's
confirmed outcome is still a hit. The last recorded missile time is not an
identified impact time, and this confirmation does not supply a numerical
miss distance or a damage result.

Model comparisons for these recorded scenarios should include the confirmed
hit outcome alongside the speed, displayed overload and angle-of-attack curves.

## Files

| file | frames | scenario | notes |
|---|---|---|---|
| `pl12_30deg_20260823.tsv` | 30 | `30degpl12`: launch 1508 km/h @ 6783 m, target nominally 1549 km/h, 8000 m, azimuth 30° (A2) | First frame at 2.2 s. Actual target maneuvered; the historical straight-target simulation is an approximation. |
| `r77_30deg_20260823.tsv` | 41 | `30degr77`: launch 1500 km/h @ 6200 m, target nominally 1550 km/h, 8000 m, azimuth 30° (A4) | 40 telemetry rows; final image has no missile readout. The historical test's “slow” label does not mean a 1200 km/h launch. |
| `pl12_30deg_slow_20260823.tsv` | 50 | `30degpl12slow`: launch 1200 km/h @ 6200 m, target nominally 1550 km/h, 8000 m, azimuth 30° (A3) | Flight time 0.2–7.5 s; retains the displayed −0.1 G. |
| `r77_1_level_20260824.tsv` | 46 | `r77-1`: launch ~1200 km/h @ 6300 m, heading 0°; straight level target at 1200 km/h, 8000 m, azimuth 30°, absolute heading 180° (parallel opposite courses, user confirmed) | IOG+DL → 惯导 → TRK (TRK from t≈0.9 s). AoA shoulder 23–24°; a shoulder alone does not establish equilibrium. |
| `pl12_level_20260824.tsv` | 12 | `pl12level`: PL-12, historical A7 setup; target-course interpretation not independently confirmed | First frame at 2.4 s; launch transient is not captured. Source filenames now present. Replay time for `20260824115621_1.jpg` is 9.081 s. |

All five TSVs contain the user-provided recovery report's values, imported on
2026-09-08 with units stripped from numeric cells and `NA` preserved. They
contain 179 screenshot records, including 178 usable missile-telemetry rows.
Flight time, replay time and filenames remain separate; no interpolation or
time shift has been applied to these source data. R-77-1 uses the user-confirmed
parallel opposite courses above. Other scenario inputs retain the historical
A2/A3/A4/A7 definitions in `tests/test_replay_anchors.py`; the camera-distance
column is never used as missile-target range.

`scripts/compare_rebuild_replays.py` runs the unchanged rebuild on these five
scenarios and writes comparison curves and numerical tables to `outputs/`.
Its separate axial balance uses measured speed/AoA and nominal propulsion at
launch altitude; it is a conditional force check, not trajectory validation.
