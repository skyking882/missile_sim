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

## Files

| file | frames | scenario | notes |
|---|---|---|---|
| `r77_1_level_20260824.tsv` | 46 | R-77-1 level shot: launch ~1200 km/h @ 6300 m, straight level target, 8000 m, azimuth 30° (the A7 geometry) | Seeker column resolves IOG+DL → 惯导 → TRK (TRK from t≈0.9 s). Alpha plateau 23.7° holds t≈1.5–2.1 s while G rises — the launch-capture release discriminant dataset. Source screenshots per row. Burnout A/B judge frames S041/S042. |
| `pl12_level_20260824.tsv` | 12 | PL-12 level shot, same A7 geometry (the 2026-08-24 'pl12level' control experiment) | First frame at flight time 2.4 s — the launch transient/release is **not** captured; anchors the de-load arc only (A7 in `tests/test_replay_anchors.py`) and is lift-law dataset #5 (k_s = 0.576±0.036). Source screenshots not archived per-row (`NA`). |

Provenance: digitized 2026-08-24 by the project owner from War Thunder server
replays; transcription pass 2026-08-26. `tests/test_replay_anchors.py` anchor
scalars (A2/A3/A4/A7) were drawn from these and two earlier captures (PL-12
30° fast/slow, R-77 30° slow — 121 frames total) whose full per-frame tables
were not archived in-repo; only the two files above survive as raw frames.
