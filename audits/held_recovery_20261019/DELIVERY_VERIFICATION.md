# Completed held-recovery verification

2026-10-19. Main diagnosis run39183651779/job117841471519 and delivery run39184380375/job117843600851 both completed success. Delivered source commit cde1af663e3e841cc1d751db30385717fe40ae47. This receipt was committed after packaging; package VERIFICATION.md and receipts/verification.json already contain the underlying results.

The delivered diagnose.py was executed again against the same immutable prior. All3 result files (summary.json, all_gates_118.json.gz, all_gates_148.json.gz) were identical after JSON decoding, numeric difference0. They are not3 independent datasets.

Independent arithmetic, without using the SciPy propagation result as its own reference:23 direction comparisons using scalar Rodrigues rotation;23 pointing comparisons with scalar matrix multiplication and angular clipping;644 beam margins evaluated in logarithmic rather than distance-root form. Maximum differences were5.9202046476941605e-15deg for direction,3.584806063106072e-15deg for pointing,7.105427357601002e-15dB for signal. These tiny numbers validate arithmetic reproducibility, not accuracy against unknown game internals.

The7 existing Q-observer tests and14 existing retry-clock tests passed. src, tests and data/profile_sets are unchanged from7a1c0c6e9ebc0a9e730ffaa5c2e79a40c2ae44f5. No production fix, full trajectory regression, fresh WRPL decode or independent AI subagent review is claimed.

250 files verified by exact size and SHA256, plus root MANIFEST.json =251 packaged files. Content15,377,011bytes excluding root manifest. Original prior is included and unchanged. The blocked followup.py does not exist; its later proposed experiments were not executed. Reproduction only covers the successful initial diagnosis and its arithmetic.

Artifact: missile-sim-held-recovery-reviewed-20261019
Artifact ID:12157008048
ZIP bytes:11,714,312
Uploaded ZIP SHA256:1becd0605a26c67323fe3b16dc844c2a6b3b242dbfa9fe799d96ffa5b996eaa7

```bash
python3 reproduce.py /absolute/path/to/new-output
```

Primary evidence remains23 held rows in one #118 episode. #148 has no held episode in the scored first6seconds and is a fresh/detector control, not a second held validation. The old acceptance trace is unchanged by logging. Correcting angular pointing does not stop the existing detector's2.4375s recovery; the negative result is retained.
