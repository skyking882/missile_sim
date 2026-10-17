# Final delivery verification — 2026-10-17

Completed successful run39074725895, job117519405143. Delivery source db510a1bc8777a414586198622fa0988efe20717. Package contains REPORT.md, CROSS_MODE_RESULT.md, protocols, three frozen derived input files, all scored cases, independent reviews and runnable code. No production dev, raw branch or native decoder was changed in this turn. Computation used completed in-session GitHub Actions; no AI subagents or local reruns are claimed.

## Actual packaged rerun

The delivered entrypoint ran the five mathematical tests, the original quantization/phase/temporal tracks, the exploratory first-capture/common-lag follow-up and the cross-mode transfer. All executed successfully. Fourteen result sets were compared:

-14/14 passed numerical comparison.
-12/14 were exactly equal as parsed JSON.
-Bridge summary/rows differed at most9.992007221626409e-16 in floating values; no structure, integer, boolean or text differences.
-Three original input file hashes matched before execution.
-The comparison used the beforehand-declared1e-5absolute+1e-9relative floating tolerance. Actual deviations are reported above, not hidden by that tolerance. Gzip metadata timestamps were not compared.

This is mathematical and research-artifact reproducibility, not a complete production simulator regression or independent new replay validation.

Other executed checks:488 independently expressed six-variable equality LPs agreed with the original four-variable minimum-slack LP feasibility results;240 explicit integer phase-count checks agreed with the interval arithmetic. The raw bundle verifier was run before and after final assembly; original46WRPL/2814 verification items remain unchanged. All80 packaged files had size/SHA256 checks; rootMANIFEST.json is the81st file.

## Cross-mode result to read carefully

The ordinary 1-L/T family's leave-scene coefficients imply capture retention0.8003692175–0.8011120665 without fitting capture rates. On the79 R77 conditional reset cases, linear complete-rate RMS0.0284757443deg/s and exact spherical feedback RMS0.0284711492deg/s; linear rate-cell compatibility73/79. The current n is expressly a diagnostic reset/projection input, not an independent direction prediction. An exact feedback step moves that input ray by RMS0.0051493869deg, maximum0.0201613140deg; the direction/output-stage question remains open.

The globalphase1 rule supports430/430 observed later25→31 intervals, but only114/501 first captures. Do not equate this to a universal1Hz radar observation frequency. All counterexamples and phase-incompatible first captures are retained in the report.

## Artifact

Name: missile-sim-ins-trk-refinement-20261017
Artifact ID:12116871425
ZIP size:25544351bytes
GitHub-uploaded ZIP SHA256:d8b4e193bbce40c2068910ef50c740f85f3a031dc40de472200a278a3120996a

Run from the extracted directory:

```bash
python3 reproduce.py /absolute/path/to/new-output
```

This receipt is written after assembly. The package already includes VERIFICATION.md and verification/reproduction_verification.json containing the underlying numerical record.
