# Final verification · 2026-10-17

Main report: REPORT.md at22088e87b46e941b0c6f61123a5a53797327dd53. Final successful run39069993344, job117505434465, sourcef06d982165a29a64fcee9005aab6ef483edab00c. All reported work completed in-session; no subagents or deferred background work were claimed.

## Actual source and independent checks

- Native exported records rescanned:1633750. Original cohort501 series;35564 rows retained including the observed duplicate-state tick;35563 unique reference states checked. Maximum differences Q=2.220446049250313e-16, w=1.3877787807814457e-17rad/s, normalized n=0.
- Seven algebra/interface tests passed, including guarded mappings that raise if current Ps/Vs/n/w are accessed by past-only predictors.
-488 independently matrix-coded reset evaluations matched stored scalar-projection results, maximum rate-error-score difference1.887379141862766e-15deg/s.
-37673 ordered row memberships matched exactly between ordinary and implicit comparisons. This includes overlapping32189-all-family and5484-R77 groups, not37673 independent observations.
-2928 current-label poisoning checks passed for the reset candidates using past-PV or pointing directions. Current-n variants are explicitly diagnostic and do use the endpoint n.

## Conditional versus past-only reset scores

Same79 R77 phase-zero fresh events, Qnew and kappa0.8 throughout:

| Projection direction | Complete rate RMS deg/s | Direction RMS deg |
|---|---:|---:|
| Current recorded n, diagnostic |0.0283891404|0 by construction, not a prediction|
| Current pointing ray, conditional |0.0379745439|4.144934399|
| Past-PV predicted direction |0.0511182392|9.345799473|

The low rate error does not certify a correct direction. In the predefined63 unsaturated members, pointing-conditioned reconstruction gives rate RMS0.0294770968deg/s and direction RMS0.0059571848deg; past-PV gives0.0305245033deg/s and3.656174917deg. This supports a rate handover relation but does not identify the source of the new direction. Pointing is still a serialized field, not proven independent target-angle measurement.

## Strict reproducibility and optimizer-refitting sensitivity

Fifteen numerical output sets were rerun. Ten were exactly equal as parsed JSON. Thirteen passed the initially declared1e-7absolute+1e-10relative tolerance. Two causality re-fitting files failed that strict threshold: maximum position-score difference3.155979359181771e-6m, with largest fitted-coefficient difference2.3775362345546114e-8. Structural checks, integers and string values matched exactly.

These failures remain in verification/first-strict-reproduction.json and are not relabelled strict successes. The reviewed wrapper permits only these two refit-result files a separate1e-5absolute comparison, requires other outputs to meet their original tolerances, and additionally rescores all models on the same samples at the ORIGINAL frozen coefficients. Frozen-coefficient position-score maximum difference was2.4832358391790876e-12m, passing1e-9absolute+1e-12relative tolerance. Thus optimizer refitting, not input drift, explains the failed strict comparisons. The original reproduction script is unchanged and still returns its strict assessment.

## Label correction and exclusions

An exploratory frame_summary.anchor_shrink list initially used UID without replay/EID, yielding6 entries where only2 fixed original anchors were intended. Final verification/frame_summary_corrected.json and frame_anchor_label_correction.json repair this presentation. Population calculations, capture timing anchors and final tables used exact identities and were unaffected. Original result files are preserved for provenance.

Two all-family implicit cases remain unconverged, both2026.10.08_12.23.06/EID2275330, native tick pairs20993→20996 and21043→21046. Their errors are included rather than silently removed. The5484R77 implicit cases converged. None of these checks is a complete production-simulator regression.

## Delivery

Original raw bundle still passes verification; dev and the original raw branch were not modified. Final107 files were size/SHA256-checked, plus rootMANIFEST.json gives108 package files.

Artifact name:missile-sim-ins-trk-mechanism-final-20261017
Artifact ID:12115135336
ZIP bytes:31728838
GitHub ZIP SHA256:e170efb0e8409c93138d23fe738e379e373feb599bc4e2513a7de15fcc3b7437

Run `python3 reproduce_reviewed.py /absolute/path/to/new-output` from the extracted delivery. Read REPORT.md and VERIFICATION.md first. The latter is also in the ZIP; this GitHub verification receipt is written after packaging, but all its numerical evidence is included in the package.
