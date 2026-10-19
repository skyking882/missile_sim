# Relock output contract and expiry flow — fixed audit

War Thunder game replay reconstruction. No new physical detector, gain, cooldown, phase adjustment, or production guidance change. Read archived raw records to distinguish supported observed state flow from speculative internal implementation.

## Main question
Can the observations be explained by a simple separation of retained angular state, available PV output and search scheduling, without inventing a distinct cooldown for code29?

## Source representation
Frozen capture_held/audit.py constructs code = type8*8 + flagA*4 + flagB*2 + flagC. For the accepted layouts type8=3, flagC=1:25=(A0,B0),29=(A1,B0),31=(A1,B1). These are a diagnostic combination of raw fields, NOT recovered game enum names. Test field/output associations, do not assign unproven meanings to A/B. A separate flag_after_timer0 must not be confused with header A/B.

## Predeclared read-only checks
1. Verify the prior relock bundle and rebuild unique transitions from its saved paths. Measure raw output changes at31→29 and29→25/31; state duration and expiry age. Current/future fields are diagnostic labels only, not simulated inputs.
2. Independently reread all five original native exports in the fixed501-entity cohort. Include all fresh→nonfresh episodes, including episodes that never return, first exits to29/25 and paths censored by end of data or timestamp ambiguity. This prevents inferring an expiry trigger solely from successful returns selected near1s.
3. For each31→29→next record and31→25 boundary, record rate/PV equality, timer advancement, direction continuation using the saved old rate, and simultaneous Ps/range/direction closure. Use no fitted gain. A PV-first/rate-later observation is an output contract, not proof of an internal callback order.
4. At expiry, compare refreshed Ps with a fixed independently recorded aircraft from prior target_eid metadata, current and previous packet timestamps, and with the last fresh PV propagated at its original Vs. Never select a new nearest target as truth. The fixed metadata may be stale or wrong after association changes; state that limitation. For #118/#148 use their previously fixed EIDs1596417 and1517569 respectively. Compare native position interpolation at packet time and state time as explicit diagnostics; neither is a guaranteed exact physical time match.
5. Check whether31→29 outputs systematically retain the old full rate and direction while PV changes, whether29 lasts one record or many, and whether exceptions invalidate a universal rule. Include code27 if present. Compare next observed branch without inventing a success probability or selecting a per-event best model.
6. For expiry episodes use observed last-fresh time, timer grids10/32767s and b5 scale1/48s. Record timer discontinuities and gap>8/nonpositive tick as censoring, not silently join them. Fresh eligibility is the original code31,timer0 observation convention, not proof of the same reflector. One-second expiry is tested, not used to force episode boundaries.

## Review and scope
Run separate archived-path and raw-stream implementations. Verify endpoint raw SHA against existing records; duplicate references must agree. Summaries must show exceptions and denominators. A second script or CI job is not an independent AI subagent. No such dispatcher is callable in this session. This study does not re-decode WRPL, rerun production flight, recover cloud positions or prove that default target metadata is current association. Any documented simplification remains a candidate until source/measurement provenance is known.
