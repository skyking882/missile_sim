# Relock timing and state-path tests — 2026-10-19

Scope: reverse engineering serialized states of the War Thunder video game. Read-only replay analysis, no new radar performance/guidance model, no fitted gain, delay, threshold, hidden observation or production modification. Fixed 1s and global phase48m+1 are competing timing hypotheses, not facts to enforce.

## Questions
H1: Do all recoveries after positive-timer held require at least1s since the last accepted observation or since the first loss?
H2: Are all recoveries restricted to global tick%48==1, or only those passing through code25?
H3: Is the approximately1s timer a lifetime of retained output rather than a universal recovery wait? Code29 remains an unknown intermediate encoding, not renamed held without evidence.

## Evidence and grain
Use the completed capture-held-reviewed artifact run39063537273:931 fixed code25→31 boundaries,1035 code31 positive-timer→zero-timer adjacent pairs, full anchor histories. Match prior input hashes/counts; no claim of new measurements. Then independently reconstruct relevant state paths from the existing five Linux native exports (run39055559128) using the unchanged native-bit map from run39055471867 and unchanged audit.parse. No new compilation or WRPL decode is claimed. Cohort is the existing501 entities, selected independently of the new cooldown result. Repeated events within an entity/scene are not independent trials.

## Conservative clocks
h=1/48s is the existing native b5 time scale, not an inferred physical substep rate. Timer gridq=10/32767s. For adjacent held(a)→fresh(b), gap=(b.tick-a.tick)h, estimate observation interval=gap+a.timer-b.timer, report +/-2q as conservative timer-only uncertainty. Also report +/-2h extra state/output-phase slack. An interval with upper bound<1s excludes universal1s wait only under declared timestamp semantics; do not report midpoint as exact capture time.

More robust full episode: actual last preserved fresh record f, one or more held records, next preserved fresh b, all same code31 and no intervening unsupported/nonmonotonic/largegap records. Since true loss must occur after f within the known episode, b.tick-f.tick supplies a conservative upper bound on waiting since loss (include2h slack). Require at least2 held rows and max timer>=.1s as a predeclared sustained-held sensitivity subset. No need to assume timer reset equals exact measurement time to use the bounding record times. Report all events as well as stricter subsets.

## State paths
A: code31 timer>0→code31 timer0; stratify flag00/11/change, family, scene, held duration, packet/time agreement, PV and rate cache continuity. A state code alone does not establish reflector identity. Preserve designation_key but never treat it as target ID.
B: code29→31, with preceding code/path and timer retained.
C: code25→31, first vs subsequent using exact previously frozen event identifiers.
For each return, bracket possible recovery by the previous original same-entity observation and first fresh record. Test whether phase1 lies anywhere in that interval; also +/-2ticks sensitivity for serialization latency. A non-phase endpoint alone is NOT evidence against periodic timing.

Fixed durations bins <=.1,.1-.25,.25-.5,.5-.9,.9-1.1,>1.1s; phase histograms enumerate all48 values, without choosing a new phase. Record distributions and representative raw windows, including shortest sustained flag00 R77 example from each scene selected after scoring as counterexamples, not held-out forecasts. Flag same-EID != verified same reflector. Confirm no duplicate integer ticks silently overwritten.

## Checks
Cross-check exact1035 previously frozen fresh-return index/SHA endpoints and931 capture boundaries. Every source scan computes SHA256. Timer continuity tests use predeclared2q; report failures rather than forcing continuous episodes. Independent integer enumeration for phase inclusion. For full episodes, recompute duration from raw ticks and show packet elapsed as independent clock diagnostic. Do not force #118 to reject at2.4375s or accept at4.208333s.

Deliver actual completed results, scripts, bounded event windows and numerical check receipts. Tests of timing hypotheses do not validate current RCS/beam assumptions, current production observer, same-target recovery or complete autonomous trajectories. No independent AI-subagent claim when only scripts and CI are used.
