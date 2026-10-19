# Retained scalar memory and #118's delayed second capture

Start with RESULTS.md and results/final/summary.json. This investigation separates three claims: (1) a saved scalar can be reconstructed; (2) using it as a candidate speed-gate center is compatible with recorded acceptance/rejection; (3) the game actually uses that acceptance algorithm. Only the first is strongly established here. The second is promising for#118 but not universal. The third remains unproven.

The source branch is audit/relock-gate-memory-20261019, pinned production parent ab93a013dbe5ed8010e3f925a4bcbc1fa83d6f8d. No production source, parameters, controller, PN, aerodynamics, raw WRPL or native parser changes. The investigation uses commercial-game replay records, not real-system measurements.

## Offline reproduction

```bash
python3 reproduce.py /absolute/path/to/new-output
```

Requires Python3.12, NumPy and SciPy (original numerical environment: NumPy1.26.4, SciPy1.11.4). No network calls. Existing output directories are rejected. It verifies the delivery manifest, reruns the full general gate comparison, six scalar expressions, independent moving-LOS review and raw-adjacent positive controls, then compares every saved JSON output. Float comparison tolerance absolute1e-8+relative1e-11 is a reproducibility tolerance, not an allowed physical gate error. Integer labels/text/structure must match exactly. Gzip timestamps are not used to score numerical equality.

input/ retains424held spells,36254fresh states and independent fixed-aircraft histories. MANIFEST and native-extraction receipts preserve raw indices andSHA. Input selection is based on the frozen501entity dataset, of which82are R-77-1; it is not every missile in the game. The raw scan actually read1,633,750rows and129,079R77states. No new WRPL decode is claimed. The compact entrypoint does not repeat that native scan.

adjacency/ stores verified per-entity previous raw indices for the accepted-fresh controls. The first review only used successivefresh entries separated by<=8ticks; that can bridge an interveningheld. Final checks use this independent adjacency map and report excluded pairs rather than silently describing them as uninterruptedtracking. Native extraction and adjacency can be rerun after restoring original native exports from run39055559128 and the fixed parser/reference from run39055471867; see workflows.

results/general/ contains fixed-range/speed/notch/angular candidate diagnostics. results/scalar/ preserves all six no-fit scalar laws, including rejected alternatives. results/review/ contains independent arithmetic, moving-LOS compensation alternatives and initial positive controls. results/final/ is authoritative for source-adjacentfresh control counts and the final moving-LOS gate checks. Receipts preserve completed workflow logs and source commits.

## Important interpretation limits

- Current saved pointing, independent target positions and owntrajectory are conditional inputs; no autonomous flight was run.
- Held prediction reads a single lastfresh n/w/closing/lastseed, not later gameclosing values. Initializing these four fields independently is still unresolved.
- The 20m/s game Dopplerwidth and300m/s SearchRange are different fields. Testing halfwidth10,20,300 does not prove which algorithm/width the game uses.
- Fixed2m² and15deg off-axis halfpower are comparison assumptions, not measuredaircraftRCS or uniquely identified beam semantics.
- A held label says noacceptedupdate was saved. It does not prove theecho existed or was explicitlyrejected by a particular gate.
- Fixed target_eid metadata is a reference association. Ps within50m is not proof of theinternalreflectorID; chaff/otherobject changes cannot beexcluded solelybydistance.
- Available countermeasure metadata has EID/owner/name but not livecloudcenters. Do not treat birthpositions as currentcloudpositions.
- A scalar_last that predictsvelocityslope is not necessarily true targetacceleration. It may be a filterderivative with errors left by capture.
- Same-data reruns and alternate scalar arithmetic are not independentAI subagents or newblind trials.
