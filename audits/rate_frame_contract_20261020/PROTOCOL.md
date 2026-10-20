# Rate coordinates: direct frame-contract experiment

User hypothesis: INS rate is world-expressed, TRK rate body-expressed. Separate output coordinates from unobservable internal coordinates. This is WT game-record reconstruction, not real missile design. No coefficients or event-specific frames/signs/timing fitted. Production src/tests/config/data remain unchanged. No callable AI subagents; separated CI computations and independent numerical implementations must not be called AI reviewers.

## Inputs and known correction
Use completed run39281930196 artifact axial-rate-mechanism-20261019 (bounded episodes424, strict139, capture boundaries931, anchors). Verify bounded manifest. The prior analyze.py defined ray z=+cos(e)sin(a), whereas frozen native parser audit.py at567cd783342961f9c5aba9005ce55437457744bb defines z=-cos(e)sin(a). Thus prior actual_gimbal frame scores are invalid and superseded here. Prior world-rate held propagation, capture axial scores and world-up/body-up constructions do not call this ray and remain separately inspectable. No raw replay or parser changes.

## Independent lines of evidence
A. Coordinate contract: native ray=(cos(e)cos(a),sin(e),-cos(e)sin(a)); verify Q@ray vs saved INS direction and compare inverse-Q/incorrect-z controls. Verify fresh TRK direction vs normalized(Ps-p) both directly and as body coordinates Q@n. Test old INS full rate against cross(Vs-vm,Ps-p)/|Ps-p|² using raw w, Qw and Q.Tw. These are same-output algebraic diagnostics, not independent target truth or complete observer generation. Report all931 and R77 subsets, no error-based exclusions.
B. Held propagation: seed n,w once at lastfresh for the same139 strict held episodes. Use saved own Q(t) with piecewise geodesic interpolation. For each interval dt, Q1=Exp(Omega_W dt)Q0, constant Omega_B=Q0.T Omega_W. Compare:
  W: n1=Exp(-w dt)n0.
  B: w_raw is inertial angular-rate vector expressed in body; b1=Exp(-(w_raw+Omega_B)dt)Q0.T n0; n1=Q1 b1.
  R: w_raw is relative-to-body angular rate expressed in body; n1=Q1 Exp(-w_raw dt)Q0.T n0.
Convention n_dot=n×w. Repeated predictions are continuous and never reset to saved current n. Q derivatives are interval averages, not instantaneous observations. R has exact endpoint solution for any intermediate Q when raw rate constant. In addition retain projected-zero and flipped-axial controls in world frame. All groups plus predetermined body rotation path>=10deg and raw rate>=0.1deg/s. No selecting a preferred frame per event.
C. Numerical review: independently integrate W/B/R world equations with scalar Rodrigues Q(t) and RK4, compare against exact segment solutions at8/16substeps; synthetic zero-body-rotation and pure-body-relative-static tests; correctly co-transform n and w invariance. This verifies formulas, not WT semantics. Recompute native ray scores and flag old gimbal-sign results as superseded. No fresh outputs used to drive held models; current n only scores predictions.

## Interpretation
If W alone predicts the same world direction while B/R fail during strong own rotation, raw TRK rate behaves as a world-expressed state. This cannot exclude a hidden body-frame implementation with world-frame serialization. Joint rotation preserves n·w; coordinates alone cannot create lambda if both inputs are transformed consistently. Current Q applied to already world-expressed seed is not an ordinary W→B conversion; its physical source remains unproven. Compare published Q_new and Q_new.T axial diagnostics without calling the gain-independent scalar test a full-vector or autonomous initialization success.

Deliver exact code, per-event errors, source hashes, correction notice and actually completed run receipts. Retain failures and maximum errors. No complete PN/flight ablation is claimed.
