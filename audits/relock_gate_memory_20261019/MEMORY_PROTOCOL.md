# Supplemental scalar-memory identification

First-pass run39259442338 found #118 at11532: ground radial measurement-535.56m/s versus held d=-534.684; instantaneous relative radial measurement-632.491 versus saved closing=659.200. The scalar_last seed is126.3466 and heldconstant. This suggests that reusing Ps/Vs geometry loses a separate extrapolated scalar state. This is a new hypothesis prompted by those data, not preregistered before the entire study.

Test six fixed expressions, no coefficient optimization:
A c0+last0*T
B c0+last0*T + n0 dot(vm(t)-vm0)
C n0 dot vm(t) -d0 +last0*T
D n_pred(T) dot vm(t) -d0 +last0*T
E n0 dot vm(t) -d0
F n_pred(T) dot vm(t) -d0

n0 is the lastfresh recorded LOS, n_pred is its held full-rate propagation; do not use current n/d/last/closing to produce predictions. Current missile velocity is an explicit allowed conditioning input. Test every strict held spell, also report spells with unchanged d and last. Same-time c output is a reconstruction label ONLY, never prove it is necessarily a Doppler acceptance gate.

Values d/c/last share the decoded scalar grid30000/32767. Report raw and midpoint-truncation-cell arithmetic as deterministic quantization alternatives, not calibrated offsets. Include an error envelope from d/last/output scalar cells and n/v quantization; this is a conservative first-order bound, not exact nonlinear feasibility. Do not change the physical gate widths to achieve results.

Then before reading updated fresh outputs, evaluate the corresponding predicted closing against independent fixed-aircraft radial velocity at held and early recovered frames. Compare fixed halfwidth10/20/300m/s. Review first false rejection of a real early return and preserve its target-input uncertainty. Pass means a necessary gate is compatible, not full autonomous recovery. Check #118 lastfresh→held→expiry and compare refreshed INS near second capture separately. No extra cooldown or speed relaxation constant.
