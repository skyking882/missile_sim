# Follow-up discriminants (exploratory, not new blind validation)

Motivation from the completed four-track run39067986176: at phase-zero R77 capture endpoints the diagnostic transverse scale is near0.8, and replacing old Vs by current Vs changes the fitted effective direction weight. These findings motivated the tests below; do not label them preregistered before the first analysis or independent new data.

## Reset contraction

Define phase as first48m+1 after lastINS. Isolate endpoints that equal that phase, timer=0 and both flags0, without using rate residuals. At these endpoints there are no later updates under this explicit phase hypothesis. Compare fixed transverse retention kappa={1,0.8,1-0.926*h/(tau+h/2),1-h/(tau+h/2)}. Preserve the projected axial component. Candidates use Qold/Qnew and three prior rate sources: stored old rate, geometric rate from old Ps/Vs and own state, and old target PV extrapolated to current own state. Direction used for projection is separately labelled current n DIAGNOSTIC, current pointing ray CONDITIONAL, or old PV prediction. A fit of one common kappa uses four scenes, predicts fifth, and is compared with the fixed constants on identical samples. Current-n projection does not count as an independent direction prediction or full forward model. Also apply direct kappa=0.8 initialization at the fixed phase and propagate subsequent ordinary updates with the existing frozen tau/alpha; retain all eligible events and tails.

## Implicit Ps/Vs candidate

Do not read current Ps/Vs/n/w during prediction. Inputs: past Ps/n/w, current pointing angles/range/radial scalar and own state/time. The radial scalar and pointing angles themselves may be serialized outputs, so this remains conditional reconstruction, not autonomous radar measurement recovery.

At each step jointly solve:

p_ang=Rot(-w_old*T)n_old,
w(n)=w_old+c(p_ang,n)/(tau+T/2),
V(n)=vm-(vm·n)n+d*n+R*(n×w(n)),
p_PV(n)=unit(Ps_old+T*V(n)-p_own),
n=slerp(p_PV(n),pointing,alpha).

Use the previously saved per-test-scene alpha trained on the other four scenes in the current-Vs diagnostic; do not refit on held-out outputs. Fixed-point solver has50 iterations; report convergence rather than accepting a small fit if the equation was not solved. Compare exact same32189 all-family45deg and5484 R77<54.9deg pairs; no post-hoc residual selection. Score position/direction/rate/velocity, all failures, and a current-label poisoning check. Its purpose is to test whether the observed endpoint relations can be realized by a different causal ordering with comparable accuracy, not to certify this as WT's actual iteration.

No production model or original replay source is edited. Computation runs locally in the declared Actions job; separate programs are not subagents.
