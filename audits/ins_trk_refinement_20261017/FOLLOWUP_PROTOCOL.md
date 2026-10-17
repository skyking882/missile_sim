# Exploratory follow-up after refinement run39073430131

These are new, explicitly exploratory candidates proposed after seeing the first three tracks. They are not a preregistered first-pass or a newly blinded dataset.

1. First-capture coverage: apply the SAME conditional map P0.8(nnew)Qnew wold to all original501 captures, separately the114 whose interval includes phase1 and the387 whose interval does not. Restrict map scores to fresh endpoints and flag0 on both sides, report counts. Test Q-copy and the .8 map with both encoding hypotheses. Do not infer actual capture time from lowest residual or discard output failures. Compare new-Q and old-Q only as common rules. Inspect own displacement/time residual for the six original rate-cell incompatibilities, not a new sample exclusion.

2. Ordinary effective angular weight: gap-specific fits .843,.926,.952 suggested that T*(1-alpha) is approximately constant. Test one common lag parameter L using alpha(T)=1-L/T and alpha(T)=T/(T+L), alongside constant alpha and the previous implicit-velocity linearization. All fits use training scenes only; independent held scene is scored for direction, complete rate and Ps. L range0..0.01s. Current pointing remains a CONDITIONAL serialized input, not a verified raw radar measurement. Do not call fitted L a recovered physical time delay; these weights act on innovation, not absolute direction propagation.

3. Repeated-current-pointing safeguard: evaluate a fixed .30..1.00 alpha grid step.025 before bounded local refinement in the best neighboring bracket, to check that the previous bounded minimizer did not miss a better interior solution. The hypothesis still repeats the same endpoint ray, and must not be conflated with multiple changing hidden measurements.

4. Independent review: reimplement reset feasibility as a six-variable equality-constrained LP, separately from the earlier four-variable minimum-slack LP. Enumerate integer ticks explicitly to verify all48 phase counts. Confirm all data hashes and counts. Check corrected tracking-loss bookkeeping through code29 (the first phase summary's ticks_since_previous_loss only tracked direct31->25 and must not be interpreted for paths through29). Keep the original output for provenance and write a corrected diagnostic file.

The small lag candidates are not claims of WT source. No production files or native/raw inputs change. No AI subagents are available or claimed.
