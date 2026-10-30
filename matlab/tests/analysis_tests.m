function results=analysis_tests
r=missile.analysis.runResponse(struct('amplitude',1));assert(all(isfinite(r.output),'all'));assert(r.metrics.settled);assert(r.metrics.settlingTime<3);
p=missile.analysis.runResponse(struct('signal','pulse','amplitude',.5,'pulseWidth',2/48));assert(abs(p.inputArea(2)-1/48)<1e-12);
f=missile.analysis.runResponse(struct('freeFlight',true,'duration',.25));assert(all(isfinite(f.output),'all'));
m=missile.analysis.linearizeModel(struct);assert(m.G.Ts==1/48);assert(isequal(size(m.L),[2 2]));assert(m.finiteDifferenceCheck<.01);assert(m.base.trimResidual<1e-4);
a=missile.analysis.compareLinear(struct('amplitude',.2,'duration',2),false);assert(a.relativeError<.1,'Linear/nonlinear mismatch: %g',a.relativeError);
[y,t]=impulse(m.T,1);assert(all(isfinite(y),'all')&&numel(t)>2);[rr,kk]=rlocus(m.scalarLoop);assert(~isempty(rr)&&~isempty(kk));[gm,pm]=margin(m.scalarLoop);assert(~isnan(gm)&&~isnan(pm));
aOpen=missile.analysis.compareLinear(struct('closedLoop',false,'amplitude',.001,'duration',.5,'startTime',.1),false);assert(aOpen.relativeError<.15,'Open-loop small-signal mismatch: %g',aOpen.relativeError);
results=struct('openLoopOverlayRelativeError',aOpen.relativeError,'stepMetrics',r.metrics,'pulseArea',p.inputArea,'finiteDifferenceCheck',m.finiteDifferenceCheck,'linearOverlayRelativeError',a.relativeError,'gainMargin',gm,'phaseMargin',pm);
disp(results)
end
