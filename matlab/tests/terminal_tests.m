function result=terminal_tests
% Targeted integration: state restore, sampled control, observer and full tangent.
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
P=missile.loadParameters('su_r_77_1');X=missile.initState(P,struct('height',5000,'speed',700,'time',2));
target=struct('position',[5000;5000;300],'velocity',[-200;0;0]);o=struct('duration',0.5,'observerTau',0.04);
S=missile.terminal.initialize(P,X,target,o);[a,ra]=missile.terminal.step(S,P,o);[b,rb]=missile.terminal.step(S,P,o);
assert(isequaln(a,b) && isequaln(ra,rb),'Snapshot restart must be deterministic');
assert(all(ra.controlConsumed==0) && norm(a.u)>0,'Stored-control causal ordering failed');
assert(abs(a.X.time-X.time-P.Ts)<1e-10,'Wrong sample interval');
z=missile.terminal.localCoordinates('encode',a,a);decoded=missile.terminal.localCoordinates('decode',z,a);
assert(norm(decoded.X.quaternion-a.X.quaternion)<1e-10 && norm(decoded.target.position-a.target.position)<1e-10,'Independent coordinates failed round trip');
missing=o;missing.absentIntervals=[0,100];[lost,rr]=missile.terminal.step(a,P,missing);
assert(~rr.observerValid && all(lost.u==0) && isequaln(lost.C,a.C),'Absent-observation gate must preserve PID memory');
r=missile.terminal.run(P,X,target,o);assert(all(isfinite([r.rows.range])) && isfinite(r.closestApproach));
L=missile.terminal.linearize(r.snapshots{min(10,numel(r.snapshots))},P,o);
assert(isequal(size(L.A),[26,26]) && all(isfinite(L.A(:))),'Full-loop matrix invalid');
assert(L.differenceRelative<0.1,'Jacobian does not converge across step sizes');
check=missile.terminal.checkLinearization(L,P,o);assert(check.passed,'Nonlinear small-perturbation response disagrees with tangent');
report=missile.terminal.analyze(r,[2,min(10,numel(r.snapshots))]);assert(numel(report.models)==2);
longOptions=o;longOptions.duration=8;full=missile.terminal.run(P,X,target,longOptions);
assert(strcmp(full.event,'capture'),'Synthetic terminal example must reach capture');
fullReport=missile.terminal.analyze(full);
assert(fullReport.excludedCaptureSnapshots>=1 && all(cellfun(@(q)q.passed,fullReport.perturbationChecks)),'Capture interval must be excluded from local tangent diagnostics');
result=struct('passed',true,'checks',10,'jacobianDifference',L.differenceRelative,'smallPerturbationRelativeError',check.relativeError,'closestApproach',r.closestApproach);
disp(result);
end
